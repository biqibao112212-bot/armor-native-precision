import numpy as np
import torch,cv2
from scipy.spatial.transform import Rotation
from armor_native.glyph import NativeGlyphPrecision,fit_glyph_pose,translation_sensitivity,point_targets
from armor_native.plate import NativePlatePrecision
from armor_native.level_pose import LINK_TO_CV

def template():
    lamp=np.array([[-.0669,.0027,-.027],[-.0669,.0027,.027],[.0669,.0027,.027],[.0669,.0027,-.027]])
    glyph=np.array([[x,-.00003,z] for x,z in [(-.03,-.05),(0,-.05),(.03,-.04),(.03,-.02),(0,-.01),(-.02,0),(.03,.01),(.03,.03),(.02,.05),(0,.05),(-.03,.04),(-.03,.02)]])
    return np.r_[lamp,glyph]

def scene():
    K=np.array([[1303.,0,720],[0,1303.,540],[0,0,1.]])
    camera=Rotation.from_euler('zyx',[.3,-.15,0]).as_quat()
    q=camera[[3,0,1,2]];wc=LINK_TO_CV@Rotation.from_quat(camera).as_matrix().T
    R=wc@Rotation.from_euler('z',1.7).as_matrix()@Rotation.from_euler('x',-15,degrees=True).as_matrix()
    rv=cv2.Rodrigues(R)[0].ravel();tv=np.array([.4,.1,4.])
    points=cv2.projectPoints(template(),rv,tv,K,np.zeros(5))[0].reshape(-1,2)
    return K,q,rv,tv,points

def test_exact_sensor_pose_and_projection():
    K,q,rv,tv,p=scene();pose=fit_glyph_pose(p,np.full(16,.9),K,template(),q)
    assert pose is not None
    assert np.linalg.norm(np.array(pose['xyz_m'])-tv)<1e-5
    assert np.max(np.abs(np.array(pose['corners'])-p[:4]))<1e-3
    assert pose['used_points']==16

def test_robust_pose_with_one_contour_outlier():
    K,q,rv,tv,p=scene();p[8]+=[5.,-3.]
    pose=fit_glyph_pose(p,np.full(16,.9),K,template(),q)
    assert pose is not None and np.linalg.norm(np.array(pose['xyz_m'])-tv)<.02

def test_translation_jacobian_matches_projection():
    K,q,rv,tv,p=scene();J=translation_sensitivity(template(),rv,tv,K,q,np.ones(16,bool))
    for axis in range(3):
        delta=np.eye(3)[axis]*1e-5
        v=cv2.projectPoints(template(),rv,tv+delta,K,np.zeros(5))[0].reshape(-1,2)
        np.testing.assert_allclose(J@(v-p).ravel(),delta,atol=5e-8)

def test_transfer_preserves_physical_heads():
    torch.manual_seed(2);base=NativePlatePrecision((2,4,8)).eval();model=NativeGlyphPrecision().eval();model.initialize_from_plate(base.state_dict())
    with torch.inference_mode():
        x=torch.rand(1,3,128,128);a=base(x);b=model(x)
        torch.testing.assert_close(a['corners'],b['corners'][:,:4],atol=1e-5,rtol=0)
        torch.testing.assert_close(a['heatmap'],b['heatmap'][:,:4],atol=1e-6,rtol=0)

def test_invisible_point_has_no_positive_or_offset():
    p=np.array([[10.25,12.75],[90.,-3.]])
    h,o,m=point_targets(p,[True,False],32)
    assert h[0,12,10]==1 and m.sum()==1 and h[1].sum()==0
    np.testing.assert_allclose(o[:2,12,10],[.25,.75])
