"""Image-only level-body pose fit using calibrated camera attitude.

Camera attitude is a same-exposure public sensor input, not target truth. The
plate mount is fixed at -15deg around its local right axis in the asset frame.
This module never consumes world target position, target yaw, slot or distance.
"""
import numpy as np, cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

LINK_TO_CV=np.array([[0.,-1,0],[0,0,-1],[1,0,0]])

def fit_level_pose(corners,K,template,camera_quaternion_wxyz,initial=None):
    corners=np.asarray(corners,float);K=np.asarray(K,float);template=np.asarray(template,float)
    q=np.asarray(camera_quaternion_wxyz,float)
    rc=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();world_to_cv=LINK_TO_CV@rc.T
    mount=Rotation.from_euler('x',-15,degrees=True).as_matrix()
    if initial is None:
        sol=cv2.solvePnPGeneric(template,corners,K,np.zeros(5),flags=cv2.SOLVEPNP_IPPE)
        starts=[(rv,t.ravel()) for rv,t in zip(sol[1],sol[2]) if t[2,0]>.05]
    else:starts=[(np.array(initial['rotation_vector']),np.array(initial['xyz_m']))]
    if not starts:return None
    rotated=template@mount.T
    def project(x):
        c,s=np.cos(x[3]),np.sin(x[3]);rz=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        xyz=rotated@rz.T@world_to_cv.T+x[:3]
        uv=xyz@K.T
        return uv[:,:2]/uv[:,2:3],xyz
    def residual(x):return (project(x)[0]-corners).ravel()
    results=[]
    for rv,t in starts:
        rw=rc@LINK_TO_CV.T@cv2.Rodrigues(np.asarray(rv))[0]
        yaw=np.arctan2(rw[1,0],rw[0,0]);x=np.r_[t,yaw]
        fit=least_squares(residual,x,method='lm',max_nfev=50,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        uv,xyz=project(fit.x)
        if not np.isfinite(fit.x).all() or np.min(xyz[:,2])<=.05:continue
        rmse=float(np.sqrt(np.mean((uv-corners)**2)))
        results.append({'xyz_m':fit.x[:3].tolist(),'corners':uv.tolist(),'yaw_world_rad':float(fit.x[3]),'reprojection_rms_px':rmse,'nfev':fit.nfev})
    return min(results,key=lambda r:r['reprojection_rms_px']) if results else None
