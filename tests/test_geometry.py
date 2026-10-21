import unittest
import cv2,numpy as np,torch
from armor_native.geometry import crop_native,corner_targets,decode_points,solve_geometry,local_camera,OBJECT_POINTS
from armor_native.model import RoiPrecision,precision_loss
class GeometryTests(unittest.TestCase):
 def test_pixel_copy_and_zero_padding(self):
  image=np.random.default_rng(12).integers(0,256,(180,220,3),np.uint8)
  for anchor in [[100.7,90.2],[0,0],[-8,170],[223,-9]]:
   patch,origin,valid=crop_native(image,anchor);yy,xx=np.nonzero(valid);np.testing.assert_array_equal(patch[yy,xx],image[yy+origin[1],xx+origin[0]]);self.assertFalse(patch[~valid].any())
 def test_fractional_pixel_decode(self):
  corners=np.array([[20.25,90.7],[20.4,30.6],[85.8,30.2],[86.3,90.9]]);hm,off,mask=corner_targets(corners);offset=torch.logit(torch.tensor(off).clamp(1e-7,1-1e-7))[None];decoded=decode_points(torch.tensor(hm)[None],offset)[0].numpy();np.testing.assert_allclose(decoded,corners,atol=1e-5)
 def test_camera_translation_preserves_pose(self):
  K=np.array([[1303.675283386667,0,720],[0,1303.675283386667,540],[0,0,1.]]);points,_=cv2.projectPoints(OBJECT_POINTS,np.array([.1,.2,.3]),np.array([.2,-.1,4.]),K,np.zeros(5));c=points.reshape(4,2);origin=np.array([640,470]);a=solve_geometry(c,K);b=solve_geometry(c-origin,local_camera(K,origin));self.assertIsNotNone(a);self.assertIsNotNone(b);np.testing.assert_allclose(a['xyz_m'],b['xyz_m'],atol=1e-7)
 def test_training_graph_has_finite_gradients(self):
  torch.set_num_threads(4);torch.manual_seed(1);hm,off,mask=corner_targets([[40.2,80.7],[40.2,50.6],[90.3,50.4],[90.5,80.2]]);batch={'heatmap':torch.tensor(hm)[None],'offset':torch.tensor(off)[None],'point_mask':torch.tensor(mask)[None]};model=RoiPrecision();loss,_=precision_loss(model(torch.rand(1,3,128,128)),batch);loss.backward();self.assertTrue(torch.isfinite(loss));self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
if __name__=='__main__':unittest.main()
