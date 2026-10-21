"""Native variable-size ROI localizer and physical-corner geometry.

RGB is only integer cropped/padded. Tensor interpolation occurs on features only.
The old128 model and its checkpoint remain unchanged.
"""
import numpy as np,torch,cv2
from torch.nn import functional as F
from .model import RoiPrecision

class NativeBucketPrecision(RoiPrecision):
 def __init__(self):super().__init__('spatial')
 def forward(self,image):
  n,c,h,w=image.shape
  if c!=3 or h!=w or h%4:raise ValueError('native square RGB crop, multiple of4 required')
  if h==128:coordinates=self.roi_coordinates
  else:
   yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=image.device),torch.linspace(-1,1,w,device=image.device),indexing='ij');coordinates=torch.stack([xx,yy])[None]
  native=self.native(torch.cat([image,coordinates.expand(n,-1,-1,-1)],1));context=self.context(native);features=self.fuse(torch.cat([native,F.interpolate(context,size=(h,w),mode='bilinear',align_corners=False)],1));return {'heatmap':self.heatmap(features),'offset':self.offset(features)}

def solve_physical(corners,K,template):
 try:
  sol=cv2.solvePnPGeneric(np.asarray(template,np.float64),np.asarray(corners,np.float64),np.asarray(K,np.float64),np.zeros(5),flags=cv2.SOLVEPNP_IPPE)
  valid=[(float(e[0]),np.asarray(t).ravel(),rv) for t,rv,e in zip(sol[2],sol[1],sol[3]) if np.isfinite(t).all() and t[2,0]>0]
  if not valid:return None
  e,t,rv=min(valid,key=lambda x:x[0]);return {'xyz_m':t.tolist(),'reprojection_rms_px':e,'rotation_vector':rv.ravel().tolist()}
 except cv2.error:return None

def statistics(values):
 a=np.asarray(values,float)
 if not len(a):return {'n':0}
 return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),'rmse':float(np.sqrt(np.mean(a*a))),'p90':float(np.quantile(a,.9)),'p95':float(np.quantile(a,.95)),'p99':float(np.quantile(a,.99)),'max':float(a.max())}
