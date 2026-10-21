"""Native variable-size ROI localizer and physical-corner geometry.

RGB is only integer cropped/padded. Tensor interpolation occurs on features only.
The old128 model and its checkpoint remain unchanged.
"""
import numpy as np,torch,cv2
from torch.nn import functional as F
from .model import RoiPrecision
from .observability import lamp_evidence

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


NATIVE_BUCKETS=(128,192,256,384,512,768,1024,1536)

class BroadNativeLocalizer:
 """RGB and image-based coarse extents -> native physical corners + calibrated pose."""
 def __init__(self,weights,device='cuda'):
  self.device=torch.device(device);ck=torch.load(weights,map_location='cpu',weights_only=False)
  if ck.get('head') not in ['native_bucket_physical','global_bucket_physical'] or 'geometry' not in ck:raise ValueError('requires native_bucket_physical weights with frozen training template')
  self.template=ck['geometry'];self.model=(NativeGlobalPrecision() if ck['head']=='global_bucket_physical' else NativeBucketPrecision()).to(self.device).eval();self.model.load_state_dict(ck['model'])
 @torch.inference_mode()
 def __call__(self,image,proposals,K):
  from .geometry import crop_native
  result=[]
  for p in proposals:
   if 'extent_px' not in p:raise ValueError('image-based coarse extent is required; no ground-truth range routing')
   extent=float(p['extent_px'])
   if not np.isfinite(extent) or extent<=0:continue
   size=next((s for s in NATIVE_BUCKETS if s>=extent+40),None)
   if size is None:
    result.append({'score':p.get('score',1.),'proposal_center':p['center'],'xyz_m':None,'reason':'native_extent_exceeds_supported_bucket'});continue
   patch,origin,_=crop_native(image,p['center'],size);x=torch.from_numpy(patch.transpose(2,0,1).copy())[None].float().to(self.device)/255
   with torch.autocast('cuda',dtype=torch.float16,enabled=self.device.type=='cuda'):pred=self.model(x)
   c=self.model.points(pred).cpu().numpy()[0]+origin;pose=solve_physical(c,K,self.template);evidence=lamp_evidence(image,c);raw_xyz=None if pose is None else pose['xyz_m']
   result.append({'score':p.get('score',1.),'proposal_center':p['center'],'native_size':size,'corners':c.tolist(),'corner_order':['physical_left_bottom','physical_left_top','physical_right_top','physical_right_bottom'],'corner_peak_probabilities':pred['heatmap'].float().sigmoid().flatten(2).max(-1).values[0].cpu().tolist(),'xyz_m':raw_xyz if evidence['accepted'] else None,'raw_xyz_m':raw_xyz,'image_evidence':evidence,'reprojection_rms_px':None if pose is None else pose['reprojection_rms_px']})
  return result


class NativeGlobalPrecision(NativeBucketPrecision):
 """Whole-ROI feature context supplies a coarse semantic prior to native peaks."""
 def __init__(self):
  super().__init__()
  from torch import nn
  from .model import Conv,Residual
  self.global_context=nn.Sequential(Conv(64,64),Residual(64),Residual(64))
  self.global_heatmap=nn.Conv2d(64,4,1);nn.init.zeros_(self.global_heatmap.weight);nn.init.zeros_(self.global_heatmap.bias)
 def forward(self,image):
  n,c,h,w=image.shape
  if c!=3 or h!=w or h%4:raise ValueError('native square RGB crop, multiple of4 required')
  if h==128:coordinates=self.roi_coordinates
  else:
   yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=image.device),torch.linspace(-1,1,w,device=image.device),indexing='ij');coordinates=torch.stack([xx,yy])[None]
  native=self.native(torch.cat([image,coordinates.expand(n,-1,-1,-1)],1));context=self.context(native);features=self.fuse(torch.cat([native,F.interpolate(context,size=(h,w),mode='bilinear',align_corners=False)],1));coarse=self.global_heatmap(self.global_context(F.adaptive_avg_pool2d(context,(8,8))))
  return {'heatmap':self.heatmap(features)+F.interpolate(coarse,size=(h,w),mode='bilinear',align_corners=False),'offset':self.offset(features),'coarse_heatmap':coarse}

def coarse_context_loss(logits,corners,native_size):
 """Supervise all4 coarse physical identities; image pixels never resampled."""
 yy,xx=torch.meshgrid(torch.arange(8,device=logits.device)+.5,torch.arange(8,device=logits.device)+.5,indexing='ij');grid=torch.stack([xx,yy],-1)
 centers=corners.float()/native_size*8;distance=(grid[None,None]-centers[:,:,None,None]).square().sum(-1);target=torch.exp(-distance/(2*.75**2)).flatten(2);target=target/target.sum(-1,keepdim=True).clamp_min(1e-8)
 return -(target*F.log_softmax(logits.float().flatten(2),dim=-1)).sum(-1).mean()
