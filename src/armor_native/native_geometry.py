"""Batched native TRF solves and integer lamp evidence; preserves reference gates."""
from pathlib import Path
import ctypes,numpy as np,cv2
from scipy.spatial.transform import Rotation
from types import SimpleNamespace
from .glyph import fit_glyph_pose as reference_fit
from .level_pose import LINK_TO_CV
from .observability import RULE,RULES
class NativeGeometry:
 """Opt-in equivalent C++ TRF and integer lamp support. Build per target host."""
 def __init__(self,library=None):
  path=Path(library) if library else Path(__file__).with_name("native")/"glyph_trf.so"
  if not path.is_file():
   raise FileNotFoundError("Build native geometry with scripts/build_native_geometry.py or pass its library path")
  self.library=ctypes.CDLL(str(path.resolve()))
  arr=np.ctypeslib.ndpointer(dtype=np.float64,flags="C_CONTIGUOUS")
  img=np.ctypeslib.ndpointer(dtype=np.uint8,flags="C_CONTIGUOUS")
  self.library.glyph_trf_many.argtypes=[ctypes.c_int,ctypes.c_int]+[arr]*7
  self.library.glyph_trf_many.restype=ctypes.c_int
  self.library.glyph_lamp.argtypes=[img,ctypes.c_int,ctypes.c_int,arr,arr,arr]
  self.library.glyph_lamp.restype=ctypes.c_int
  self.whole_pose=hasattr(self.library,"glyph_pose")
  if self.whole_pose:
   self.library.glyph_pose.argtypes=[ctypes.c_int]+[arr]*6
   self.library.glyph_pose.restype=ctypes.c_int
 def lamp_evidence(self,image,corners,rule=None):
  rule=RULE if rule is None else rule
  assert rule["version"] in RULES
  c=np.asarray(corners,float)
  if c.shape!=(4,2) or not np.isfinite(c).all():return {"accepted":False,"reason":"invalid_corners","support":[0.,0.]}
  config=np.array([rule["chroma_min"],rule["peak_min"],rule["support_min"],rule["minimum_length_px"],rule["minimum_separation_px"],rule.get("cyan_green_tolerance",0),rule.get("pixel_center_offset",0),rule["version"]!="two-lamp-v1"],dtype=np.float64)
  out=np.empty(9,np.float64);im=np.ascontiguousarray(image,dtype=np.uint8)
  self.library.glyph_lamp(im,im.shape[0],im.shape[1],np.ascontiguousarray(c),config,out)
  return {"rule_version":rule["version"],"accepted":bool(out[0]),"reason":["two_lamps_supported","degenerate_or_unresolved_quad","insufficient_two_lamp_image_evidence"][int(out[1])],"support":out[2:4].tolist(),"median_chroma":out[4:6].tolist(),"lengths_px":out[6:8].tolist(),"separation_px":float(out[8])}
 def fit_glyph_pose(self,points,probabilities,K,template,camera_q=None,mode="glyph_level"):
  if mode!="glyph_level" or camera_q is None:return reference_fit(points,probabilities,K,template,camera_q,mode)
  if self.whole_pose:
   pts=np.ascontiguousarray(points,dtype=np.float64);n=len(pts)
   if not 8<=n<=64:raise ValueError("Native glyph geometry expects 8 to 64 points")
   args=[np.ascontiguousarray(v,dtype=np.float64) for v in [probabilities,K,template,camera_q]]
   if pts.shape!=(n,2) or args[0].shape!=(n,) or args[1].shape!=(3,3) or args[2].shape!=(n,3) or args[3].shape!=(4,):raise ValueError("Invalid native glyph input shapes")
   out=np.empty(9+2*n,dtype=np.float64);status=self.library.glyph_pose(n,pts,*args,out)
   if status==0:return None
   if status!=1:raise RuntimeError("Native whole-pose failure "+str(status))
   pp=out[6:6+2*n].reshape(n,2)
   return {"xyz_m":out[:3].tolist(),"rotation_vector":out[3:6].tolist(),"corners":pp[:4].tolist(),"projected_landmarks":pp.tolist(),"reprojection_rms_px":float(out[6+2*n]),"used_points":int(out[8+2*n]),"cost":float(out[7+2*n]),"camera_attitude_used":True}
  points=np.asarray(points,float);template=np.asarray(template,float);K=np.asarray(K,float);probabilities=np.asarray(probabilities,float)
  valid=(probabilities>=.1)&np.isfinite(points).all(1)
  if valid.sum()<8 or valid[4:].sum()<4:return None
  observed=points[valid];xyz=template[valid];weights=np.sqrt(np.clip(probabilities[valid],.1,1.))
  starts=[]
  try:
   sol=cv2.solvePnPGeneric(template[:4],points[:4],K,np.zeros(5),flags=cv2.SOLVEPNP_IPPE)
   starts=[(rv.ravel(),t.ravel()) for rv,t in zip(sol[1],sol[2]) if np.isfinite(t).all() and t[2,0]>.05]
  except cv2.error:pass
  try:
   ok,rv,tv=cv2.solvePnP(xyz,observed,K,np.zeros(5),flags=cv2.SOLVEPNP_SQPNP)
   if ok and np.isfinite(tv).all() and tv[2,0]>.05:starts.append((rv.ravel(),tv.ravel()))
  except cv2.error:pass
  if not starts:return None
  q=np.asarray(camera_q,float);wc=LINK_TO_CV@Rotation.from_quat(q[[1,2,3,0]]).as_matrix().T
  mount=Rotation.from_euler("x",-15,degrees=True).as_matrix();all_mounted=template@mount.T
  initial=[]
  for rv,tv in starts:
   rw=wc.T@cv2.Rodrigues(rv)[0];initial.append(np.r_[tv,np.arctan2(rw[1,0],rw[0,0])])
  args=[np.ascontiguousarray(v,dtype=np.float64) for v in [all_mounted[valid],wc,observed,weights,[K[0,0],K[1,1],K[0,2],K[1,2]],initial]]
  output=np.empty((len(starts),7),np.float64);status=self.library.glyph_trf_many(len(observed),len(starts),*args,output)
  if status:raise RuntimeError("TRF numerical failure "+str(status))
  for idx in np.argsort(output[:,4],kind="stable"):
   x=output[idx,:4];tv=x[:3];c,s=np.cos(x[3]),np.sin(x[3]);rz=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
   camera=all_mounted@rz.T@wc.T+tv
   all_points=camera[:,:2]/camera[:,2:3]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
   if not np.isfinite(all_points).all() or not np.isfinite(tv).all() or tv[2]<=.05:continue
   rv=cv2.Rodrigues(wc@rz@mount)[0].ravel()
   error=float(np.sqrt(np.mean(((all_points[valid]-observed)*weights[:,None])**2)))
   return {"xyz_m":tv.tolist(),"rotation_vector":rv.tolist(),"corners":all_points[:4].tolist(),"projected_landmarks":all_points.tolist(),"reprojection_rms_px":error,"used_points":int(valid.sum()),"cost":float(output[idx,4]),"camera_attitude_used":True}
  return None
