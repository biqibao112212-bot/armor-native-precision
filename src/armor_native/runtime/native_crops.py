from pathlib import Path
import ctypes,numpy as np
P=ctypes.POINTER(ctypes.c_double)
class NativeCropSizes:
 def __init__(self,library):
  self.lib=ctypes.CDLL(str(Path(library).resolve()));self.fn=self.lib.glyph_crop_sizes
  self.fn.argtypes=[ctypes.c_int,ctypes.c_int,P,P,P,ctypes.POINTER(ctypes.c_int)];self.fn.restype=ctypes.c_int
 def __call__(self,proposals,K,template):
  if not proposals:return []
  c=np.ascontiguousarray([p["coarse_corners"] for p in proposals],np.float64);k=np.ascontiguousarray(K,np.float64);t=np.ascontiguousarray(template,np.float64);out=np.empty(len(c),np.int32)
  if c.shape!=(len(c),4,2) or k.shape!=(3,3) or t.ndim!=2 or t.shape[1]!=3:raise ValueError("invalid geometry")
  if self.fn(len(c),len(t),c.ctypes.data_as(P),k.ctypes.data_as(P),t.ctypes.data_as(P),out.ctypes.data_as(ctypes.POINTER(ctypes.c_int))):raise ValueError("native crop error")
  return [None if s<0 else int(s) for s in out]
