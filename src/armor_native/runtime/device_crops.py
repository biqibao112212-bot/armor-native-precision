import torch
def native_device_crops(frame,origins,size,out=None):
 """Integer crop/copy/zero-pad of already normalized native CHW pixels; no resampling."""
 if frame.ndim!=3 or frame.shape[0]!=3 or frame.dtype!=torch.float32:raise ValueError("normalized float32 native CHW required")
 shape=(len(origins),3,size,size)
 if out is None:out=torch.empty(shape,dtype=frame.dtype,device=frame.device)
 elif tuple(out.shape)!=shape or out.device!=frame.device or out.dtype!=frame.dtype:raise ValueError("crop output mismatch")
 out.zero_();h,w=frame.shape[1:]
 for i,origin in enumerate(origins):
  x,y=map(int,origin);x0=max(x,0);y0=max(y,0);x1=min(x+size,w);y1=min(y+size,h)
  if x1>x0 and y1>y0:out[i,:,y0-y:y1-y,x0-x:x1-x].copy_(frame[:,y0:y1,x0:x1])
 return out
