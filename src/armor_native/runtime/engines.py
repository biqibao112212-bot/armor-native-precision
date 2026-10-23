from pathlib import Path
import numpy as np,torch
import tensorrt as trt
class Engine:
 def __init__(self,path):
  self.logger=trt.Logger(trt.Logger.ERROR);self.runtime=trt.Runtime(self.logger)
  self.engine=self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
  if self.engine is None:raise RuntimeError("deserialize failed")
  self.context=self.engine.create_execution_context()
  self.names=[self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
  self.outputs=[n for n in self.names if self.engine.get_tensor_mode(n)==trt.TensorIOMode.OUTPUT]
  self.cache={};self.shape=None
 def __call__(self,x):
  x=x.contiguous();shape=tuple(x.shape)
  if shape!=self.shape:
   assert self.context.set_input_shape("rgb",shape);self.shape=shape
  if shape not in self.cache:
   self.cache[shape]={n:torch.empty(tuple(self.context.get_tensor_shape(n)),dtype=torch.from_numpy(np.empty((),dtype=trt.nptype(self.engine.get_tensor_dtype(n)))).dtype,device=x.device) for n in self.outputs}
  out=self.cache[shape]
  self.context.set_tensor_address("rgb",x.data_ptr())
  for n,y in out.items():self.context.set_tensor_address(n,y.data_ptr())
  if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):raise RuntimeError("TRT enqueue failed")
  return out

class CoarseTRTGraph:
 def __init__(self,path):
  self.model=Engine(path);self.input=torch.zeros((1,3,1088,1440),device="cuda",dtype=torch.float32)
  self.raw=torch.zeros((1080,1440,3),device="cuda",dtype=torch.uint8)
  with torch.inference_mode():
   self._forward();self._forward();torch.cuda.synchronize()
   self.stream=torch.cuda.Stream();self.graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(self.graph,stream=self.stream):self.packed=self._forward()
   torch.cuda.synchronize()
 def _forward(self):
  self.input[0,:,:1080].copy_(self.raw.permute(2,0,1));self.input[0,:,:1080].div_(255)
  out=self.model(self.input)
  return torch.cat([out["scores"][:,:,None],out["centers"],out["corners"].flatten(2)],-1)[0]
 @torch.inference_mode()
 def __call__(self,image,threshold=.5,include_geometry=False):
  if image.shape!=(1080,1440,3):raise ValueError("1440x1080 native RGB required")
  self.raw.copy_(torch.from_numpy(np.ascontiguousarray(image)));self.graph.replay()
  packed=self.packed.cpu().numpy();pp=[]
  for v in packed:
   score=float(v[0]);center=v[1:3].astype(float)
   if score<.05 or not(0<=center[0]<1440 and 0<=center[1]<1080):continue
   p={"score":score,"center":center.tolist()}
   if include_geometry:
    corners=v[3:].reshape(4,2).astype(float)
    p.update(coarse_corners=corners.tolist(),extent_px=float(np.max(np.ptp(corners,axis=0))))
   pp.append(p)
  kept=[]
  for p in sorted(pp,key=lambda p:p["score"],reverse=True):
   if all(np.linalg.norm(np.array(p["center"])-q["center"])>10 for q in kept):kept.append(p)
  return [p for p in kept if p["score"]>=threshold]

class SharedPackedGraphEngine:
 def __init__(self,path):
  self.base=Engine(path);self.cache={}
  self.base.context=None
  self.workspace=torch.empty(max(256,self.base.engine.device_memory_size_v2),device="cuda",dtype=torch.uint8)
  assert self.workspace.data_ptr()%256==0
 @torch.inference_mode()
 def __call__(self,x):
  shape=tuple(x.shape)
  if shape not in self.cache:
   ctx=self.base.engine.create_execution_context_without_device_memory()
   ctx.device_memory=self.workspace.data_ptr()
   assert ctx.set_input_shape("rgb",shape)
   buf=torch.empty(shape,device=x.device,dtype=x.dtype);buf.copy_(x)
   out={n:torch.empty(tuple(ctx.get_tensor_shape(n)),device=x.device,dtype=torch.from_numpy(np.empty((),dtype=trt.nptype(self.base.engine.get_tensor_dtype(n)))).dtype) for n in self.base.outputs}
   ctx.set_tensor_address("rgb",buf.data_ptr())
   for n,y in out.items():ctx.set_tensor_address(n,y.data_ptr())
   stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(stream):
    for _ in range(2):assert ctx.execute_async_v3(stream.cuda_stream)
   torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph,stream=stream):
    assert ctx.execute_async_v3(stream.cuda_stream)
    out["packed"]=torch.cat([out["points"],out["peak_logits"].float().sigmoid()[:,:,None]],-1)
   torch.cuda.synchronize()
   self.cache[shape]=(ctx,buf,out,graph,stream)
  _,buf,out,graph,_=self.cache[shape];buf.copy_(x);graph.replay();return out

class SharedPackedFineTRTGraph:
 def __init__(self,paths):
  self.models={s:SharedPackedGraphEngine(p) for s,p in paths.items()}
 def __call__(self,x):return {"packed":self.models[x.shape[-1]](x)["packed"]}
