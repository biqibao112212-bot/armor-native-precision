import torch
class QuadGraph:
 def __init__(self,model):self.model=model;self.cache={}
 @torch.inference_mode()
 def __call__(self,x,anchors):
  shape=tuple(x.shape)
  if shape not in self.cache:
   xx=torch.empty_like(x);aa=torch.empty_like(anchors);xx.copy_(x);aa.copy_(anchors)
   stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(stream),torch.autocast("cuda",dtype=torch.float16,cache_enabled=False):
    for _ in range(3):self.model(xx,aa)
   torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph,stream=stream),torch.autocast("cuda",dtype=torch.float16,cache_enabled=False):
    raw=self.model(xx,aa);out={"corners":raw["corners"],"heatmap":raw["heatmap"].float().flatten(2).max(-1).values[:,:,None,None]}
   torch.cuda.synchronize();self.cache[shape]=(xx,aa,out,graph,stream)
  xx,aa,out,graph,_=self.cache[shape];xx.copy_(x);aa.copy_(anchors);graph.replay();return out

import torch
class ChunkedQuadGraph:
 """Bound cached fallback batch sizes; preserve every candidate, no dropping."""
 def __init__(self,model,chunk_size=4):
  self.graph=QuadGraph(model);self.chunk_size=chunk_size
 @torch.inference_mode()
 def __call__(self,x,anchors):
  if len(x)<=self.chunk_size:return self.graph(x,anchors)
  result=[]
  for start in range(0,len(x),self.chunk_size):
   out=self.graph(x[start:start+self.chunk_size],anchors[start:start+self.chunk_size])
   result.append({k:v.clone() for k,v in out.items()})
  return {k:torch.cat([r[k] for r in result],0) for k in result[0]}
