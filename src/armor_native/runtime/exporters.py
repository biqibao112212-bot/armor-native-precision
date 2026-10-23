from pathlib import Path
import onnx
import torch,numpy as np
from torch import nn
from torch.nn import functional as F
class FineExport(nn.Module):
 def __init__(self,model):super().__init__();self.model=model
 def forward(self,image):
  b=self.model;n,c,h,w=image.shape
  if h==128:coordinates=b.roi_coordinates
  else:
   yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=image.device),torch.linspace(-1,1,w,device=image.device),indexing="ij")
   coordinates=torch.stack([xx,yy])[None]
  native=b.native(torch.cat([image,coordinates.expand(n,-1,-1,-1)],1))
  context=b.wide_context(b.context(native))
  feat=b.fuse(torch.cat([native,F.interpolate(context,size=(h,w),mode="bilinear",align_corners=False)],1))
  hm=b.heatmap(feat);off=b.offset(feat);k=hm.shape[1]
  peak=hm.flatten(2).argmax(-1);cy,cx=peak//w,peak%w
  dy,dx=torch.meshgrid(torch.arange(-3,4,device=image.device),torch.arange(-3,4,device=image.device),indexing="ij")
  xx=(cx[:,:,None]+dx.flatten()).clamp(0,w-1);yy=(cy[:,:,None]+dy.flatten()).clamp(0,h-1);ids=yy*w+xx
  logits=hm.float().flatten(2).gather(2,ids);weights=logits.softmax(-1)
  offset=off.float().reshape(n,k,2,h*w).gather(3,ids[:,:,None].expand(-1,-1,2,-1)).sigmoid()
  coords=torch.stack([xx,yy],2).float()+offset
  points=(weights[:,:,None]*coords).sum(-1)
  return {"corners":points,"heatmap":hm.float().flatten(2).max(-1).values[:,:,None,None]}
class CoarseExport(nn.Module):
 def __init__(self,model):super().__init__();self.model=model
 def forward(self,x):
  hm,geo=self.model(x);prob=hm.float().sigmoid();mask=(prob==F.max_pool2d(prob,3,1,1))&(prob>.05)
  scores,ids=torch.topk(torch.where(mask,prob,torch.full_like(prob,-1.)).flatten(1),12,dim=1)
  g=geo.float().flatten(2).gather(2,ids[:,None].expand(-1,11,-1)).transpose(1,2)
  center=(torch.stack([ids%360,ids//360],-1).float()+g[:,:,:2].clamp(-30,30).sigmoid())*4.
  corners=g[:,:,2:10].reshape(-1,12,4,2)*32.+center[:,:,None]
  return {"scores":scores,"centers":center,"corners":corners}

class FastGlyphExport(nn.Module):
 def __init__(self,model):super().__init__();self.model=model
 def forward(self,image):
  hm,off=self.model.maps(image);n,k,h,w=hm.shape
  peak=hm.flatten(2).argmax(-1);cy,cx=peak//w,peak%w
  dy,dx=torch.meshgrid(torch.arange(-3,4,device=image.device),torch.arange(-3,4,device=image.device),indexing="ij")
  xx=(cx[:,:,None]+dx.flatten()).clamp(0,w-1);yy=(cy[:,:,None]+dy.flatten()).clamp(0,h-1);ids=yy*w+xx
  logits=hm.float().flatten(2).gather(2,ids);weights=logits.softmax(-1)
  offset=off.float().reshape(n,k,2,h*w).gather(3,ids[:,:,None].expand(-1,-1,2,-1)).sigmoid()
  points=(weights[:,:,None]*(torch.stack([xx,yy],2).float()+offset)).sum(-1)
  return {"corners":points,"heatmap":hm.float().flatten(2).max(-1).values[:,:,None,None]}

class SizedGlyphMaps(nn.Module):
 def __init__(self,base,size):
  super().__init__();self.base=base;self.register_buffer("coordinates",getattr(base,"coordinates_"+str(size)))
 def maps(self,x):
  b=self.base;n=x.shape[0];a=b.native(torch.cat([x,self.coordinates.expand(n,-1,-1,-1)],1));ctx=b.context(a)
  z=b.fuse(torch.cat([a,F.interpolate(ctx,size=a.shape[-2:],mode="bilinear",align_corners=False)],1))
  return F.pixel_shuffle(b.heatmap(z),4),F.pixel_shuffle(b.offset(z),4)
class FineTuple(nn.Module):
 def __init__(self,base,size):super().__init__();self.m=FastGlyphExport(SizedGlyphMaps(base,size))
 def forward(self,x):
  d=self.m(x);return d["corners"],d["heatmap"].flatten(2)[:,:,0]
class CoarseTuple(nn.Module):
 def __init__(self,base):super().__init__();self.m=CoarseExport(base)
 def forward(self,x):
  d=self.m(x);return d["scores"],d["centers"],d["corners"]

def repair_norm(model):
 inferred=onnx.shape_inference.infer_shapes(model)
 types={v.name:v.type.tensor_type.elem_type for v in list(inferred.graph.input)+list(inferred.graph.value_info)+list(inferred.graph.output)}
 types.update({v.name:v.data_type for v in model.graph.initializer})
 nodes=[];count=0
 for node in model.graph.node:
  if node.op_type=="InstanceNormalization" and types.get(node.input[0])==onnx.TensorProto.FLOAT16 and types.get(node.input[1])==onnx.TensorProto.FLOAT:
   name=node.output[0];inp=node.input[0]+"_normfp32_"+str(count);temp=name+"_normfp32"
   nodes.append(onnx.helper.make_node("Cast",[node.input[0]],[inp],to=onnx.TensorProto.FLOAT))
   node.input[0]=inp;node.output[0]=temp;nodes.append(node)
   nodes.append(onnx.helper.make_node("Cast",[temp],[name],to=onnx.TensorProto.FLOAT16));count+=1
  else:nodes.append(node)
 del model.graph.node[:];model.graph.node.extend(nodes)
 onnx.checker.check_model(model,full_check=True)
 return model,count

class SmallTuple(nn.Module):
 def __init__(self,m):super().__init__();self.m=FineExport(m)
 def forward(self,x):
  d=self.m(x);return d["corners"],d["heatmap"].flatten(2)[:,:,0]
