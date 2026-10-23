"""Efficient native-pixel glyph head. Unshuffle/shuffle rearrange pixels without interpolation."""
import torch
from torch import nn
from torch.nn import functional as F
from armor_native.quad import continuous_points

class CG(nn.Sequential):
 def __init__(self,a,b,k=1,stride=1,groups=1,dilation=1):
  super().__init__(nn.Conv2d(a,b,k,stride,dilation*(k//2),dilation=dilation,groups=groups,bias=False),nn.GroupNorm(8,b),nn.SiLU())
class Residual(nn.Module):
 def __init__(self,c,dilation=1):
  super().__init__();self.body=nn.Sequential(CG(c,c,5,groups=c,dilation=dilation),nn.Conv2d(c,c,1,bias=False),nn.GroupNorm(8,c))
 def forward(self,x):return F.silu(x+self.body(x))
class FastGlyphPrecision(nn.Module):
 def __init__(self,point_count=16):
  super().__init__();self.point_count=point_count
  self.native=nn.Sequential(nn.PixelUnshuffle(4),CG(80,48),Residual(48),Residual(48))
  self.context=nn.Sequential(CG(48,64,3,2),Residual(64),CG(64,64,3,2),Residual(64,2),Residual(64,4))
  self.fuse=nn.Sequential(CG(112,64),Residual(64),Residual(64))
  self.heatmap=nn.Conv2d(64,point_count*16,1);self.offset=nn.Conv2d(64,point_count*32,1)
  nn.init.normal_(self.heatmap.weight,std=.001);nn.init.constant_(self.heatmap.bias,-4.6)
  nn.init.zeros_(self.offset.weight);nn.init.zeros_(self.offset.bias)
  for size in [128,192,256,384,512]:
   yy,xx=torch.meshgrid(torch.linspace(-1,1,size),torch.linspace(-1,1,size),indexing="ij")
   self.register_buffer("coordinates_"+str(size),torch.stack([xx,yy])[None],persistent=False)
 def maps(self,image):
  n,c,h,w=image.shape
  xy=getattr(self,"coordinates_"+str(h)).expand(n,-1,-1,-1)
  a=self.native(torch.cat([image,xy],1));ctx=self.context(a)
  z=self.fuse(torch.cat([a,F.interpolate(ctx,size=a.shape[-2:],mode="bilinear",align_corners=False)],1))
  return F.pixel_shuffle(self.heatmap(z),4),F.pixel_shuffle(self.offset(z),4)
 def forward(self,image):
  hm,off=self.maps(image)
  return {"heatmap":hm,"offset":off,"corners":continuous_points(hm,off)}
