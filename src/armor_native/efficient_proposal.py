"""Small native-pixel proposal CNN. PixelUnshuffle is an exact rearrangement, not RGB resampling."""
import torch
from torch import nn
from torch.nn import functional as F
class CBR(nn.Sequential):
 def __init__(self,a,b,k=1,stride=1,groups=1):
  super().__init__(nn.Conv2d(a,b,k,stride,k//2,groups=groups,bias=False),nn.BatchNorm2d(b,momentum=.03),nn.SiLU(inplace=False))
class DS(nn.Module):
 def __init__(self,a,b,stride=1):
  super().__init__();self.depth=CBR(a,a,5,stride,a);self.point=CBR(a,b)
 def forward(self,x):return self.point(self.depth(x))
class Block(nn.Module):
 def __init__(self,c):
  super().__init__();self.conv=DS(c,c)
 def forward(self,x):return x+self.conv(x)
class FastProposalNetwork(nn.Module):
 def __init__(self):
  super().__init__()
  self.stem=nn.Sequential(nn.PixelUnshuffle(4),CBR(48,24),Block(24))
  self.p8=nn.Sequential(DS(24,32,2),Block(32))
  self.p16=nn.Sequential(DS(32,48,2),Block(48))
  self.p32=nn.Sequential(DS(48,64,2),Block(64),Block(64))
  self.lateral8=CBR(32,24);self.lateral16=CBR(48,24);self.lateral32=CBR(64,24)
  self.fuse=nn.Sequential(CBR(96,32),Block(32))
  self.center=nn.Conv2d(32,1,1);self.geo=nn.Conv2d(32,11,1)
  nn.init.normal_(self.center.weight,std=.001);nn.init.constant_(self.center.bias,-4.6)
  nn.init.zeros_(self.geo.weight);nn.init.zeros_(self.geo.bias)
 def forward(self,x):
  a=self.stem(x);b=self.p8(a);c=self.p16(b);d=self.p32(c);size=a.shape[-2:]
  fused=self.fuse(torch.cat([a,F.interpolate(self.lateral8(b),size=size,mode="bilinear",align_corners=False),F.interpolate(self.lateral16(c),size=size,mode="bilinear",align_corners=False),F.interpolate(self.lateral32(d),size=size,mode="bilinear",align_corners=False)],1))
  return self.center(fused),self.geo(fused)
