"""Candidate RGB ROI localizer with native-resolution features.

This is a small custom convolutional model, not an implementation or pretrained
checkpoint of Lite-HRNet. Feature interpolation never resizes input RGB pixels.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .geometry import SIZE,decode_points

class Conv(nn.Sequential):
    def __init__(self,cin,cout,stride=1):
        super().__init__(nn.Conv2d(cin,cout,3,stride,1,bias=False),nn.GroupNorm(8,cout),nn.SiLU())

class Residual(nn.Module):
    def __init__(self,c):
        super().__init__();self.layers=nn.Sequential(nn.Conv2d(c,c,3,1,1,groups=c,bias=False),nn.GroupNorm(8,c),nn.SiLU(),nn.Conv2d(c,c,1,bias=False),nn.GroupNorm(8,c))
    def forward(self,x):return F.silu(x+self.layers(x))

class RoiPrecision(nn.Module):
    def __init__(self,head='spatial'):
        super().__init__();assert head in ['spatial','center_regression'];self.head=head
        yy,xx=torch.meshgrid(torch.linspace(-1,1,SIZE),torch.linspace(-1,1,SIZE),indexing='ij');self.register_buffer('roi_coordinates',torch.stack([xx,yy])[None])
        self.native=nn.Sequential(Conv(5,32),Residual(32),Residual(32))
        self.context=nn.Sequential(Conv(32,48,2),Residual(48),Conv(48,64,2),Residual(64),Residual(64))
        self.fuse=nn.Sequential(nn.Conv2d(96,48,1,bias=False),nn.GroupNorm(8,48),nn.SiLU(),Residual(48),Residual(48))
        if head=='spatial':
            self.heatmap=nn.Conv2d(48,4,1);self.offset=nn.Conv2d(48,8,1);nn.init.constant_(self.heatmap.bias,-4.6);nn.init.zeros_(self.offset.weight);nn.init.zeros_(self.offset.bias)
        else:
            self.regression=nn.Sequential(nn.Linear(48,128),nn.SiLU(),nn.Linear(128,8));nn.init.zeros_(self.regression[-1].weight);nn.init.zeros_(self.regression[-1].bias)
    def forward(self,image):
        assert image.shape[1:]==(3,SIZE,SIZE)
        native=self.native(torch.cat([image,self.roi_coordinates.expand(image.shape[0],-1,-1,-1)],1));context=self.context(native);features=self.fuse(torch.cat([native,F.interpolate(context,size=(SIZE,SIZE),mode='bilinear',align_corners=False)],1))
        if self.head=='spatial':return {'heatmap':self.heatmap(features),'offset':self.offset(features)}
        return {'corners':SIZE/2+32*self.regression(features[:,:,SIZE//2,SIZE//2]).reshape(-1,4,2)}
    def points(self,prediction):
        return decode_points(prediction['heatmap'].float(),prediction['offset'].float()) if self.head=='spatial' else prediction['corners'].float()

def precision_loss(prediction,batch):
    if 'corners' in prediction:
        loss=F.smooth_l1_loss((prediction['corners'].float()-SIZE/2)/32,(batch['corners'].float()-SIZE/2)/32)
        return loss,{'coordinate':float(loss.detach())}
    hm=batch['heatmap'];mask=batch['point_mask'];n=mask.sum().clamp_min(1);p=prediction['heatmap'].float().sigmoid().clamp(1e-6,1-1e-6)
    focal=-(torch.log(p)*(1-p)**2*(hm==1)+torch.log(1-p)*p**2*(1-hm)**4*(hm<1)).sum()/n
    mask2=mask.repeat_interleave(2,dim=1);off=(F.smooth_l1_loss(prediction['offset'].float().sigmoid(),batch['offset'],reduction='none')*mask2).sum()/n
    return focal+off,{'heatmap':float(focal.detach()),'offset':float(off.detach())}
