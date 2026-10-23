"""Joint subpixel readout of frozen native corner neighborhoods."""
import torch
from torch import nn

class JointCornerResidual(nn.Module):
    def __init__(self, channels, radius=2, bound_px=4.0):
        super().__init__()
        self.radius=radius;self.bound_px=bound_px
        width=2*radius+1
        self.corner_token=nn.Sequential(nn.Linear(channels*width*width+2,128),nn.SiLU())
        self.joint=nn.Sequential(nn.Linear(4*128+8,256),nn.SiLU(),nn.Linear(256,8))
        nn.init.zeros_(self.joint[-1].weight);nn.init.zeros_(self.joint[-1].bias)
    def forward(self, features, base_corners, anchors):
        n,k,_=base_corners.shape
        _,channels,h,w=features.shape
        # Integer feature selection preserves the native image grid.
        centers=base_corners.detach().floor().long()
        yy,xx=torch.meshgrid(torch.arange(-self.radius,self.radius+1,device=features.device),torch.arange(-self.radius,self.radius+1,device=features.device),indexing='ij')
        x=(centers[...,0,None]+xx.flatten()).clamp(0,w-1)
        y=(centers[...,1,None]+yy.flatten()).clamp(0,h-1)
        ids=(y*w+x).reshape(n*k,1,-1).expand(-1,channels,-1)
        local=features.flatten(2).gather(2,ids).reshape(n,k,-1).float()
        fraction=base_corners.detach()-centers.float()
        token=self.corner_token(torch.cat([local,fraction],-1))
        global_points=base_corners.detach()+anchors.floor()-64
        relative=(global_points-global_points.mean(1,keepdim=True))/128
        joint_input=torch.cat([token.flatten(1),relative.flatten(1)],1)
        return self.bound_px*self.joint(joint_input).float().reshape(n,4,2).tanh()
