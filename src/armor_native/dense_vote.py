"""Consistent local endpoint vectors for native-pixel subpixel decoding."""
import torch
from torch.nn import functional as F

def dense_vote_loss(vectors, corners, radius=6, sigma=2.0):
    """Each neighboring pixel learns the displacement to the same exact point.

    Labels are used only here, during training. The inference window is centered
    on the image-derived peak, independently of this ground-truth neighborhood.
    """
    n,k,_=corners.shape;h,w=vectors.shape[-2:]
    yy,xx=torch.meshgrid(torch.arange(-radius,radius+1,device=vectors.device),torch.arange(-radius,radius+1,device=vectors.device),indexing='ij')
    center=corners.detach().floor().long()
    x=center[...,0,None]+xx.flatten();y=center[...,1,None]+yy.flatten()
    valid=(x>=0)&(x<w)&(y>=0)&(y<h)
    ids=(y.clamp(0,h-1)*w+x.clamp(0,w-1))[:,:,None].expand(-1,-1,2,-1)
    observed=vectors.float().reshape(n,k,2,h*w).gather(3,ids)
    target=corners[...,None]-torch.stack([x,y],2).float()
    weights=torch.exp(-target.square().sum(2)/(2*sigma*sigma))*valid
    per_coordinate=F.smooth_l1_loss(observed,target,reduction='none',beta=.25)
    return ((per_coordinate*weights[:,:,None]).sum((2,3))/(2*weights.sum(-1).clamp_min(1e-12))).mean()
