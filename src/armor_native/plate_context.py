"""Identity-initialized dilated feature context; RGB sampling is unchanged."""
import torch
from torch import nn

class DilatedContext(nn.Module):
    def __init__(self,channels,dilation):
        super().__init__()
        self.branch=nn.Sequential(nn.Conv2d(channels,channels,3,padding=dilation,dilation=dilation,groups=channels,bias=False),nn.GroupNorm(8,channels),nn.SiLU(),nn.Conv2d(channels,channels,1))
        nn.init.zeros_(self.branch[-1].weight);nn.init.zeros_(self.branch[-1].bias)
    def forward(self,x):
        return x+self.branch(x)

def context_stack(dilations):
    assert all(isinstance(d,int) and d>=1 for d in dilations)
    return nn.Sequential(*(DilatedContext(64,d) for d in dilations))
