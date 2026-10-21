"""Frozen YOLO11 feature-based proposals. Ultralytics components: AGPL-3.0."""
import json
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F

class ProposalNetwork(nn.Module):
 def __init__(self,config):
  super().__init__()
  from ultralytics.nn.tasks import PoseModel
  self.backbone=PoseModel(cfg=config,ch=3,nc=config['nc'],verbose=False);self.feature_index=self.backbone.model[-1].f[0];channels=self.backbone.model[-1].cv2[0][0].conv.in_channels
  self.project=nn.Conv2d(channels,64,1);self.rgb=nn.Sequential(nn.Conv2d(3,24,3,2,1),nn.GroupNorm(4,24),nn.SiLU(),nn.Conv2d(24,32,3,2,1),nn.GroupNorm(4,32),nn.SiLU());self.shared=nn.Sequential(nn.Conv2d(96,64,3,1,1),nn.GroupNorm(8,64),nn.SiLU(),nn.Conv2d(64,64,3,1,1),nn.GroupNorm(8,64),nn.SiLU());self.center=nn.Conv2d(64,1,1);self.geo=nn.Conv2d(64,11,1);self.register_buffer('mean',torch.tensor([123.675,116.28,103.53])[None,:,None,None]);self.register_buffer('std',torch.tensor([58.395,57.12,57.375])[None,:,None,None])
 def forward(self,x):
  assert x.shape[-2:]==(448,448);z=x;cache=[]
  for m in self.backbone.model[:-1]:
   if m.f!=-1:z=cache[m.f] if isinstance(m.f,int) else [z if j==-1 else cache[j] for j in m.f]
   z=m(z);cache.append(z if m.i in self.backbone.save else None)
  feat=F.interpolate(self.project(cache[self.feature_index]),size=(112,112),mode='bilinear',align_corners=False);h=self.shared(torch.cat([feat,self.rgb(x)],1));return self.center(h),self.geo(h)

class NativeProposer:
 def __init__(self,weights,config,device='cuda'):
  self.device=torch.device(device);self.model=ProposalNetwork(json.loads(Path(config).read_text())).to(self.device);self.model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True)['model']);self.model.eval()
 @torch.inference_mode()
 def __call__(self,image,threshold=.5):
  if image.shape!=(1080,1440,3):raise ValueError('Validated proposal grid requires1440x1080 RGB; supply external centers for other image sizes')
  pp=[]
  for y in [0,336,632]:
   for x in [0,336,672,992]:
    tile=image[y:y+448,x:x+448];t=torch.from_numpy(tile.transpose(2,0,1).copy())[None].float().to(self.device)/255
    with torch.autocast('cuda',dtype=torch.float16,enabled=self.device.type=='cuda'):hm,geo=self.model(t)
    prob=hm.float().sigmoid();peaks=(prob==F.max_pool2d(prob,3,1,1))&(prob>.05);indices=peaks[0,0].nonzero();scores=prob[0,0][peaks[0,0]]
    if len(scores)>12:scores,ii=torch.topk(scores,12);indices=indices[ii]
    positions=indices.cpu().numpy();ss=scores.cpu().numpy();g=geo[0].float().cpu().numpy()
    for (iy,ix),score in zip(positions,ss):
     v=g[:,iy,ix];fraction=1/(1+np.exp(-np.clip(v[:2],-30,30)));center=(np.array([ix,iy])+fraction)*4+[x,y];pp.append({'score':float(score),'center':center.tolist()})
  kept=[]
  for p in sorted(pp,key=lambda p:p['score'],reverse=True):
   if all(np.linalg.norm(np.array(p['center'])-q['center'])>10 for q in kept):kept.append(p)
  return [p for p in kept if p['score']>=threshold]
