"""Four native crops with optional cross-corner feature conditioning.

Input coordinates are image-based proposal corners. No pose, range, or slot is
an online input. Geometry sensitivity is used only as a TRAIN loss.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .broad import NativeBucketPrecision, solve_physical
from .geometry import crop_native
from .observability import lamp_evidence

def continuous_points(heatmap, offset, radius=3, offsets_are_vectors=False):
    """Local differentiable expectation around a detached peak; native pixels."""
    n,k,h,w=heatmap.shape
    peak=heatmap.detach().flatten(2).argmax(-1)
    cy,cx=peak//w,peak%w
    dy,dx=torch.meshgrid(torch.arange(-radius,radius+1,device=heatmap.device),torch.arange(-radius,radius+1,device=heatmap.device),indexing='ij')
    xx=(cx[:,:,None]+dx.flatten()).clamp(0,w-1)
    yy=(cy[:,:,None]+dy.flatten()).clamp(0,h-1)
    ids=yy*w+xx
    logits=heatmap.float().flatten(2).gather(2,ids)
    weights=logits.softmax(-1)
    off=offset.float().reshape(n,k,2,h*w)
    if not offsets_are_vectors:off=off.sigmoid()
    off=off.gather(3,ids[:,:,None].expand(-1,-1,2,-1))
    coords=torch.stack([xx,yy],2).float()+off
    return (weights[:,:,None]*coords).sum(-1)

def gaussian_coordinate_points(heatmap, radius=4):
    """Continuous coordinates from a local spatial distribution, no cell offset."""
    n,k,h,w=heatmap.shape
    peak=heatmap.detach().flatten(2).argmax(-1)
    cy,cx=peak//w,peak%w
    dy,dx=torch.meshgrid(torch.arange(-radius,radius+1,device=heatmap.device),torch.arange(-radius,radius+1,device=heatmap.device),indexing='ij')
    xx=(cx[:,:,None]+dx.flatten()).clamp(0,w-1)
    yy=(cy[:,:,None]+dy.flatten()).clamp(0,h-1)
    logits=heatmap.float().flatten(2).gather(2,yy*w+xx)
    weights=logits.softmax(-1)
    return (weights[:,:,None]*torch.stack([xx,yy],2).float()).sum(-1)

def gaussian_coordinate_loss(logits, corners, sigma=1.25):
    """KL supervision centered at exact floating-point labels on the native grid."""
    h,w=logits.shape[-2:]
    yy,xx=torch.meshgrid(torch.arange(h,device=logits.device,dtype=torch.float32),torch.arange(w,device=logits.device,dtype=torch.float32),indexing='ij')
    d2=(xx-corners[...,0,None,None]).square()+(yy-corners[...,1,None,None]).square()
    target_logits=(-d2/(2*sigma*sigma)).flatten(2)
    logq=target_logits.log_softmax(-1)
    logp=logits.float().flatten(2).log_softmax(-1)
    return (logq.exp()*(logq-logp)).sum(-1).mean()

class NativeQuadPrecision(nn.Module):
    def __init__(self,joint=True,coordinate_mode="legacy",width_multiplier=1,refinement_mode="none"):
        super().__init__()
        assert width_multiplier in {1,2}
        self.width_multiplier=width_multiplier
        self.backbone=NativeBucketPrecision()
        if width_multiplier!=1:
            from .model import Conv,Residual
            b=self.backbone;w=width_multiplier
            b.native=nn.Sequential(Conv(5,32*w),Residual(32*w),Residual(32*w))
            b.context=nn.Sequential(Conv(32*w,48*w,2),Residual(48*w),Conv(48*w,64*w,2),Residual(64*w),Residual(64*w))
            b.fuse=nn.Sequential(nn.Conv2d(96*w,48*w,1,bias=False),nn.GroupNorm(8,48*w),nn.SiLU(),Residual(48*w),Residual(48*w))
            b.heatmap=nn.Conv2d(48*w,4,1);b.offset=nn.Conv2d(48*w,8,1)
        assert refinement_mode in {"none","joint_residual","joint_raw"}
        self.refinement_mode=refinement_mode
        if refinement_mode in {"joint_residual","joint_raw"}:
            assert coordinate_mode=="legacy"
            from .residual import JointCornerResidual
            self.refiner=JointCornerResidual(48*width_multiplier) if refinement_mode=="joint_residual" else JointCornerResidual(3,radius=8)
        self.joint=joint
        assert coordinate_mode in {"legacy","continuous_gaussian","dense_votes"}
        self.coordinate_mode=coordinate_mode
        if coordinate_mode=="dense_votes":
            self.dense_head=nn.Conv2d(48*width_multiplier,8,1)
            nn.init.zeros_(self.dense_head.weight);nn.init.zeros_(self.dense_head.bias)
        if coordinate_mode=="continuous_gaussian":
            self.coordinate_head=nn.Conv2d(48*width_multiplier,4,1)
            self.coordinate_head.load_state_dict(self.backbone.heatmap.state_dict())
        if joint:
            self.condition=nn.Sequential(nn.Linear(4*64*width_multiplier+8,128),nn.SiLU(),nn.Linear(128,4*96*width_multiplier))
            nn.init.zeros_(self.condition[-1].weight)
            nn.init.zeros_(self.condition[-1].bias)
    def forward(self,images,anchors):
        n,k,c,h,w=images.shape
        assert (k,c,h,w)==(4,3,128,128)
        b=self.backbone;x=images.reshape(n*4,3,h,w)
        native=b.native(torch.cat([x,b.roi_coordinates.expand(n*4,-1,-1,-1)],1))
        context=b.context(native)
        features=b.fuse(torch.cat([native,F.interpolate(context,size=(h,w),mode='bilinear',align_corners=False)],1))
        if self.joint:
            relative=(anchors-anchors.mean(1,keepdim=True))/128
            pooled=context.float().mean((-2,-1)).reshape(n,4*64*self.width_multiplier)
            modulation=self.condition(torch.cat([pooled,relative.flatten(1)],1)).reshape(n*4,2,48*self.width_multiplier,1,1)
            features=features*(1+.1*modulation[:,0].tanh())+.1*modulation[:,1]
        hm=b.heatmap(features).reshape(n,4,4,h,w)
        off=b.offset(features).reshape(n,4,4,2,h,w)
        ids=torch.arange(4,device=x.device)
        heatmap=hm[:,ids,ids];offset=off[:,ids,ids].reshape(n,8,h,w)
        result={'heatmap':heatmap,'offset':offset,'corners':continuous_points(heatmap,offset)}
        if self.coordinate_mode=="dense_votes":
            residual=self.dense_head(features).reshape(n,4,4,2,h,w)[:,ids,ids].reshape(n,8,h,w).float()
            vectors=offset.float().sigmoid()+residual
            result.update(vote_vectors=vectors,corners=continuous_points(heatmap,vectors,offsets_are_vectors=True))
        if self.coordinate_mode=="continuous_gaussian":
            logits=self.coordinate_head(features).reshape(n,4,4,h,w)[:,ids,ids]
            result.update(coordinate_heatmap=logits,corners=gaussian_coordinate_points(logits))
        if self.refinement_mode in {"joint_residual","joint_raw"}:
            base=result["corners"]
            correction=self.refiner(features if self.refinement_mode=="joint_residual" else x,base,anchors)
            result.update(base_corners=base,corner_correction=correction,corners=base+correction)
        return result
    @staticmethod
    def points(prediction):return prediction['corners']

class QuadLocalizer:
    def __init__(self,weights,device='cuda'):
        self.device=torch.device(device)
        ck=torch.load(weights,map_location='cpu',weights_only=False)
        assert ck['head']=='native_quad_geometry'
        self.model=NativeQuadPrecision(ck['config']['joint'],ck['config'].get('coordinate_mode','legacy'),ck['config'].get('width_multiplier',1),ck['config'].get('refinement_mode','none')).to(self.device).eval()
        self.model.load_state_dict(ck['model'])
        self.template=ck['geometry'];self.config=ck['inference_config']
        self.proposal_threshold=self.config['proposal_threshold']
    @torch.inference_mode()
    def __call__(self,image,proposals,K):
        if not proposals:return []
        patches=[];origins=[];anchors=[]
        for p in proposals:
            aa=np.asarray(p['coarse_corners'],float)
            pp=[];oo=[]
            for a in aa:
                crop,origin,_=crop_native(image,a,128);pp.append(crop);oo.append(origin)
            patches.append(pp);origins.append(oo);anchors.append(aa)
        x=torch.from_numpy(np.asarray(patches).transpose(0,1,4,2,3).copy()).float().to(self.device)/255
        anchor=torch.as_tensor(np.asarray(anchors),dtype=torch.float32,device=self.device)
        with torch.autocast('cuda',dtype=torch.float16,enabled=self.device.type=='cuda'):pred=self.model(x,anchor)
        local=pred['corners'].cpu().numpy();corners=local+np.asarray(origins)
        peaks=pred['heatmap'].float().sigmoid().flatten(2).max(-1).values.cpu().numpy()
        result=[]
        for p,c,l,peak in zip(proposals,corners,local,peaks):
            evidence=lamp_evidence(image,c,rule=self.config.get('observability_rule'));pose=solve_physical(c,K,self.template)
            raw=None if pose is None else pose['xyz_m']
            boundary=bool((l<4).any() or (l>124).any() or (c<0).any() or (c[:,0]>=image.shape[1]).any() or (c[:,1]>=image.shape[0]).any())
            accepted=raw is not None and evidence['accepted'] and not boundary and min(peak)>=self.config['minimum_corner_probability']
            reason='accepted' if accepted else ('crop_boundary' if boundary else ('low_corner_probability' if min(peak)<self.config['minimum_corner_probability'] else evidence['reason']))
            result.append({'score':p['score'],'proposal_center':p['center'],'center':((c.min(0)+c.max(0))/2).tolist(),'corners':c.tolist(),'native_size':128,'xyz_m':raw if accepted else None,'raw_xyz_m':raw,'image_evidence':dict(evidence,accepted=bool(accepted),reason=reason),'corner_peak_probabilities':peak.tolist(),'reprojection_rms_px':None if pose is None else pose['reprojection_rms_px']})
        accepted=[]
        for q in sorted(result,key=lambda q:q['score'],reverse=True):
            if q['xyz_m'] is None:continue
            if any(np.linalg.norm(np.asarray(q['corners'])-v,axis=1).max()<3 for v in accepted):
                q['xyz_m']=None;q['image_evidence']=dict(q['image_evidence'],accepted=False,reason='duplicate_after_refinement')
            else:accepted.append(np.asarray(q['corners']))
        return result
