"""Joint native plate features with a dense planar-coordinate auxiliary head.

RGB is integer cropped/padded only. Dense UV labels derive from TRAIN physical
corners; online dense fits consume only predicted UV and foreground confidence.
"""
import cv2, numpy as np, torch
from torch import nn
from torch.nn import functional as F
from .model import Conv
from .broad import NativeBucketPrecision,NATIVE_BUCKETS,solve_physical
from .quad import continuous_points,QuadLocalizer
from .geometry import crop_native
from .observability import lamp_evidence

UV_CORNERS=np.array([[0.,1.],[0.,0.],[1.,0.],[1.,1.]],np.float32)

class NativePlatePrecision(NativeBucketPrecision):
    def __init__(self,context_dilations=()):
        super().__init__()
        from .plate_context import context_stack
        self.wide_context=context_stack(context_dilations)
        self.dense=nn.Sequential(Conv(48,48),nn.Conv2d(48,3,1))
        nn.init.zeros_(self.dense[-1].weight)
        with torch.no_grad():self.dense[-1].bias.copy_(torch.tensor([.5,.5,-4.6]))
    def forward(self,image):
        n,c,h,w=image.shape
        if c!=3 or h!=w or h%4:raise ValueError('native RGB square required')
        if h==128:coordinates=self.roi_coordinates
        else:
            yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=image.device),torch.linspace(-1,1,w,device=image.device),indexing='ij');coordinates=torch.stack([xx,yy])[None]
        native=self.native(torch.cat([image,coordinates.expand(n,-1,-1,-1)],1));context=self.wide_context(self.context(native))
        features=self.fuse(torch.cat([native,F.interpolate(context,size=(h,w),mode='bilinear',align_corners=False)],1))
        hm=self.heatmap(features);off=self.offset(features);dense=self.dense(features)
        return {'heatmap':hm,'offset':off,'corners':continuous_points(hm,off),'uv':dense[:,:2].float(),'foreground':dense[:,2:3].float()}

def dense_targets(corners,size):
    H=cv2.getPerspectiveTransform(np.asarray(corners,np.float32),UV_CORNERS)
    yy,xx=np.mgrid[:size,:size];grid=np.stack([xx,yy,np.ones_like(xx)],-1).astype(float)
    mapped=grid@H.T;uv=mapped[...,:2]/mapped[...,2:3]
    mask=((uv>=0)&(uv<=1)).all(-1)
    uv=np.nan_to_num(uv,nan=0,posinf=0,neginf=0)
    uv=np.clip(uv,-2,3)
    return uv.transpose(2,0,1).astype(np.float32),mask[None].astype(np.float32)

def fit_dense_quad(uv,foreground,corner_hint):
    h,w=foreground.shape;yy,xx=np.mgrid[:h,:w]
    mask=(foreground>.5)&np.isfinite(uv).all(0)&(uv[0]>-.02)&(uv[0]<1.02)&(uv[1]>-.02)&(uv[1]<1.02)
    # A predicted-corner neighborhood prevents unrelated plates sharing a crop.
    roi=np.zeros((h,w),np.uint8);hull=cv2.convexHull(np.rint(corner_hint).astype(np.int32));cv2.fillConvexPoly(roi,hull,1)
    roi=cv2.dilate(roi,np.ones((5,5),np.uint8));mask &= roi.astype(bool)
    if mask.sum()<16:return None
    U,V=uv[:,mask].astype(float);X=(xx[mask]-(w-1)/2)/w;Y=(yy[mask]-(h-1)/2)/h
    wt=np.sqrt(foreground[mask].astype(float));one=np.ones_like(U);zero=np.zeros_like(U)
    a=np.stack([U,V,one,zero,zero,zero,-X*U,-X*V],1);b=np.stack([zero,zero,zero,U,V,one,-Y*U,-Y*V],1)
    A=np.concatenate([a*wt[:,None],b*wt[:,None]]);B=np.r_[X*wt,Y*wt]
    if len(U)>2048:
        ids=np.linspace(0,len(U)-1,2048).astype(int);A=np.concatenate([a[ids]*wt[ids,None],b[ids]*wt[ids,None]]);B=np.r_[X[ids]*wt[ids],Y[ids]*wt[ids]]
    solution,_,rank,s=np.linalg.lstsq(A,B,rcond=1e-7)
    if rank<8 or s[0]/s[-1]>1e5:return None
    H=np.r_[solution,1.].reshape(3,3);q=np.c_[UV_CORNERS,np.ones(4)]@H.T
    if np.min(abs(q[:,2]))<1e-5:return None
    q=q[:,:2]/q[:,2:3];q=q*[w,h]+[(w-1)/2,(h-1)/2]
    if not np.isfinite(q).all():return None
    return q

class PlateLocalizer:
    def __init__(self,weights,device='cuda'):
        self.device=torch.device(device);ck=torch.load(weights,map_location='cpu',weights_only=False)
        assert ck['head']=='native_dense_plate'
        self.model=NativePlatePrecision(tuple(ck['config'].get('context_dilations',()))).to(self.device).eval();self.model.load_state_dict(ck['model'])
        self.template=ck['geometry'];self.config=ck['inference_config'];self.proposal_threshold=self.config['proposal_threshold'];self.decoder=self.config.get('decoder','corners')
        self.fallback=QuadLocalizer(self.config['fallback_weights'],device) if self.config.get('fallback_weights') else None
    @torch.inference_mode()
    def __call__(self,image,proposals,K):
        results=[];groups={};large=[]
        for p in proposals:
            extent=float(p['extent_px']);size=next((s for s in NATIVE_BUCKETS if s>=extent+48),None)
            if size is None or size>512:large.append(p);continue
            patch,origin,_=crop_native(image,p['center'],size);groups.setdefault(size,[]).append((p,patch,origin))
        for size,items in groups.items():
            # Bound memory without changing pixels or proposal order.
            maximum=max(1,262144//(size*size))
            for start in range(0,len(items),maximum):
                block=items[start:start+maximum];x=torch.from_numpy(np.stack([z[1] for z in block]).transpose(0,3,1,2).copy()).float().to(self.device)/255
                with torch.autocast('cuda',dtype=torch.float16,enabled=self.device.type=='cuda'):pred=self.model(x)
                local=pred['corners'].cpu().numpy();peaks=pred['heatmap'].float().sigmoid().flatten(2).max(-1).values.cpu().numpy()
                uv=pred['uv'].cpu().numpy();fg=pred['foreground'].sigmoid().cpu().numpy()[:,0]
                for (p,patch,origin),c,peak,u,v in zip(block,local,peaks,uv,fg):
                    dense=None;source='corner_heatmap'
                    if self.decoder in {'dense','blend'}:
                        dense=fit_dense_quad(u,v,c)
                        if dense is not None and np.max(np.linalg.norm(dense-c,axis=1))<6:
                            c=dense if self.decoder=='dense' else .5*(dense+c);source=self.decoder
                    corners=c+origin;pose=solve_physical(corners,K,self.template);evidence=lamp_evidence(image,corners,rule=self.config.get('observability_rule'))
                    boundary=bool((c<4).any() or (c>size-4).any() or (corners<0).any() or (corners[:,0]>=image.shape[1]).any() or (corners[:,1]>=image.shape[0]).any())
                    good=pose is not None and evidence['accepted'] and not boundary and min(peak)>=self.config['minimum_corner_probability']
                    reason='accepted' if good else ('crop_boundary' if boundary else ('low_corner_probability' if min(peak)<self.config['minimum_corner_probability'] else evidence['reason']))
                    raw=None if pose is None else pose['xyz_m']
                    results.append({'score':p['score'],'proposal_center':p['center'],'center':((corners.min(0)+corners.max(0))/2).tolist(),'corners':corners.tolist(),'native_size':size,'xyz_m':raw if good else None,'raw_xyz_m':raw,'image_evidence':dict(evidence,accepted=bool(good),reason=reason),'corner_peak_probabilities':peak.tolist(),'reprojection_rms_px':None if pose is None else pose['reprojection_rms_px'],'decoder_source':source})
        if large and self.fallback:results.extend(self.fallback(image,large,K))
        accepted=[]
        for q in sorted(results,key=lambda q:q['score'],reverse=True):
            if q['xyz_m'] is None:continue
            if any(np.linalg.norm(np.asarray(q['corners'])-c,axis=1).max()<3 for c in accepted):q['xyz_m']=None;q['image_evidence']=dict(q['image_evidence'],accepted=False,reason='duplicate_after_refinement')
            else:accepted.append(np.asarray(q['corners']))
        return results
