"""Native numeral contour landmarks plus physical lamp corners.

The numeral template is fixed known geometry. Camera attitude is an optional
public sensor input; target pose/identity/range are never inference inputs.
"""
import numpy as np,torch,cv2
from pathlib import Path
from torch import nn
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .plate import NativePlatePrecision
from .geometry import crop_native
from .broad import NATIVE_BUCKETS,solve_physical
from .observability import lamp_evidence
from .level_pose import LINK_TO_CV,fit_level_pose

class NativeGlyphPrecision(NativePlatePrecision):
    def __init__(self,point_count=16,context_dilations=(2,4,8)):
        super().__init__(context_dilations)
        self.heatmap=nn.Conv2d(48,point_count,1);self.offset=nn.Conv2d(48,2*point_count,1)
        nn.init.zeros_(self.heatmap.weight);nn.init.constant_(self.heatmap.bias,-4.6)
        nn.init.zeros_(self.offset.weight);nn.init.zeros_(self.offset.bias)
    def initialize_from_plate(self,state):
        shared={k:v for k,v in state.items() if not k.startswith(('heatmap.','offset.'))}
        missing,extra=self.load_state_dict(shared,strict=False)
        assert not extra and all(k.startswith(('heatmap.','offset.','wide_context.')) for k in missing)
        with torch.no_grad():
            self.heatmap.weight[:4].copy_(state['heatmap.weight']);self.heatmap.bias[:4].copy_(state['heatmap.bias'])
            self.offset.weight[:8].copy_(state['offset.weight']);self.offset.bias[:8].copy_(state['offset.bias'])

def point_targets(points,visible,size,sigma=1.5):
    points=np.asarray(points,float);visible=np.asarray(visible,bool);n=len(points)
    hm=np.zeros((n,size,size),np.float32);off=np.zeros((2*n,size,size),np.float32);mask=np.zeros_like(hm);yy,xx=np.ogrid[:size,:size]
    for k,(p,good) in enumerate(zip(points,visible)):
        if not good:continue
        x,y=p;ix,iy=np.floor(p).astype(int)
        if not (0<=ix<size and 0<=iy<size):raise ValueError('visible landmark outside native crop')
        hm[k]=np.exp(-((xx-ix)**2+(yy-iy)**2)/(2*sigma*sigma));off[2*k:2*k+2,iy,ix]=p-[ix,iy];mask[k,iy,ix]=1
    return hm,off,mask

def translation_sensitivity(template,rv,tv,K,camera_q,visible):
    points=np.asarray(template,float)@cv2.Rodrigues(np.asarray(rv,float))[0].T;xyz=points+np.asarray(tv)
    q=np.asarray(camera_q,float);wc=LINK_TO_CV@Rotation.from_quat(q[[1,2,3,0]]).as_matrix().T
    x,y,z=xyz.T;P=np.zeros((len(x),2,3));P[:,0,0]=K[0,0]/z;P[:,0,2]=-K[0,0]*x/z**2;P[:,1,1]=K[1,1]/z;P[:,1,2]=-K[1,1]*y/z**2
    yaw_velocity=np.cross(np.broadcast_to(wc[:,2],points.shape),points)
    G=np.concatenate([P,np.einsum('nij,nj->ni',P,yaw_velocity)[...,None]],axis=2).reshape(-1,4)
    keep=np.repeat(np.asarray(visible,bool),2);inv=np.linalg.pinv(G[keep],rcond=1e-9)
    J=np.zeros((3,len(points)*2));J[:,keep]=inv[:3]
    return J

def fit_glyph_pose(points,probabilities,K,template,camera_q=None,mode='glyph_level'):
    points=np.asarray(points,float);template=np.asarray(template,float);K=np.asarray(K,float);probabilities=np.asarray(probabilities,float)
    base=solve_physical(points[:4],K,template[:4])
    if mode=='direct':
        return None if base is None else dict(base,corners=points[:4].tolist(),used_points=4)
    if mode=='lamp_level':
        if camera_q is None:return None if base is None else dict(base,corners=points[:4].tolist(),used_points=4)
        result=fit_level_pose(points[:4],K,template[:4],camera_q)
        return None if result is None else dict(result,used_points=4,camera_attitude_used=True)
    valid=(probabilities>=.1)&np.isfinite(points).all(1)
    if valid.sum()<8 or valid[4:].sum()<4:return None
    observed=points[valid];xyz=template[valid];weights=np.sqrt(np.clip(probabilities[valid],.1,1.))[:,None]
    starts=[]
    try:
        sol=cv2.solvePnPGeneric(template[:4],points[:4],K,np.zeros(5),flags=cv2.SOLVEPNP_IPPE)
        starts=[(rv.ravel(),t.ravel()) for rv,t in zip(sol[1],sol[2]) if np.isfinite(t).all() and t[2,0]>.05]
    except cv2.error:pass
    try:
        ok,rv,tv=cv2.solvePnP(xyz,observed,K,np.zeros(5),flags=cv2.SOLVEPNP_SQPNP)
        if ok and np.isfinite(tv).all() and tv[2,0]>.05:starts.append((rv.ravel(),tv.ravel()))
    except cv2.error:pass
    if not starts:return None
    sensor=(mode=='glyph_level' and camera_q is not None)
    if sensor:
        q=np.asarray(camera_q,float);wc=LINK_TO_CV@Rotation.from_quat(q[[1,2,3,0]]).as_matrix().T
        mount=Rotation.from_euler('x',-15,degrees=True).as_matrix()
    def unpack(x):
        if sensor:
            rot=wc@Rotation.from_euler('z',x[3]).as_matrix()@mount
            return cv2.Rodrigues(rot)[0].ravel(),x[:3]
        return x[3:],x[:3]
    def project(x,object_points):
        rv,tv=unpack(x)
        return cv2.projectPoints(object_points,rv,tv,K,np.zeros(5))[0].reshape(-1,2)
    def residual(x):return ((project(x,xyz)-observed)*weights).ravel()
    candidates=[]
    for rv,tv in starts:
        if sensor:
            rw=wc.T@cv2.Rodrigues(rv)[0];x=np.r_[tv,np.arctan2(rw[1,0],rw[0,0])]
        else:x=np.r_[tv,rv]
        lower=np.full(len(x),-np.inf);lower[2]=.05
        fit=least_squares(residual,x,bounds=(lower,np.full(len(x),np.inf)),loss='soft_l1',f_scale=.5,max_nfev=40,ftol=1e-7,xtol=1e-7,gtol=1e-7)
        rv,tv=unpack(fit.x);all_points=project(fit.x,template);error=float(np.sqrt(np.mean(residual(fit.x)**2)))
        if not np.isfinite(all_points).all() or not np.isfinite(tv).all() or tv[2]<=.05:continue
        candidates.append({'xyz_m':tv.tolist(),'rotation_vector':rv.tolist(),'corners':all_points[:4].tolist(),'projected_landmarks':all_points.tolist(),'reprojection_rms_px':error,'used_points':int(valid.sum()),'cost':float(fit.cost),'camera_attitude_used':sensor})
    return min(candidates,key=lambda x:x['cost']) if candidates else None

def glyph_crop_size(coarse,K,template):
    coarse=np.asarray(coarse,float);pose=solve_physical(coarse,K,np.asarray(template)[:4])
    extent=float(np.max(np.ptp(coarse,axis=0)))
    if pose is not None:
        p=cv2.projectPoints(np.asarray(template,float),np.asarray(pose['rotation_vector']),np.asarray(pose['xyz_m']),np.asarray(K,float),np.zeros(5))[0].reshape(-1,2)
        if np.isfinite(p).all():extent=max(extent,float(np.max(np.ptp(p,axis=0))))
    return next((s for s in NATIVE_BUCKETS if s>=extent+48),None)

class GlyphLocalizer:
    supports_camera_attitude=True
    def __init__(self,weights,device='cuda',geometry_backend='scipy',native_library=None):
        if geometry_backend not in {'scipy','native-trf'}:raise ValueError(geometry_backend)
        self.geometry_backend=geometry_backend;self.native_geometry=None
        if geometry_backend=='native-trf':
            from .native_geometry import NativeGeometry
            self.native_geometry=NativeGeometry(native_library)
        ck=torch.load(weights,map_location='cpu',weights_only=False);assert ck['head']=='native_glyph_geometry'
        self.device=torch.device(device);self.template=ck['geometry'];self.glyph_template=ck['glyph_geometry']['object_points'];self.config=ck['inference_config'];self.proposal_threshold=self.config['proposal_threshold'];self.mode=self.config.get('decoder','glyph_level')
        self.model=NativeGlyphPrecision(len(self.glyph_template),tuple(ck['config']['context_dilations'])).to(self.device).eval();self.model.load_state_dict(ck['model'])
        from .quad import QuadLocalizer
        fallback=Path(self.config['fallback_weights'])
        if not fallback.is_absolute():fallback=Path(weights).resolve().parent/fallback
        self.fallback=QuadLocalizer(str(fallback),device)
    @torch.inference_mode()
    def __call__(self,image,proposals,K,camera_quaternion_wxyz=None):
        pose_solver=fit_glyph_pose if self.native_geometry is None else self.native_geometry.fit_glyph_pose
        evidence_reader=lamp_evidence if self.native_geometry is None else self.native_geometry.lamp_evidence
        results=[];groups={};large=[]
        for p in proposals:
            size=glyph_crop_size(p['coarse_corners'],K,self.glyph_template)
            if size is None or size>512:large.append(p);continue
            patch,origin,_=crop_native(image,p['center'],size);groups.setdefault(size,[]).append((p,patch,origin))
        for size,items in groups.items():
            batch=max(1,262144//(size*size))
            for start in range(0,len(items),batch):
                block=items[start:start+batch];x=torch.from_numpy(np.stack([z[1] for z in block]).transpose(0,3,1,2).copy()).float().to(self.device)/255
                with torch.autocast('cuda',dtype=torch.float16,enabled=self.device.type=='cuda'):pred=self.model(x)
                local=pred['corners'].cpu().numpy();peaks=pred['heatmap'].float().sigmoid().flatten(2).max(-1).values.cpu().numpy()
                for (p,patch,origin),points,confidence in zip(block,local,peaks):
                    points=points+origin;pose=pose_solver(points,confidence,K,self.glyph_template,camera_quaternion_wxyz,self.mode)
                    source=self.mode
                    if pose is None and min(confidence[:4])>=.1:
                        pose=pose_solver(points,confidence,K,self.glyph_template,None,'direct');source='direct_fallback'
                    corners=np.asarray(pose['corners'] if pose is not None else points[:4]);evidence=evidence_reader(image,corners,rule=self.config['observability_rule'])
                    boundary=bool((corners<0).any() or (corners[:,0]>=image.shape[1]).any() or (corners[:,1]>=image.shape[0]).any())
                    accepted=pose is not None and evidence['accepted'] and not boundary
                    if self.mode in {'direct','lamp_level'}:accepted=accepted and min(confidence[:4])>=.1
                    reason='accepted' if accepted else ('unsupported_landmarks' if pose is None else ('image_boundary' if boundary else ('low_corner_probability' if self.mode in {'direct','lamp_level'} and min(confidence[:4])<.1 else evidence['reason'])))
                    consensus=None
                    if self.config.get('landmark_consensus') is not None:
                        from .glyph_quality import landmark_consensus
                        consensus=landmark_consensus(points,confidence,None if pose is None or source!='glyph_level' else pose.get('projected_landmarks'),self.config['landmark_consensus'])
                        if accepted and not consensus['accepted']:
                            accepted=False;reason=consensus['reason']
                    raw=None if pose is None else pose['xyz_m']
                    results.append({'score':p['score'],'proposal_center':p['center'],'center':((corners.min(0)+corners.max(0))/2).tolist(),'corners':corners.tolist(),'landmarks':points.tolist(),'point_probabilities':confidence.tolist(),'corner_peak_probabilities':confidence[:4].tolist(),'native_size':size,'xyz_m':raw if accepted else None,'raw_xyz_m':raw,'image_evidence':dict(evidence,accepted=bool(accepted),reason=reason),'reprojection_rms_px':None if pose is None else pose['reprojection_rms_px'],'used_points':None if pose is None else pose['used_points'],'decoder_source':source,'camera_attitude_used':False if pose is None else pose.get('camera_attitude_used',False),'landmark_consensus':consensus})
        from .glyph_quality import restore_preduplicate,suppress_duplicates
        if large:
            fallback=self.fallback(image,large,K)
            # One global suppression pass, including native large-crop results.
            if self.config.get('duplicate_relative_extent',0.)>0:
                fallback=[restore_preduplicate(q) for q in fallback]
            results.extend(fallback)
        return suppress_duplicates(results,self.config.get('duplicate_relative_extent',0.))
