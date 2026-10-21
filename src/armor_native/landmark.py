"""Coarse landmark -> native128 point refinement, with report-precision agreement.

No target pose, range or truth is an input. Four separate integer crops preserve
pixels even when a close plate is much larger than a single128 ROI.
"""
import numpy as np,torch
from .broad import NativeBucketPrecision,solve_physical
from .geometry import crop_native
from .observability import lamp_evidence
class LandmarkNativeLocalizer:
 def __init__(self,weights,device='cuda'):
  self.device=torch.device(device);ck=torch.load(weights,map_location='cpu',weights_only=False);assert ck['head']=='native_landmark_fusion';self.template=ck['geometry'];self.config=ck['inference_config'];self.model=NativeBucketPrecision().to(self.device).eval();self.model.load_state_dict(ck['model']);self.report_model=NativeBucketPrecision().to(self.device).eval();self.report_model.load_state_dict(ck['report_model']);self.proposal_threshold=self.config['proposal_threshold']
 @torch.inference_mode()
 def __call__(self,image,proposals,K):
  result=[]
  for p in proposals:
   anchors=np.asarray(p['coarse_corners'],float)
   if anchors.shape!=(4,2) or not np.isfinite(anchors).all():continue
   patches=[];origins=[]
   for anchor in anchors:
    patch,origin,_=crop_native(image,anchor,128);patches.append(patch);origins.append(origin)
   x=torch.from_numpy(np.stack(patches).transpose(0,3,1,2).copy()).float().to(self.device)/255
   with torch.autocast('cuda',dtype=torch.float16,enabled=self.device.type=='cuda'):pred=self.model(x)
   decoded=self.model.points(pred).cpu().numpy();point=np.array([decoded[i,i] for i in range(4)]);c=point+np.array(origins);peak=pred['heatmap'].float().sigmoid().flatten(2).max(-1).values.cpu().numpy();probs=[float(peak[i,i]) for i in range(4)];evidence=lamp_evidence(image,c);selected='landmark128';agreement=None;report_probs=None
   if float(p['extent_px'])+40<=128:
    patch,origin,_=crop_native(image,p['center'],128);x0=torch.from_numpy(patch.transpose(2,0,1).copy())[None].float().to(self.device)/255
    with torch.autocast('cuda',dtype=torch.float16,enabled=self.device.type=='cuda'):pr=self.report_model(x0)
    old=self.report_model.points(pr).cpu().numpy()[0]+origin;old_evidence=lamp_evidence(image,old);agreement=float(np.linalg.norm(old-c,axis=1).max());report_probs=pr['heatmap'].float().sigmoid().flatten(2).max(-1).values[0].cpu().tolist()
    if old_evidence['accepted'] and evidence['accepted'] and agreement<=self.config['report_agreement_max_px']:c=old;evidence=old_evidence;selected='report128_agreed';probs=report_probs
   boundary=bool((point<4).any() or (point>124).any() or (c<0).any() or (c[:,0]>=image.shape[1]).any() or (c[:,1]>=image.shape[0]).any());pose=solve_physical(c,K,self.template);raw=None if pose is None else pose['xyz_m'];accepted=raw is not None and evidence['accepted'] and not boundary and min(probs)>=self.config['minimum_corner_probability'];reason='accepted' if accepted else ('crop_boundary' if boundary else ('low_corner_probability' if min(probs)<self.config['minimum_corner_probability'] else evidence['reason']))
   result.append({'score':p['score'],'proposal_center':p['center'],'center':((c.min(0)+c.max(0))/2).tolist(),'native_size':128,'native_landmark_patches':4,'corners':c.tolist(),'xyz_m':raw if accepted else None,'raw_xyz_m':raw,'image_evidence':dict(evidence,accepted=bool(accepted),reason=reason),'corner_peak_probabilities':probs,'selected_head':selected,'report_agreement_px':agreement,'report_corner_peak_probabilities':report_probs,'reprojection_rms_px':None if pose is None else pose['reprojection_rms_px']})
  return result
