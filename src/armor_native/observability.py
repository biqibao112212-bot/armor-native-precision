"""Image-only two-lamp evidence; no range, pose, target truth or error thresholds.

Simulator LED appearance gate, not a certified real-camera occlusion classifier.
The identical operator can annotate known GT strips for offline supervision.
Integer nearest pixel reads retain input pixels; no RGB interpolation.
"""
import numpy as np

RULE={'version':'two-lamp-v1','chroma_min':35,'peak_min':80,'support_min':0.65,'minimum_length_px':4.,'minimum_separation_px':3.}
CYAN_RULE={**RULE,'version':'two-lamp-cyan-v2','cyan_green_tolerance':10}
CYAN_PIXEL_CENTER_RULE={**CYAN_RULE,'version':'two-lamp-cyan-pixelcenter-v3','pixel_center_offset':.5}
RULES={r['version']:r for r in [RULE,CYAN_RULE,CYAN_PIXEL_CENTER_RULE]}
def lamp_evidence(image,corners,rule=None):
 rule=RULE if rule is None else rule
 assert rule['version'] in RULES
 c=np.asarray(corners,float);empty={'accepted':False,'reason':'invalid_corners','support':[0.,0.]}
 if c.shape!=(4,2) or not np.isfinite(c).all():return empty
 v0=c[1]-c[0];v1=c[2]-c[3];lengths=[float(np.linalg.norm(v0)),float(np.linalg.norm(v1))];separation=float(np.linalg.norm((c[0]+c[1]-c[2]-c[3])*.5));scores=[];chroma_values=[]
 for a,b,length in [(c[0],c[1],lengths[0]),(c[3],c[2],lengths[1])]:
  if length<1:scores.append(0.);chroma_values.append(0.);continue
  direction=(b-a)/length;normal=np.array([-direction[1],direction[0]]);radius=max(1,min(6,int(np.ceil(length*.06)),int(max(1,separation*.15))));ts=np.linspace(.15,.85,max(9,min(81,int(length))));offset=np.arange(-radius,radius+1);q=np.rint(a+ts[:,None,None]*(b-a)+offset[None,:,None]*normal-rule.get('pixel_center_offset',0.)).astype(int);inside=(q[:,:,0]>=0)&(q[:,:,0]<image.shape[1])&(q[:,:,1]>=0)&(q[:,:,1]<image.shape[0]);q[:,:,0]=q[:,:,0].clip(0,image.shape[1]-1);q[:,:,1]=q[:,:,1].clip(0,image.shape[0]-1);rgb=image[q[:,:,1],q[:,:,0]].astype(float);red=rgb[:,:,0]-np.maximum(rgb[:,:,1],rgb[:,:,2]);blue=rgb[:,:,2]-np.maximum(rgb[:,:,0],rgb[:,:,1]);
  if rule['version'] in {'two-lamp-cyan-v2','two-lamp-cyan-pixelcenter-v3'}:
   # Cyan LEDs retain a strong blue-red opponent signal under bright ambient light.
   # Keep the same threshold; exclude green/yellow rather than demanding B>G+35.
   blue=np.where(rgb[:,:,2]>=rgb[:,:,1]-rule['cyan_green_tolerance'],rgb[:,:,2]-rgb[:,:,0],blue)
  chroma=np.maximum(red,blue);evidence=(chroma>=rule['chroma_min'])&(rgb.max(2)>=rule['peak_min'])&inside;scores.append(float(evidence.any(1).mean()));chroma_values.append(float(np.median(np.where(inside,chroma,0).max(1))))
 cross=np.cross(c[1]-c[0],c[2]-c[1]);crosses=[float(np.cross(c[(i+1)%4]-c[i],c[(i+2)%4]-c[(i+1)%4])) for i in range(4)];convex=all(x>0 for x in crosses) or all(x<0 for x in crosses)
 resolved=min(lengths)>=rule['minimum_length_px'] and separation>=rule['minimum_separation_px'] and convex
 accepted=resolved and min(scores)>=rule['support_min']
 return {'rule_version':rule['version'],'accepted':bool(accepted),'reason':'two_lamps_supported' if accepted else ('degenerate_or_unresolved_quad' if not resolved else 'insufficient_two_lamp_image_evidence'),'support':scores,'median_chroma':chroma_values,'lengths_px':lengths,'separation_px':separation}
