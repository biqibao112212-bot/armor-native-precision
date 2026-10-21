"""Image-only two-lamp evidence; no range, pose, target truth or error thresholds.

Simulator LED appearance gate, not a certified real-camera occlusion classifier.
The identical operator can annotate known GT strips for offline supervision.
Integer nearest pixel reads retain input pixels; no RGB interpolation.
"""
import numpy as np

RULE={'version':'two-lamp-v1','chroma_min':35,'peak_min':80,'support_min':0.65,'minimum_length_px':4.,'minimum_separation_px':3.}
def lamp_evidence(image,corners):
 c=np.asarray(corners,float);empty={'accepted':False,'reason':'invalid_corners','support':[0.,0.]}
 if c.shape!=(4,2) or not np.isfinite(c).all():return empty
 v0=c[1]-c[0];v1=c[2]-c[3];lengths=[float(np.linalg.norm(v0)),float(np.linalg.norm(v1))];separation=float(np.linalg.norm((c[0]+c[1]-c[2]-c[3])*.5));scores=[];chroma_values=[]
 for a,b,length in [(c[0],c[1],lengths[0]),(c[3],c[2],lengths[1])]:
  if length<1:scores.append(0.);chroma_values.append(0.);continue
  direction=(b-a)/length;normal=np.array([-direction[1],direction[0]]);radius=max(1,min(6,int(np.ceil(length*.06)),int(max(1,separation*.15))));ts=np.linspace(.15,.85,max(9,min(81,int(length))));offset=np.arange(-radius,radius+1);q=np.rint(a+ts[:,None,None]*(b-a)+offset[None,:,None]*normal).astype(int);inside=(q[:,:,0]>=0)&(q[:,:,0]<image.shape[1])&(q[:,:,1]>=0)&(q[:,:,1]<image.shape[0]);q[:,:,0]=q[:,:,0].clip(0,image.shape[1]-1);q[:,:,1]=q[:,:,1].clip(0,image.shape[0]-1);rgb=image[q[:,:,1],q[:,:,0]].astype(float);red=rgb[:,:,0]-np.maximum(rgb[:,:,1],rgb[:,:,2]);blue=rgb[:,:,2]-np.maximum(rgb[:,:,0],rgb[:,:,1]);chroma=np.maximum(red,blue);evidence=(chroma>=RULE['chroma_min'])&(rgb.max(2)>=RULE['peak_min'])&inside;scores.append(float(evidence.any(1).mean()));chroma_values.append(float(np.median(np.where(inside,chroma,0).max(1))))
 cross=np.cross(c[1]-c[0],c[2]-c[1]);crosses=[float(np.cross(c[(i+1)%4]-c[i],c[(i+2)%4]-c[(i+1)%4])) for i in range(4)];convex=all(x>0 for x in crosses) or all(x<0 for x in crosses)
 resolved=min(lengths)>=RULE['minimum_length_px'] and separation>=RULE['minimum_separation_px'] and convex
 accepted=resolved and min(scores)>=RULE['support_min']
 return {'accepted':bool(accepted),'reason':'two_lamps_supported' if accepted else ('degenerate_or_unresolved_quad' if not resolved else 'insufficient_two_lamp_image_evidence'),'support':scores,'median_chroma':chroma_values,'lengths_px':lengths,'separation_px':separation}
