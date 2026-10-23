"""TRAIN-only cross-distance gradient diagnostic, CPU; no model updates."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from train_native_quad import Samples,read
from armor_native.model import precision_loss
from armor_native.quad import NativeQuadPrecision
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);ap.add_argument('--seconds',type=int,default=240);a=ap.parse_args();n=Path(a.root);out=Path(a.output);out.mkdir(exist_ok=False);torch.set_num_threads(2)
 initial=n/'capacity-extra-control/checkpoint-004000.pt';manifest=n/'retrain-anchors-cyan/train.jsonl';ck=torch.load(initial,map_location='cpu',weights_only=False)
 model=NativeQuadPrecision(True).eval();model.load_state_dict(ck['model']);params=list(model.parameters());all_rows=read(manifest);groups={'near':lambda z:z<3,'primary':lambda z:3<=z<=5,'far':lambda z:z>5}
 plan={'question':'Does the current broad-distance loss produce dominant or opposing gradients for primary 3-5m precision?','scope':'96 augmented TRAIN objects, 8 batches per near/primary/far group; CPU FP32, no parameter updates. Local sampled gradient diagnosis, not a proof of global optimization conflict.','initial_sha256':hashlib.sha256(initial.read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'steps_per_group':8,'offset':24000,'seconds_limit':a.seconds,'predictions':{'opposition':'Other-domain averaged gradient opposes primary and dominates mixture: supports loss/sampling interference as follow-up','aligned':'No measured opposition: does not support this sampled interference explanation','inconclusive':'Noisy or weak averaged directions do not diagnose broad-domain impossibility'},'native_rgb_resized':False,'local_backup':False}
 (out/'plan.json').write_text(json.dumps(plan,indent=2));start=time.monotonic();vectors={};records=[]
 for name,predicate in groups.items():
  rows=[dict(r,objects=[o for o in r['objects'] if predicate(o['range_m'])]) for r in all_rows];rows=[r for r in rows if r['objects']]
  ds=Samples(rows,8,ck['geometry'],2026102203,'image_coarse',24000,'distance');vv=[]
  for i in range(8):
   if time.monotonic()-start>a.seconds:raise TimeoutError('bounded CPU gradient diagnostic')
   b=ds[i];pred=model(b['image'],b['anchors']);base,terms=precision_loss({'heatmap':pred['heatmap'],'offset':pred['offset']},b)
   e=pred['corners']-b['corners'];coord=F.smooth_l1_loss(e/.236,torch.zeros_like(e));good=e.detach().abs().amax((1,2))<4
   d=torch.bmm(b['jacobian'],e.reshape(-1,8,1)).squeeze(-1);geo=(F.smooth_l1_loss(d/torch.tensor([.0106,.0106,.0743]),torch.zeros_like(d),reduction='none').mean(1)*good).sum()/good.sum().clamp_min(1)
   shape=F.smooth_l1_loss((e-e.mean(1,keepdim=True))/.236,torch.zeros_like(e));loss=base+.1*coord+.2*(geo+.25*shape)
   g=torch.autograd.grad(loss,params);v=torch.cat([x.detach().flatten() for x in g]);assert torch.isfinite(v).all();vv.append(v)
   records.append({'group':name,'batch':i,'range_bins':b['range_bin'].tolist(),'loss':float(loss.detach()),'coordinate':float(coord.detach()),'geometry':float(geo.detach()),'shape':float(shape.detach()),'gradient_norm':float(v.norm()),'seconds':time.monotonic()-start,**terms})
  vectors[name]=torch.stack(vv);print(json.dumps({'group_complete':name,'seconds':time.monotonic()-start}),flush=True)
 means={k:v.mean(0).double() for k,v in vectors.items()}
 def cosine(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-12))
 primary=means['primary'];mix=(3*means['near']+2*primary+3*means['far'])/8
 result={'complete':True,'seconds':time.monotonic()-start,'scope':plan['scope'],'mean_gradient_norm':{k:float(v.norm()) for k,v in means.items()},'cosines':{a:{b:cosine(x,means[b]) for b in means} for a,x in means.items()},'nominal_distance_balanced_mixture_primary_cosine':cosine(mix,primary),'mean_gradient_coherence':{k:float(v.mean(0).norm()/v.norm(dim=1).mean()) for k,v in vectors.items()},'records':records,'interpretation_limit':'Averaged loss gradients on only sampled TRAIN objects; Adam state, clipping, future updates, data geometry differ. No causal or convergence conclusion by itself.'}
 torch.save(vectors,out/'gradient-vectors.pt');(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
if __name__=='__main__':main()
