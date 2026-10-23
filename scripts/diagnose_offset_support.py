"""Measure how native coordinate decoding uses sparsely supervised offsets."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np,torch
from armor_native.quad import NativeQuadPrecision
from armor_native.geometry import crop_native
from armor_native.data import rgb
from armor_native.broad import statistics
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);ap.add_argument('--seconds',type=int,default=120);ap.add_argument('--weights');a=ap.parse_args();n=Path(a.root);out=Path(a.output);out.mkdir(exist_ok=False);start=time.monotonic();torch.set_num_threads(2)
 weights=Path(a.weights) if a.weights else n/'capacity-extra-control/checkpoint-004000.pt';ck=torch.load(weights,map_location='cpu',weights_only=False);mode=ck['config'].get('coordinate_mode','legacy');assert mode in {'legacy','dense_votes'} and ck['config'].get('refinement_mode','none')=='none';net=NativeQuadPrecision(ck['config']['joint'],mode,ck['config'].get('width_multiplier',1)).cuda().eval();net.load_state_dict(ck['model']);dense_direct=mode=='dense_votes' and ck['config'].get('dense_vote_weight',0)>0
 selected=json.loads((n/'decoder-diagnostic-capacity-control/records.json').read_text());ids={(r['session'],r['frame_seq'],r['slot']) for r in selected};lookup={}
 for line in (n/'retrain-anchors-cyan/train.jsonl').open():
  f=json.loads(line)
  for o in f['objects']:
   key=f['session'],f['frame_seq'],o['slot']
   if key in ids:lookup[key]=f,o
 plan={'question':'How much decoding mass lies inside the directly supervised label region, and how consistent are its endpoint votes?','scope':'same 192 fixed TRAIN objects as previous decoder diagnosis, GPU AMP, unchanged weights; no model selection or GT inference','weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'coordinate_mode':mode,'dense_direct_supervision':dense_direct,'seconds_limit':a.seconds,'decision':'Compare original sparse-label support and trained dense-label support; reduced vote spread alone does not prove improved endpoint accuracy.'}
 (out/'plan.json').write_text(json.dumps(plan,indent=2));rows=[];reconstruction=0
 dy,dx=np.meshgrid(np.arange(-3,4),np.arange(-3,4),indexing='ij')
 for rec in selected:
  if time.monotonic()-start>a.seconds:raise TimeoutError('bounded offset support diagnostic')
  f,o=lookup[rec['session'],rec['frame_seq'],rec['slot']];im=rgb(f['image_path']);anchors=np.asarray(o['training_proposal_corners'],np.float32);native=[crop_native(im,p,128) for p in anchors];patches=np.asarray([v[0] for v in native]).transpose(0,3,1,2).copy()
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):pred=net(torch.from_numpy(patches[None]).float().cuda()/255,torch.from_numpy(anchors[None]).cuda())
  logits=pred['heatmap'][0].float().cpu().numpy();offsets=(pred['vote_vectors'][0].float() if mode=='dense_votes' else pred['offset'][0].float().sigmoid()).cpu().numpy().reshape(4,2,128,128);local_gt=np.asarray(o['corners'])-np.asarray([v[1] for v in native]);decoded=pred['corners'][0].cpu().numpy()
  for k in range(4):
   iy,ix=np.unravel_index(logits[k].argmax(),(128,128));xx=np.clip(ix+dx.ravel(),0,127);yy=np.clip(iy+dy.ravel(),0,127);ll=logits[k,yy,xx].astype(float);w=np.exp(ll-ll.max());w/=w.sum();off=offsets[k][:,yy,xx].T;grid=np.stack([xx,yy],1);votes=grid+off;got=(w[:,None]*votes).sum(0);target=local_gt[k];gt_floor=np.floor(target);frac=target-gt_floor;grid_error=(w[:,None]*grid).sum(0)-gt_floor;offset_error=(w[:,None]*off).sum(0)-frac
   reconstruction=max(reconstruction,float(np.max(np.abs(got-decoded[k]))))
   rows.append({'session':rec['session'],'frame_seq':rec['frame_seq'],'slot':rec['slot'],'corner':k,'gt_floor_cell_mass':float(w[(xx==gt_floor[0])&(yy==gt_floor[1])].sum()),'direct_label_region_mass':float(w[(np.abs(xx-gt_floor[0])<=6)&(np.abs(yy-gt_floor[1])<=6)].sum()) if dense_direct else float(w[(xx==gt_floor[0])&(yy==gt_floor[1])].sum()),'peak_mass':float(w.max()),'vote_rms_spread_px':float(np.sqrt((w[:,None]*(votes-got)**2).sum()/2)),'grid_error_xy_px':grid_error.tolist(),'offset_error_xy_px':offset_error.tolist(),'decoded_error_xy_px':(got-target).tolist()})
 result={'complete':True,'scope':plan['scope'],'seconds':time.monotonic()-start,'objects':len(selected),'points':len(rows),'coordinate_mode':mode,'dense_direct_supervision':dense_direct,'direct_label_region_mass':statistics([r['direct_label_region_mass'] for r in rows]),'direct_label_region_definition':'GT-floor +/-6 square for dense-supervised field, single floor cell otherwise','decoder_reconstruction_max_delta_px':reconstruction,'gt_floor_cell_mass':statistics([r['gt_floor_cell_mass'] for r in rows]),'outside_floor_cell_mass':statistics([1-r['gt_floor_cell_mass'] for r in rows]),'peak_mass':statistics([r['peak_mass'] for r in rows]),'vote_rms_spread_px':statistics([r['vote_rms_spread_px'] for r in rows]),'limitations':'All cells still receive indirect final-coordinate gradients and shared-parameter learning. This is a representation/explicit-supervision alignment diagnostic, not proof of causality or an independent-information gain.'}
 assert reconstruction<5e-5; (out/'result.json').write_text(json.dumps(result,indent=2));(out/'records.json').write_text(json.dumps(rows));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
