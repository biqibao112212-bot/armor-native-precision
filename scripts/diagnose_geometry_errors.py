"""Post-hoc decomposition of saved image-only predictions; no model fitting.

Counterfactual corners use labels OFFLINE to diagnose error propagation, never
as deployable outputs. Opened confirmation data are diagnostic data thereafter.
"""
import argparse,collections,json,hashlib
from pathlib import Path
import numpy as np
from armor_native.broad import solve_physical,statistics
from armor_native.matching import match_centers

def read(p):return [json.loads(l) for l in Path(p).open()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--predictions',required=True);ap.add_argument('--scored',required=True);ap.add_argument('--geometry',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 frames=read(a.manifest);preds=read(a.predictions);scores={(r['session'],r['frame_seq'],r['slot'],r['method']):r for r in read(a.scored)};template=json.loads(Path(a.geometry).read_text())['object_points_armor_m'];records=[]
 for f,p in zip(frames,preds):
  assert (f['session'],f['frame_seq'])==(p['session'],p['frame_seq'])
  for method in ['broad_native','new_fine_old_proposals']:
   pp=p['methods'][method];pairs=match_centers(pp,f['objects'])
   for j,g in enumerate(f['objects']):
    s=scores[f['session'],f['frame_seq'],g['slot'],method];r=dict(s);r['reason']='unmatched'
    if j in pairs:
     q=pp[pairs[j]];c=np.asarray(q['corners']);gt=np.asarray(g['corners']);truth=np.asarray(g['xyz_m']);K=g['K'];e=c-gt;mean=e.mean(0);centered=gt-gt.mean(0);scale=float((e*centered).sum()/(centered**2).sum());r.update(reason=q['image_evidence']['reason'],head=q['selected_head'],corner_error_xy=e.tolist(),common_shift_px=mean.tolist(),relative_scale_error=scale,corner_peak_min=min(q['corner_peak_probabilities']),reprojection_rms_px=q['reprojection_rms_px'])
     raw=solve_physical(c,K,template);assert abs(float(np.sqrt(np.mean(e*e)))-s['corner_rms_px'])<1e-6
     for key,cc in [('raw',c),('gt_oracle',gt),('remove_common_shift',c-mean),('common_shift_only',gt+mean),('remove_isotropic_scale',c-scale*centered)]:
      sol=solve_physical(cc,K,template)
      if sol:
       d=(np.asarray(sol['xyz_m'])-truth)*1000;r[key+'_depth_mm']=float(abs(d[2]));r[key+'_xy_mm']=float(np.linalg.norm(d[:2]))
     # Central finite differences measure local sensitivity, not a predicted gain.
     J=np.zeros((3,8));epsilon=.001
     for k in range(8):
      plus=gt.copy();minus=gt.copy();plus.flat[k]+=epsilon;minus.flat[k]-=epsilon;sp=solve_physical(plus,K,template);sm=solve_physical(minus,K,template)
      if sp and sm:J[:,k]=(np.asarray(sp['xyz_m'])-sm['xyz_m'])*1000/(2*epsilon)
     r['linearized_depth_error_mm']=float((J@e.ravel())[2]);r['depth_sensitivity_mm_per_px_l2']=float(np.linalg.norm(J[2]));r['common_shift_rms_px']=float(np.sqrt(np.mean(mean**2)));r['shape_rms_px']=float(np.sqrt(np.mean((e-mean)**2)))
    records.append(r)
 def stats(rr):
  keys=['corner_rms_px','raw_depth_mm','raw_xy_mm','gt_oracle_depth_mm','gt_oracle_xy_mm','remove_common_shift_depth_mm','common_shift_only_depth_mm','remove_isotropic_scale_depth_mm','depth_sensitivity_mm_per_px_l2','shape_rms_px','common_shift_rms_px']
  return {'n':len(rr),'detected':sum(r['detected'] for r in rr),'valid':sum(r['valid_pnp'] for r in rr),'reasons':dict(collections.Counter(r['reason'] for r in rr)),**{k:statistics([r[k] for r in rr if k in r]) for k in keys}}
 result={'purpose':__doc__,'inputs':{k:{'path':getattr(a,k),'sha256':sha(getattr(a,k))} for k in ['manifest','predictions','scored','geometry']},'diagnostic_only':True,'absolute_targets':{'corner_p95_px':.236,'depth_p95_mm':74.3,'xy_p95_mm':10.6,'reference_valid_fraction':812/817,'future_xy_p95_3_5m_mm':50},'scopes':{}}
 for scope,ff in [('all',lambda r:True),('3_5m',lambda r:3<=r['range_m']<=5)]:
  rr=[r for r in records if r['method']=='broad_native' and r['observable'] and ff(r)];vv=[r for r in rr if r['valid_pnp']];rej=[r for r in rr if r['detected'] and not r['valid_pnp']]
  result['scopes'][scope]={'eligible':stats(rr),'accepted':stats(vv),'refused':stats(rej),'by_head':{h:stats([r for r in vv if r['head']==h]) for h in sorted({r['head'] for r in vv})},'by_range':{str(k):stats([r for r in vv if r['range_bin']==k]) for k in sorted({r['range_bin'] for r in vv})},'refused_raw_at_absolute_targets':sum(r.get('raw_depth_mm',float('inf'))<=74.3 and r.get('raw_xy_mm',float('inf'))<=10.6 for r in rej)}
 with (out/'records.jsonl').open('x') as f:
  for r in records:f.write(json.dumps(r)+'\n')
 (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['scopes']['3_5m'],indent=2))
if __name__=='__main__':main()
