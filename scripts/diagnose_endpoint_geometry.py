"""Frozen DEV component interventions and sensor-attitude-only pose candidate."""
import json, argparse,time
from pathlib import Path
import numpy as np,torch
from armor_native.matching import match_centers
from armor_native.broad import solve_physical,statistics
from armor_native.level_pose import fit_level_pose
from analyze_quad_shape import decomposition

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
 N=Path(a.root);out=Path(a.output);out.mkdir(exist_ok=False);start=time.monotonic()
 template=torch.load(N/'capacity-extra-control/checkpoint-004000.pt',map_location='cpu',weights_only=False)['geometry']
 mf=N/'height40-anchors/dev.jsonl';frames={(r['session'],r['frame_seq']):r for r in map(json.loads,mf.open())}
 plan={'question':'Does common-center error or relative endpoint geometry cause depth tails, and can level-target geometry with known camera attitude exploit that leverage?', 'control':'frozen current model on unchanged scopedDEV', 'interventions':'GT-substituted translation/length components are diagnostic only; level pose fit uses only image corners and public camera attitude', 'predictions':'If correcting length strongly improves depth and image-only level fit transfers, improve geometry representation/uncertainty; if neither, do not use geometry fitter as an upgrade', 'time_bound_s':300,'no_test_used':True}
 (out/'plan.json').write_text(json.dumps(plan,indent=2))
 rows=[];exact=[]
 for p in map(json.loads,(N/'height40-dev/image-only.jsonl').open()):
  if time.monotonic()-start>300:raise TimeoutError('geometry diagnostic budget')
  f=frames[(p['session'],p['frame_seq'])];pp=p['methods']['height40_initial'];matches=match_centers(pp,f['objects'])
  for j,g in enumerate(f['objects']):
   if j not in matches:continue
   pred=pp[matches[j]]
   if pred['xyz'] is None:continue
   truth=np.array(g['corners']);got=np.array(pred['corners']);error=got-truth
   variants={'baseline':got,'oracle_remove_translation':got-error.mean(0),'oracle_remove_relative_shape':truth+error.mean(0)}
   along=np.zeros_like(error);anti=np.zeros_like(error)
   for bottom,top in [(0,1),(3,2)]:
    unit=truth[top]-truth[bottom];unit/=np.linalg.norm(unit)
    along[[bottom,top]]=np.outer(error[[bottom,top]]@unit,unit)
    differential=(error[top]-error[bottom])@unit/2
    anti[top]=differential*unit;anti[bottom]=-differential*unit
   variants['oracle_remove_along_lamp']=got-along;variants['oracle_remove_length_component']=got-anti
   level=fit_level_pose(got,g['K'],template,f['camera_quaternion_wxyz'])
   rr={'session':f['session'],'frame_seq':f['frame_seq'],'slot':g['slot'],'range_m':g['range_m'],'range_bin':g['range_bin'],'light':f['light'],'height_difference_m':g['height_difference_m'],'facing_cos':g['facing_cos'],'metrics':{}}
   for name,c in variants.items():
    pose=solve_physical(c,g['K'],template)
    if pose is None:continue
    delta=(np.array(pose['xyz_m'])-g['xyz_m'])*1000
    rr['metrics'][name]={'corner_rms_px':float(np.sqrt(np.mean((c-truth)**2))),'depth_mm':abs(float(delta[2])),'xy_mm':float(np.linalg.norm(delta[:2]))}
   if level:
    delta=(np.array(level['xyz_m'])-g['xyz_m'])*1000
    rr['metrics']['level_sensor_pose']={'corner_rms_px':float(np.sqrt(np.mean((np.array(level['corners'])-truth)**2))),'depth_mm':abs(float(delta[2])),'xy_mm':float(np.linalg.norm(delta[:2]))}
   if len(exact)<80:
    v=fit_level_pose(truth,g['K'],template,f['camera_quaternion_wxyz'])
    if v:exact.append({'corner':float(np.sqrt(np.mean((np.array(v['corners'])-truth)**2))),'position_mm':float(np.linalg.norm((np.array(v['xyz_m'])-g['xyz_m'])*1000))})
   rows.append(rr)
 def report(rr):
  return {name:{k:statistics([r['metrics'][name][k] for r in rr if name in r['metrics']]) for k in ['corner_rms_px','depth_mm','xy_mm']} for name in ['baseline','level_sensor_pose',*list(variants)[1:]]}
 result={'complete':True,'seconds':time.monotonic()-start,'objects':len(rows),'primary':report([r for r in rows if 3<=r['range_m']<=5]),'all':report(rows),'by_light':{k:report([r for r in rows if 3<=r['range_m']<=5 and r['light']==k]) for k in ['dark','normal','bright']},'exact_geometry_control':{k:statistics([r[k] for r in exact]) for k in ['corner','position_mm']},'scope':'GT interventions are not deployable and not performance bounds; level_sensor_pose is a candidate using calibrated camera attitude, DEV only'}
 (out/'records.json').write_text(json.dumps(rows));(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({'primary':result['primary'],'exact':result['exact_geometry_control'],'seconds':result['seconds']}))
if __name__=='__main__':main()
