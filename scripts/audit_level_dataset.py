"""Check complete native level-body acquisition before model assessment."""
import argparse,json,collections,hashlib
from pathlib import Path
import numpy as np,cv2
from scipy.spatial.transform import Rotation

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();root=Path(a.data)
 assert json.loads((root/'collection-result.json').read_text())['complete']
 plan=json.loads((root/'plan.json').read_text());assert plan['target_level_fixed_zero'] and plan['roll_fixed_zero']
 frames={s:[json.loads(l) for l in (root/'prepared'/(s+'.jsonl')).open()] for s in ['train','dev','sealed']}
 expected=collections.Counter()
 for e in plan['episodes']:expected[e['split']]+=e['frames']
 assert all(len(frames[s])==expected[s] for s in frames), (expected,{s:len(v) for s,v in frames.items()})
 sessions={s:{r['session'] for r in rows} for s,rows in frames.items()};assert not sessions['train']&sessions['dev']
 allrows=frames['train']+frames['dev']+frames['sealed'];keys=[tuple(r[k] for k in ['producer_epoch','frame_seq','timestamp_ns','target_id']) for r in allrows];assert len(set(keys))==len(keys)
 for r in allrows:
  assert r['requested_level_target_verified'] and r['requested_roll_zero_verified']
  assert abs(r['actual_target_pitch_deg'])<.01 and abs(r['actual_target_roll_deg'])<.01 and abs(r['actual_camera_roll_deg'])<.01
  assert r['camera_roll_deg']==0 and r['target_roll_deg']==0 and r['target_pitch_deg']==0
  assert r['frame_seq']>=r['ack_seq']
 eulers=np.array([Rotation.from_quat(np.array(r['target_quaternion_wxyz'])[[1,2,3,0]]).as_euler('ZYX',degrees=True) for r in allrows]);assert np.max(np.abs(eulers[:,1:]))<.01
 K={tuple(r['intrinsics']) for r in allrows};assert len(K)==1
 # Asset mounting angle is distinct from the target body's world pitch.
 label_source=json.loads((root/'label-source.json').read_text());wanted=set(keys);tilts=[];assets=set();plate_counts=collections.Counter()
 with Path(label_source['path']).open() as stream:
  for line in stream:
   l=json.loads(line);key=tuple(l[k] for k in ['producer_epoch','frame_seq','timestamp_ns','target_id'])
   if key in wanted:
    t=l['plate_geometry']['tilt_from_vertical_deg'];assert abs(t-15)<.05;tilts.append(t);assets.add(l['plate_geometry']['asset_sha256']);plate_counts[key]+=1
 assert tilts and len(assets)==1
 prep=json.loads((root/'prepared/audit.json').read_text())
 # IPPE is an initial planar pose solution. Grazing near-field cases can have
 # nonzero initial reprojection residual even for exact, consistent labels.
 # Check template consistency with ordinary pixel-only LM, never fitting t to GT.
 template=np.asarray(json.loads((root/'prepared/geometry.json').read_text())['object_points_armor_m'])
 refined_errors=[];initial_outliers=[]
 for r in allrows:
  for o in r['objects']:
   corners=np.asarray(o['corners']);intrinsics=np.asarray(o['K'])
   pp=cv2.solvePnPGeneric(template,corners,intrinsics,np.zeros(5),flags=cv2.SOLVEPNP_IPPE)
   choices=[(float(e[0]),rv,tv) for rv,tv,e in zip(pp[1],pp[2],pp[3]) if tv[2,0]>0]
   assert choices
   initial_rms,rv,tv=min(choices,key=lambda x:x[0])
   initial_error=float(np.linalg.norm(tv.ravel()-o['xyz_m'])*1000)
   rr,tt=cv2.solvePnPRefineLM(template,corners,intrinsics,np.zeros(5),rv.copy(),tv.copy())
   error=float(np.linalg.norm(tt.ravel()-o['xyz_m'])*1000);refined_errors.append(error)
   if initial_error>.5:
    reproj=cv2.projectPoints(template,rr,tt,intrinsics,np.zeros(5))[0].reshape(-1,2)
    initial_outliers.append({'session':r['session'],'frame_seq':r['frame_seq'],'slot':o['slot'],'range_m':o['range_m'],'facing_cos':o['facing_cos'],'initial_error_mm':initial_error,'initial_reprojection_rms_px':initial_rms,'refined_error_mm':error,'refined_reprojection_rms_px':float(np.sqrt(np.mean((reproj-corners)**2)))})
 refined_control={'objects':len(refined_errors),'max':max(refined_errors),'median':float(np.median(refined_errors)),'p95':float(np.quantile(refined_errors,.95))}
 assert refined_control['max']<.5,refined_control
 assert prep['exact_corner_pnp_control_mm']['p95']<.1,prep['exact_corner_pnp_control_mm']
 counts={}
 for split,rows in frames.items():
  counts[split]={'frames':len(rows),'sessions':len(sessions[split]),'geometric_objects':sum(len(r['objects']) for r in rows),'by_range_light':dict(collections.Counter(str((o['range_bin'],r['light'])) for r in rows for o in r['objects']))}
 out={'complete':True,'source':'all acquired exposures and exact joined asset labels; no model predictions used','frame_counts':counts,'target_pitch_roll_max_abs_deg':np.max(np.abs(eulers[:,1:]),0).tolist(),'camera_roll_max_abs_deg':max(abs(r['actual_camera_roll_deg']) for r in allrows),'armor_mount_tilt_deg_minmax':[min(tilts),max(tilts)],'exact_plate_labels':len(tilts),'frames_with_exact_plate_labels':len(plate_counts),'asset_sha256':list(assets),'K':list(next(iter(K))),'independent_split_sessions':True,'native_rgb_resized':False,'exact_corner_pnp_control_mm':prep['exact_corner_pnp_control_mm'],'pixel_only_lm_geometry_control_mm':refined_control,'initial_ippe_outliers':initial_outliers,'model_pnp_unchanged':True,'protected_external_labels':label_source,'prepared_manifest_sha256':{s:hashlib.sha256((root/'prepared'/(s+'.jsonl')).read_bytes()).hexdigest() for s in frames}}
 with Path(a.output).open('x') as f:json.dump(out,f,indent=2)
 print(json.dumps({k:v for k,v in out.items() if k not in ['frame_counts','prepared_manifest_sha256']},indent=2))
if __name__=='__main__':main()
