"""Exact exposure join, physical semantic corners, explicit visibility/unknown ledger."""
import argparse,json,hashlib,collections
from pathlib import Path
import numpy as np,cv2
from scipy.spatial.transform import Rotation

def rotation(q):return Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()
def physical_order(p):
 p=np.asarray(p);left=np.argsort(p[:,0])[:2];right=np.argsort(p[:,0])[2:];return [left[np.argmin(p[left,2])],left[np.argmax(p[left,2])],right[np.argmax(p[right,2])],right[np.argmin(p[right,2])]]
def dump(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('data');ap.add_argument('--extra-data',nargs='*',default=[]);ap.add_argument('--geometry',help='existing TRAIN-only fixed template for validation-only datasets');a=ap.parse_args();root=Path(a.data);assert json.loads((root/'collection-result.json').read_text())['complete'];out=root/'prepared';out.mkdir(exist_ok=False)
 frames=[];labels={};other_quads={}
 for data_root in [root,*map(Path,a.extra_data)]:
  assert json.loads((data_root/'collection-result.json').read_text())['complete']
  new=[json.loads(l) for p in sorted((data_root/'episodes').glob('*/frames.jsonl')) for l in p.open()]
  for r in new:
   r['image_path']=str((data_root/r['image_path']).resolve());spec=json.loads((data_root/'episodes'/r['session']/'spec.json').read_text());r['camera_roll_deg']=spec['config']['camera_roll_deg'];r['target_roll_deg']=spec['config']['target_rpy_deg'][0];r['target_pitch_deg']=spec['config']['target_rpy_deg'][1]
  frames.extend(new);wanted={tuple(r[k] for k in ['producer_epoch','frame_seq','timestamp_ns','target_id']) for r in new};exposures={key[:3]:key[3] for key in wanted}
  for line in (data_root/'exact-corners.jsonl').open():
   l=json.loads(line);key=tuple(l[k] for k in ['producer_epoch','frame_seq','timestamp_ns','target_id'])
   if key in wanted:labels[key+(l['relative_slot'],)]=l
   elif key[:3] in exposures and l['target_number']==3:other_quads.setdefault(key[:3],[]).append(l['exact_corners_px'])
 counts=collections.Counter();dist=collections.Counter();template=None;errors=[];by_split={k:[] for k in ['train','dev','sealed']};ledger=[]
 # Fixed asset template selected solely from the first TRAIN annotation; never test fitting.
 for r in frames:
  if r['split']!='train':continue
  for a in r['armors']:
   l=labels.get(tuple(r[k] for k in ['producer_epoch','frame_seq','timestamp_ns','target_id'])+(a['slot'],))
   if l is not None:
    raw=np.array(l['plate_geometry']['object_corners_armor_m']);template=raw[physical_order(raw)];break
  if template is not None:break
 if a.geometry:template=np.array(json.loads(Path(a.geometry).read_text())['object_points_armor_m'])
 if template is None:raise RuntimeError('no training geometry')
 dump(out/'geometry.json',{'schema':'armor-native.physical-corners/1','object_points_armor_m':template.tolist(),'order':['physical_left_bottom','physical_left_top','physical_right_top','physical_right_bottom'],'source':'existing TRAIN-only template' if a.geometry else 'first TRAIN asset annotation only; fixed across all splits','physical_armor_origin':True})
 for r in frames:
  Rc=rotation(r['camera_quaternion_wxyz']);Rt=rotation(r['target_quaternion_wxyz']);cam=np.array(r['camera_position_world']);tp=np.array(r['target_position_world']);fx,fy,cx,cy=r['intrinsics'];K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]]);objs=[];statuses=[]
  for a in r['armors']:
   l=labels.get(tuple(r[k] for k in ['producer_epoch','frame_seq','timestamp_ns','target_id'])+(a['slot'],));world=Rt@a['position']+tp;delta=world-cam;link=Rc.T@delta;xyz=np.array([-link[1],-link[2],link[0]]);distance=float(np.linalg.norm(delta));cos=float(-np.dot(Rt@a['normal'],delta)/distance);status='unknown_label'
   if l is not None:
    status=l['visibility']['classification'];raw=np.array(l['plate_geometry']['object_corners_armor_m']);corners=np.array(l['exact_corners_px'])[physical_order(raw)]
    if cos<.12:status='back_or_extreme_grazing'
    elif xyz[2]<=.1 or not .3<=distance<=8:status='outside_range'
    elif status=='fully_in_frame':
     extent=float(np.max(np.ptp(corners,axis=0)));size=next((s for s in [128,192,256,384,512,768,1024,1536] if s>=extent+40),None)
     if size is None:status='native_roi_unsupported'
     else:
      pp=cv2.solvePnPGeneric(template,corners,K,np.zeros(5),flags=cv2.SOLVEPNP_IPPE);solutions=[(float(e[0]),t.ravel()) for t,e in zip(pp[2],pp[3]) if t[2,0]>0]
      if not solutions:raise RuntimeError('exact geometry has no positive pose')
      e,t=min(solutions,key=lambda x:x[0]);errors.append(float(np.linalg.norm(t-xyz)*1000));center=(np.min(corners,axis=0)+np.max(corners,axis=0))/2;band=int(np.searchsorted([1,2,3,4,5,6,7,8],distance,side='right'));obj={'slot':a['slot'],'corners':corners.tolist(),'screen_corners':l['exact_corners_px'],'center':center.tolist(),'K':K.tolist(),'xyz_m':xyz.tolist(),'range_m':distance,'range_bin':band,'facing_cos':cos,'height_difference_m':float(cam[2]-world[2]),'native_size':size,'extent_px':extent};objs.append(obj);dist[(r['split'],band)]+=1;status='fine_supervised'
   counts[(r['split'],status)]+=1;statuses.append({'slot':a['slot'],'status':status,'range_m':distance,'facing_cos':cos})
  ignored=[]
  for a,status in zip(r['armors'],statuses):
   if status['status']=='fine_supervised':continue
   optical=Rc.T@(Rt@a['position']+tp-cam);z=optical[0]
   if z>.1:
    u=cx-fx*optical[1]/z;v=cy-fy*optical[2]/z;rad=.18*max(fx,fy)/z+24;ignored.append([u-rad,v-rad,u+rad,v+rad])
  for quad in other_quads.get(tuple(r[k] for k in ['producer_epoch','frame_seq','timestamp_ns']),[]):
   q=np.array(quad);ignored.append([*(q.min(0)-16),*(q.max(0)+16)])
  row={**r,'coarse_ignored_boxes':ignored,'image_path':str((root/r['image_path']).resolve()),'objects':objs,'slot_ledger':statuses};by_split[r['split']].append(row)
 for split,rows in by_split.items():
  with (out/(split+'.jsonl')).open('x') as f:
   for r in rows:f.write(json.dumps(r)+'\n')
 summary={'frames':{s:len(v) for s,v in by_split.items()},'supervised_objects':{s:sum(len(r['objects']) for r in v) for s,v in by_split.items()},'slot_counts':{'/'.join(k):v for k,v in counts.items()},'physical_range_bin_counts':{f'{s}/{b}':n for (s,b),n in dist.items()},'native_size_counts':dict(collections.Counter(str(o['native_size']) for r in by_split['train'] for o in r['objects'])),'exact_corner_pnp_control_mm':{'max':max(errors),'median':float(np.median(errors)),'p95':float(np.quantile(errors,.95))},'unknown_slots_are_not_negatives':True,'occlusion_truth_available':False,'manifest_sha256':{s:hashlib.sha256((out/(s+'.jsonl')).read_bytes()).hexdigest() for s in by_split}}
 dump(out/'audit.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
