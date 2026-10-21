"""Convert only original TRAIN/dev rows to portable manifests; never open old test."""
import argparse,json,sys,numpy as np
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--consumer-source',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();sys.path.insert(0,a.consumer_source)
 from train import rows_for
 from data_balance_control import new_rows
 out=Path(a.output);out.mkdir(exist_ok=False,parents=True);counts={}
 for split,rows in [('train',rows_for('train')+new_rows('train')),('dev',rows_for('val')+new_rows('dev'))]:
  with (out/(split+'.jsonl')).open('x') as f:
   for r in rows:
    r.update(image_path=r['path'],split=split,light='legacy',domain='legacy',camera_roll_deg=0.,target_roll_deg=0.,target_pitch_deg=0.,coarse_ignored_boxes=[])
    objects=[]
    for o in r['objects']:
     c=np.array(o['corners']);extent=float(np.ptp(c,axis=0).max());distance=float(np.linalg.norm(o['xyz_m']));size=next((s for s in [128,192,256,384,512,768,1024,1536] if s>=extent+40),None)
     if size is None or distance>8:continue
     o.update(screen_corners=o['corners'],extent_px=extent,native_size=size,range_m=distance,range_bin=int(np.searchsorted([1,2,3,4,5,6,7,8],distance,side='right')),height_difference_m=None);objects.append(o)
    r['objects']=objects;f.write(json.dumps(r)+'\n')
  counts[split]={'frames':len(rows),'sessions':len(set(r['session'] for r in rows)),'objects':sum(len(r['objects']) for r in rows)}
 (out/'audit.json').write_text(json.dumps({'counts':counts,'old_test_opened':False,'corners':'unchanged original labels; no GT rendered image modification','legacy_xyz_reference':'measured exact-corner rectangle center; physical-root new domains differ by ~2.7mm; corner supervision identical'},indent=2));print(json.dumps(counts))
if __name__=='__main__':main()
