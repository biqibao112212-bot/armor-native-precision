"""DEV-only check of interference versus scale-specific frozen precision branch."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np,torch,cv2
from train_broad import read,pool,sample
from armor_native.broad import NativeBucketPrecision,solve_physical,statistics
from armor_native.observability import lamp_evidence

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();r=Path(a.run);p=Path(__file__).resolve().parents[1];torch.set_num_threads(3);cv2.setNumThreads(1);out=r/'expert-dev-comparison';out.mkdir(exist_ok=False);models={};sources={'original128':p/'weights/native_roi_spatial.pt','broad10000':r/'fine/checkpoint-010000.pt','observable6000':r/'fine-observable/checkpoint-006000.pt'};template=json.loads((r/'data/prepared/geometry.json').read_text())['object_points_armor_m']
 for name,path in sources.items():
  ck=torch.load(path,map_location='cpu',weights_only=False);m=NativeBucketPrecision().cuda().eval();m.load_state_dict(ck['model']);models[name]=m
 rows=read(r/'observable-data/dev.jsonl');pp=pool(rows);keys=sorted(pp);rng=np.random.default_rng(1042);records=[];start=time.monotonic()
 with torch.inference_mode():
  for i in range(384):
   key=keys[i%len(keys)];ss=sorted(pp[key]);pairs=pp[key][ss[int(rng.integers(len(ss)))]];row,obj=pairs[int(rng.integers(len(pairs)))];s=sample(row,obj,rng,False);im=(s['image'].permute(1,2,0).numpy()*255).round().astype(np.uint8);x=s['image'][None].cuda();methods={}
   for name,m in models.items():
    if name=='original128' and obj['native_size']!=128:continue
    with torch.autocast('cuda',dtype=torch.float16):pred=m(x)
    c=m.points(pred).cpu().numpy()[0];e=lamp_evidence(im,c);pose=solve_physical(c,s['K'],template);d=None if pose is None else (np.array(pose['xyz_m'])-s['xyz'])*1000;methods[name]={'accepted':e['accepted'] and d is not None,'corner_rms_px':float(np.sqrt(np.mean((c-s['corners'])**2))),'depth_mm':None if d is None else float(abs(d[2])),'xy_mm':None if d is None else float(np.linalg.norm(d[:2])),'support':e['support']}
   for fallback in ['broad10000','observable6000']:
    selected='original128' if 'original128' in methods and methods['original128']['accepted'] else fallback;methods['native128_then_'+fallback]={**methods[selected],'selected':selected}
   records.append({'session':row['session'],'frame_seq':row['frame_seq'],'slot':obj['slot'],'range_m':obj['range_m'],'size':obj['native_size'],'light':row['light'],'methods':methods})
 def stats(rr,name):
  vv=[x['methods'][name] for x in rr if name in x['methods']];aa=[x for x in vv if x['accepted']];return {'eligible':len(vv),'accepted':len(aa),'coverage':len(aa)/max(1,len(vv)),**{k:statistics([x[k] for x in aa]) for k in ['corner_rms_px','depth_mm','xy_mm']}}
 names=sorted(set(k for x in records for k in x['methods']));result={'scope':'observable DEV, GT-centered jittered native crops only; not end-to-end or test','all':{n:stats(records,n) for n in names},'primary3_5':{n:stats([x for x in records if 3<=x['range_m']<=5],n) for n in names},'large':{n:stats([x for x in records if x['size']>128],n) for n in names},'small':{n:stats([x for x in records if x['size']==128],n) for n in names},'seconds':time.monotonic()-start};(out/'result.json').write_text(json.dumps(result,indent=2));(out/'records.json').write_text(json.dumps(records,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
