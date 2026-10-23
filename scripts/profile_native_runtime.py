"""Bounded runtime decomposition on native DEV images; no model changes."""
import argparse,collections,hashlib,json,time
from pathlib import Path
import cv2,numpy as np,torch
import armor_native.quad as qm
from armor_native.proposals import NativeProposer
from armor_native.data import rgb

def stats(v):
 a=np.asarray(v,float)
 return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(a.max())) if len(a) else None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);ap.add_argument('--seconds',type=int,default=240);a=ap.parse_args()
 n=Path(a.root);out=Path(a.output);out.mkdir(exist_ok=False);repo=Path(__file__).resolve().parents[1]
 start=time.monotonic();torch.set_num_threads(3);cv2.setNumThreads(1)
 ck=n/'capacity-extra-control/checkpoint-004000.pt';pk=n/'proposal-frame-statistics.pt'
 manifest=n/'observable-cyan/dev.jsonl';frames=[json.loads(l) for l in manifest.open()]
 groups=collections.defaultdict(list)
 for f in frames:groups[(tuple(f['configured_range_bin']),f['light'],f['mode'])].append(f)
 rng=np.random.default_rng(2026102301);frames=[vv[int(rng.integers(len(vv)))] for _,vv in sorted(groups.items())]
 plan=dict(question='Which native observation stages exceed the new 10 ms requirement?',scope='T4 PyTorch AMP development diagnostic; excludes image decode and acquisition, not NX acceptance',frames=len(frames),warmup=8,max_seconds=a.seconds,weights_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ck,pk]},manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),selection='one seeded frame per configured distance/light/motion cell',rgb_resized=False,decision='Use measured stage costs and batch scaling to choose latency-aware architecture before further training')
 (out/'plan.json').write_text(json.dumps(plan,indent=2))
 p=NativeProposer(str(pk),repo/'configs/proposal_yolo11n_pose.json',mode='full_native');f=qm.QuadLocalizer(str(ck))
 K=np.array([[1303.675283386667,0,720],[0,1303.675283386667,540],[0,0,1]])
 def timed(fn):
  torch.cuda.synchronize();t=time.perf_counter();v=fn();torch.cuda.synchronize();return v,(time.perf_counter()-t)*1000
 images=[rgb(x['image_path']) for x in frames]
 for im in images[:8]:pp=p(im,threshold=.1,include_geometry=True);f(im,pp,K)
 baseline=[]
 for frame,im in zip(frames,images):
  if time.monotonic()-start>a.seconds:raise TimeoutError('profile bound')
  t=time.perf_counter();pp,pm=timed(lambda:p(im,threshold=.1,include_geometry=True));qq,fm=timed(lambda:f(im,pp,K));total=(time.perf_counter()-t)*1000
  baseline.append(dict(session=frame['session'],frame_seq=frame['frame_seq'],coarse_ms=pm,fine_total_ms=fm,total_ms=total,candidates=len(pp),accepted=sum(q['xyz_m'] is not None for q in qq),proposals=pp,corners=[q['corners'] for q in qq]))
 components=collections.defaultdict(list);current={}
 def wrap(obj,name,label,gpu=False):
  original=getattr(obj,name)
  def measured(*args,**kwargs):
   if gpu:torch.cuda.synchronize()
   t=time.perf_counter();v=original(*args,**kwargs)
   if gpu:torch.cuda.synchronize()
   current[label]=current.get(label,0)+(time.perf_counter()-t)*1000
   return v
  setattr(obj,name,measured)
 wrap(p.model,'forward','coarse_forward_ms',True);wrap(f.model,'forward','fine_forward_ms',True)
 wrap(qm,'crop_native','crop_ms');wrap(qm,'lamp_evidence','visibility_ms');wrap(qm,'solve_physical','pnp_ms')
 rows=[];max_delta=0
 for im,base in zip(images,baseline):
  if time.monotonic()-start>a.seconds:raise TimeoutError('profile bound')
  current={};pp,pm=timed(lambda:p(im,threshold=.1,include_geometry=True));qq,fm=timed(lambda:f(im,pp,K))
  assert len(pp)==base['candidates'] and len(qq)==len(base['corners'])
  if qq:max_delta=max(max_delta,float(np.max(np.abs(np.asarray([q['corners'] for q in qq])-base['corners']))))
  row={k:v for k,v in base.items() if k not in ['corners','proposals']};row['instrumented']={**current,'coarse_total_ms':pm,'fine_total_ms':fm}
  row['instrumented']['coarse_other_ms']=pm-current.get('coarse_forward_ms',0)
  row['instrumented']['fine_other_ms']=fm-sum(current.get(k,0) for k in ['fine_forward_ms','crop_ms','visibility_ms','pnp_ms'])
  rows.append(row)
  for k,v in row['instrumented'].items():components[k].append(v)
 (out/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 idx=max(range(len(baseline)),key=lambda j:baseline[j]['candidates']);im=images[idx];pp=baseline[idx]['proposals'];scaling={}
 for count in [1,2,4,8,12]:
  subset=[pp[i%len(pp)] for i in range(count)]
  for _ in range(3):f(im,subset,K)
  vals=[];forward=[]
  for _ in range(12):
   current={};_,elapsed=timed(lambda:f(im,subset,K));vals.append(elapsed);forward.append(current['fine_forward_ms'])
  scaling[str(count)]={'fine_total_ms':stats(vals),'fine_forward_ms':stats(forward),'scope':'Repeated actual proposals only for workload scaling, not accuracy evaluation'}
 result=dict(complete=True,seconds=time.monotonic()-start,device=torch.cuda.get_device_name(),torch_version=torch.__version__,scope=plan['scope'],frames=len(rows),unmodified_stages={k:stats([r[k] for r in rows]) for k in ['coarse_ms','fine_total_ms','total_ms','candidates','accepted']},instrumented_stages={k:stats(v) for k,v in components.items()},instrumentation_corner_max_delta_px=max_delta,fine_candidate_scaling=scaling,by_candidate_count={str(count):{k:stats([r[k] for r in rows if r['candidates']==count]) for k in ['coarse_ms','fine_total_ms','total_ms']} for count in sorted(set(r['candidates'] for r in rows))})
 (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
