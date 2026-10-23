"""All observable DEV episodes, image-only coarse proposals, fixed thresholds."""
import argparse,json,time,hashlib,collections,sys
from pathlib import Path
import cv2,numpy as np,torch
from armor_native.quad import QuadLocalizer
from armor_native.landmark import LandmarkNativeLocalizer
from armor_native.proposals import NativeProposer
from armor_native.data import rgb
from armor_native.broad import statistics,load_localizer
from evaluate_broad import score_frame,summarize

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--proposal',required=True);ap.add_argument('--models',nargs='+',required=True);ap.add_argument('--names',nargs='+',required=True);ap.add_argument('--output',required=True);ap.add_argument('--seconds',type=int,default=900);ap.add_argument('--model-proposals',nargs='+');ap.add_argument('--consumer-source',help='Optional existing private SZU baseline decoder directory');a=ap.parse_args()
 assert len(a.models)==len(a.names);proposal_paths=a.model_proposals or [a.proposal]*len(a.models);assert len(proposal_paths)==len(a.models);assert len(set(a.names))==len(a.names);assert not set(a.names)&{'szu_native','szu_frozen_scalar'};out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(3);cv2.setNumThreads(1);root=Path(a.data);frames=[json.loads(l) for l in (root/'dev.jsonl').open()];assert frames and all(r['split']=='dev' for r in frames)
 nets=[]
 for weights in a.models:
  ck=torch.load(weights,map_location='cpu',weights_only=False)
  nets.append(load_localizer(weights))
 repo=Path(__file__).resolve().parents[1];coarse_nets={p:NativeProposer(p,repo/'configs/proposal_yolo11n_pose.json',mode='full_native') for p in dict.fromkeys(proposal_paths)}
 szu=None
 if a.consumer_source:
  sys.path.insert(0,a.consumer_source);from diagnose_baselines import detector,MODEL
  import onnxruntime as ort
  szu=ort.InferenceSession(str(MODEL),providers=[('CUDAExecutionProvider',{'cudnn_conv_algo_search':'HEURISTIC'}),'CPUExecutionProvider']);assert szu.get_providers()[0]=='CUDAExecutionProvider'
  (out/'szu-provenance.json').write_text(json.dumps({'model_sha256':hashlib.sha256(Path(MODEL).read_bytes()).hexdigest(),'native_tiles':9,'tile_size':640,'threshold':.5,'frozen_scalar':1.04221015588693},indent=2))
 K=np.array([[1303.675283386667,0,720],[0,1303.675283386667,540],[0,0,1]]);start=time.monotonic();records=collections.defaultdict(list);latencies=collections.defaultdict(list)
 plan={'scope':'All corrected-scope DEV images; no sealed-test selection','frames':len(frames),'models':{n:{'path':w,'sha256':hashlib.sha256(Path(w).read_bytes()).hexdigest()} for n,w in zip(a.names,a.models)},'manifest_sha256':hashlib.sha256((root/'dev.jsonl').read_bytes()).hexdigest(),'model_proposals':{n:{'path':w,'sha256':hashlib.sha256(Path(w).read_bytes()).hexdigest()} for n,w in zip(a.names,proposal_paths)},'rgb_resized':False,'threshold':.1,'absolute_targets':{'corner_p95_px':.236,'depth_p95_mm':74.3,'xy_p95_mm':10.6,'coverage_reference':812/817}}
 plan['same_exposure_camera_attitude_input']={name:bool(getattr(net,'supports_camera_attitude',False)) for name,net in zip(a.names,nets)}
 (out/'plan.json').write_text(json.dumps(plan,indent=2))
 with (out/'image-only.jsonl').open('x') as log:
  for i,f in enumerate(frames):
   if time.monotonic()-start>a.seconds:raise TimeoutError('bounded full DEV')
   im=rgb(f['image_path']);proposals={};coarse_ms={};methods={};times={}
   for path,net in coarse_nets.items():
    torch.cuda.synchronize();t=time.perf_counter();proposals[path]=net(im,threshold=.1,include_geometry=True);torch.cuda.synchronize();coarse_ms[path]=(time.perf_counter()-t)*1000
   for name,net,path in zip(a.names,nets,proposal_paths):
    t=time.perf_counter();qq=net(im,proposals[path],K,f['camera_quaternion_wxyz']) if getattr(net,'supports_camera_attitude',False) else net(im,proposals[path],K);torch.cuda.synchronize();times[name]=coarse_ms[path]+(time.perf_counter()-t)*1000;methods[name]=[dict(q,xyz=q['xyz_m']) for q in qq]
   if szu is not None:
    t=time.perf_counter();methods['szu_native']=detector(szu,im,True);torch.cuda.synchronize();times['szu_native']=(time.perf_counter()-t)*1000
    methods['szu_frozen_scalar']=[dict(q,xyz=None if q['xyz'] is None else (np.array(q['xyz'])*1.04221015588693).tolist()) for q in methods['szu_native']];times['szu_frozen_scalar']=times['szu_native']
   for name,ms in times.items():latencies[name].append(ms)
   log.write(json.dumps({'session':f['session'],'frame_seq':f['frame_seq'],'coarse_proposals':proposals[proposal_paths[0]],'coarse_by_model':{n:proposals[p] for n,p in zip(a.names,proposal_paths)},'methods':methods,'latency_ms':times})+'\n');log.flush()
   # Inference finishes before any GT scoring for this frame.
   for name,qq in methods.items():records[name].extend(score_frame(f,qq,name not in ['szu_native','szu_frozen_scalar']))
   if (i+1)%128==0:print(json.dumps({'frames':i+1,'elapsed_s':time.monotonic()-start}),flush=True)
 def report(rr):
  return {'all':summarize(rr),'primary_3_5':summarize([r for r in rr if 3<=r['range_m']<=5]),'by_range':{str(i):summarize([r for r in rr if r['range_bin']==i]) for i in range(8)},'by_light':{l:summarize([r for r in rr if r['light']==l]) for l in ['dark','normal','bright']},'by_motion':{m:summarize([r for r in rr if r['mode']==m]) for m in ['stationary','linear','spin','combined']},'by_height':{name:summarize([r for r in rr if lo<=abs(r['height_difference_m'])<hi]) for name,lo,hi in [('0-.2m',0,.2),('.2-.5m',.2,.5),('.5m+',.5,10)]},'by_facing':{name:summarize([r for r in rr if lo<=r['facing_cos']<hi]) for name,lo,hi in [('12-25',.12,.25),('25-50',.25,.5),('50-75',.5,.75),('75-100',.75,1.001)]}}
 identity=lambda r:(r['session'],r['frame_seq'],r['slot'])
 common=set.intersection(*[{identity(r) for r in rr if r['valid_pnp']} for rr in records.values()])
 result={'complete':True,'frames':len(frames),'seconds':time.monotonic()-start,'development_only':True,'methods':{n:report(rr) for n,rr in records.items()},'common_valid_objects':len(common),'common_valid_only':{n:report([r for r in rr if identity(r) in common]) for n,rr in records.items()},'latency_ms':{n:statistics(v) for n,v in latencies.items()},'latency_scope':'Observation compute only; excludes image decode/acquisition/queue; T4, not NX'}
 (out/'result.json').write_text(json.dumps(result,indent=2))
 with (out/'records.jsonl').open('x') as f:
  for name,rr in records.items():
   for r in rr:f.write(json.dumps(dict(r,method=name))+'\n')
 print(json.dumps({n:v['primary_3_5'] for n,v in result['methods'].items()},indent=2))
if __name__=='__main__':main()
