"""Sealed recognition confirmation after all candidate and threshold choices freeze.

No predictions use target labels. Persist and hash every image-only output
before scoring. The eligibility rule is fixed independently of model refusal.
"""
import argparse,collections,hashlib,json,time,sys
from pathlib import Path
import cv2,numpy as np,torch
from armor_native.data import rgb
from armor_native.broad import load_localizer,statistics
from armor_native.proposals import NativeProposer
from armor_native.observability import lamp_evidence,CYAN_RULE
from evaluate_broad import score_frame,summarize
from collect_level_confirmation import verify_freeze

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def report(rr):
    return {'all':summarize(rr),'primary_3_5':summarize([r for r in rr if 3<=r['range_m']<=5]),
        'by_range':{str(i):summarize([r for r in rr if r['range_bin']==i]) for i in range(8)},
        'by_light':{x:summarize([r for r in rr if r['light']==x]) for x in ['dark','normal','bright']},
        'by_facing':{n:summarize([r for r in rr if lo<=r['facing_cos']<hi]) for n,lo,hi in [('12-25',.12,.25),('25-50',.25,.5),('50-75',.5,.75),('75-100',.75,1.001)]},
        'by_height':{n:summarize([r for r in rr if lo<=abs(r['height_difference_m'])<hi]) for n,lo,hi in [('0-.15m',0,.15),('.15-.3m',.15,.3),('.3-.4m',.3,.40001)]},
        'by_motion':{x:summarize([r for r in rr if r['mode']==x]) for x in ['stationary','linear','spin','combined']}}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True);ap.add_argument('--freeze-receipt',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--consumer-source')
    ap.add_argument('--seconds',type=float,default=2400);a=ap.parse_args()
    frozen=verify_freeze(a.freeze_receipt)
    assets={'candidate':frozen['candidate'],**frozen.get('comparators',{})}
    for name,x in assets.items():assert sha(x['path'])==x['sha256'],name+' hash mismatch'
    root=Path(a.data);frames=[json.loads(x) for x in (root/'sealed.jsonl').open()]
    assert frames and all(f['split']=='sealed' for f in frames)
    assert len({(f['session'],f['frame_seq']) for f in frames})==len(frames)
    assert all(abs(o['height_difference_m'])<=.40001 for f in frames for o in f['objects'])
    out=Path(a.output);out.mkdir(exist_ok=False)
    torch.set_num_threads(3);cv2.setNumThreads(1)
    nets={name:load_localizer(asset['path']) for name,asset in assets.items()}
    repo=Path(__file__).resolve().parents[1]
    coarse=NativeProposer(frozen['proposal']['path'],repo/'configs/proposal_yolo11n_pose.json',mode='full_native')
    szu=None
    if a.consumer_source:
        sys.path.insert(0,a.consumer_source)
        from diagnose_baselines import detector,MODEL
        import onnxruntime as ort
        szu=ort.InferenceSession(str(MODEL),providers=[('CUDAExecutionProvider',{'cudnn_conv_algo_search':'HEURISTIC'}),'CPUExecutionProvider'])
        assert szu.get_providers()[0]=='CUDAExecutionProvider'
        (out/'szu-provenance.json').write_text(json.dumps({'model_sha256':sha(MODEL),'native_tiles':9,'tile_size':640,'threshold':.5,'frozen_scalar':1.04221015588693},indent=2))
    plan={'scope':'new independent sealed single-frame recognition confirmation','freeze_receipt':frozen,
        'source_sha256':sha(__file__),'manifest_sha256':sha(root/'sealed.jsonl'),'frames':len(frames),
        'rgb_resized':False,'image_only_predictions_before_scoring':True,
        'eligibility_rule':CYAN_RULE,'no_model_based_population_filter':True,
        'camera_attitude_input':{n:bool(getattr(net,'supports_camera_attitude',False)) for n,net in nets.items()},
        'absolute_targets':{'corner_p95_px':.236,'depth_p95_mm':74.3,'xy_p95_mm':10.6,'coverage_reference':812/817}}
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    start=time.monotonic();latencies=collections.defaultdict(list)
    with (out/'image-only.jsonl').open('x') as log:
        for i,f in enumerate(frames):
            if time.monotonic()-start>a.seconds:raise TimeoutError('confirmation compute budget')
            im=rgb(f['image_path']);fx,fy,cx,cy=f['intrinsics'];K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
            torch.cuda.synchronize();t=time.perf_counter();pp=coarse(im,threshold=.1,include_geometry=True);torch.cuda.synchronize();coarse_ms=(time.perf_counter()-t)*1000
            methods={};times={}
            for name,net in nets.items():
                t=time.perf_counter()
                qq=net(im,pp,K,f['camera_quaternion_wxyz']) if getattr(net,'supports_camera_attitude',False) else net(im,pp,K)
                torch.cuda.synchronize();times[name]=coarse_ms+(time.perf_counter()-t)*1000
                methods[name]=[dict(q,xyz=q['xyz_m']) for q in qq]
            if szu is not None:
                t=time.perf_counter();methods['szu_native']=detector(szu,im,True);torch.cuda.synchronize();times['szu_native']=(time.perf_counter()-t)*1000
                methods['szu_frozen_scalar']=[dict(q,xyz=None if q['xyz'] is None else (np.asarray(q['xyz'])*1.04221015588693).tolist()) for q in methods['szu_native']]
                times['szu_frozen_scalar']=times['szu_native']
            for name,ms in times.items():latencies[name].append(ms)
            log.write(json.dumps({'session':f['session'],'frame_seq':f['frame_seq'],'coarse_proposals':pp,'methods':methods,'latency_ms':times})+'\n');log.flush()
            if (i+1)%128==0:print(json.dumps({'stage':'image_only','frames':i+1,'elapsed_s':time.monotonic()-start}),flush=True)
    seal={'sha256':sha(out/'image-only.jsonl'),'frames':len(frames),'target_truth_in_inference':False}
    (out/'prediction-seal.json').write_text(json.dumps(seal,indent=2))
    records=collections.defaultdict(list);unsupported=collections.defaultdict(list);counts=collections.defaultdict(collections.Counter)
    with (out/'image-only.jsonl').open() as predictions:
        for f,line in zip(frames,predictions):
            p=json.loads(line);assert (p['session'],p['frame_seq'])==(f['session'],f['frame_seq'])
            im=rgb(f['image_path']);annotated=[]
            for obj in f['objects']:annotated.append(dict(obj,lamp_evidence=lamp_evidence(im,obj['corners'],rule=CYAN_RULE)))
            eligible=dict(f,objects=[o for o in annotated if o['lamp_evidence']['accepted']])
            unobservable=dict(f,objects=[o for o in annotated if not o['lamp_evidence']['accepted']])
            for name,qq in p['methods'].items():
                physical=name not in ['szu_native','szu_frozen_scalar']
                scored=score_frame(eligible,qq,physical);records[name].extend(scored)
                unsupported[name].extend(score_frame(unobservable,qq,physical))
                counts[name].update(proposals=len(qq),accepted=sum(q.get('xyz') is not None for q in qq),
                    eligible_matches=sum(r['detected'] for r in scored))
    all_records=[dict(r,method=n) for n,rr in records.items() for r in rr]
    with (out/'records.jsonl').open('x') as f:
        for row in all_records:f.write(json.dumps(row)+'\n')
    methods={n:report(rr) for n,rr in records.items()}
    primary=methods['candidate']['primary_3_5']
    checks={key:primary[metric].get('p95') is not None and primary[metric]['p95']<=limit for key,metric,limit in [
        ('corner','corner_rms_px',.236),('depth','depth_mm',74.3),('xy','xy_mm',10.6)]}
    checks['coverage']=primary['coverage']>=812/817
    result={'complete':True,'independent_confirmation':True,'frames':len(frames),'seconds':time.monotonic()-start,
        'methods':methods,'unsupported_geometry_only':{n:report(rr) for n,rr in unsupported.items()},
        'counts':{n:dict(c) for n,c in counts.items()},'unmatched_outputs_are_not_all_confirmed_false_positives':True,
        'primary_acceptance_checks':checks,'primary_pass':all(checks.values()),
        'prediction_evaluation_performed':False,'latency_ms':{n:statistics(v) for n,v in latencies.items()},
        'latency_scope':'T4 PyTorch AMP observation only, excludes acquisition/queue, not NX acceptance'}
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'primary_pass':result['primary_pass'],'methods':{n:v['primary_3_5'] for n,v in methods.items()}}),flush=True)
if __name__=='__main__':main()
