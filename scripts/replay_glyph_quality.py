"""Bounded DEV replay of unchanged image landmarks with declared quality rules.

Only stored network outputs and public camera metadata enter postprocessing.
All original eligible objects remain in scoring. This is not a new test set.
"""
import argparse,copy,json,time,collections,hashlib
from pathlib import Path
import numpy as np
from armor_native.glyph import fit_glyph_pose
from armor_native.glyph_quality import DEFAULT_CONSENSUS,landmark_consensus,restore_preduplicate,suppress_duplicates
from evaluate_broad import score_frame
from evaluate_recognition_confirm import report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evaluation',required=True);ap.add_argument('--manifest',required=True)
    ap.add_argument('--geometry',required=True);ap.add_argument('--output',required=True);ap.add_argument('--method',default='final_level')
    ap.add_argument('--seconds',type=int,default=600);a=ap.parse_args()
    src=Path(a.evaluation);assert json.loads((src/'result.json').read_text())['complete'];out=Path(a.output);out.mkdir(exist_ok=False)
    manifest=Path(a.manifest);frames={(f['session'],f['frame_seq']):f for f in map(json.loads,manifest.open())}
    source_plan=json.loads((src/'plan.json').read_text());assert hashlib.sha256(manifest.read_bytes()).hexdigest()==source_plan['manifest_sha256']
    template=json.loads(Path(a.geometry).read_text())['object_points']
    plan={'scope':'DEV postprocessing replay; no network retraining, no sealed test accessed',
        'input_prediction_sha256':hashlib.sha256((src/'image-only.jsonl').read_bytes()).hexdigest(),
        'manifest_sha256':source_plan['manifest_sha256'],'consensus':DEFAULT_CONSENSUS,'duplicate_relative_extent':.1,
        'rgb_resized':False,'population_unchanged':True,'target_truth_in_postprocessing':False,'live_parity_required':True}
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    start=time.monotonic();records=collections.defaultdict(list);seen=set();max_delta=0.;count=0
    with (out/'image-only.jsonl').open('x') as log:
        for i,row in enumerate(map(json.loads,(src/'image-only.jsonl').open())):
            if time.monotonic()-start>a.seconds:raise TimeoutError('quality replay budget')
            key=row['session'],row['frame_seq'];assert key not in seen;seen.add(key);f=frames[key]
            fx,fy,cx,cy=f['intrinsics'];K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
            base=row['methods'][a.method];raw=[restore_preduplicate(q) for q in base]
            quality=copy.deepcopy(raw)
            for q in quality:
                if q['xyz_m'] is None or 'landmarks' not in q:continue
                projected=None
                if q.get('decoder_source')=='glyph_level':
                    pose=fit_glyph_pose(q['landmarks'],q['point_probabilities'],K,template,f['camera_quaternion_wxyz'],'glyph_level')
                    assert pose is not None
                    delta=float(np.max(np.abs(np.asarray(pose['corners'])-q['corners'])));max_delta=max(max_delta,delta)
                    assert delta<.002,(key,delta)
                    projected=pose.get('projected_landmarks');count+=1
                gate=landmark_consensus(q['landmarks'],q['point_probabilities'],projected)
                q['landmark_consensus']=gate
                if not gate['accepted']:
                    q['xyz']=q['xyz_m']=None;q['image_evidence']=dict(q['image_evidence'],accepted=False,reason=gate['reason'])
            methods={'baseline':base,'relative_nms':suppress_duplicates(copy.deepcopy(raw),.1),
                     'consensus':suppress_duplicates(copy.deepcopy(quality),0.),
                     'consensus_nms':suppress_duplicates(copy.deepcopy(quality),.1)}
            pp=row['coarse_by_model'][a.method]
            log.write(json.dumps({'session':key[0],'frame_seq':key[1],'methods':methods,'coarse_proposals':pp,'coarse_by_model':{n:pp for n in methods}})+'\n');log.flush()
            for name,qq in methods.items():records[name].extend(score_frame(f,qq,True))
            if (i+1)%128==0:print(json.dumps({'frames':i+1,'pose_replays':count,'seconds':time.monotonic()-start}),flush=True)
    assert seen==set(frames)
    with (out/'records.jsonl').open('x') as log:
        for name,rr in records.items():
            for row in rr:log.write(json.dumps(dict(row,method=name))+'\n')
    result={'complete':True,'development_only':True,'frames':len(frames),'seconds':time.monotonic()-start,
        'maximum_replayed_corner_delta_px':max_delta,'pose_replays':count,'methods':{n:report(rr) for n,rr in records.items()},
        'live_parity_validated':False}
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({n:v['primary_3_5'] for n,v in result['methods'].items()}),flush=True)
if __name__=='__main__':main()
