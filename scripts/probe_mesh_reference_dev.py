"""Full DEV diagnostic reference using sealed image-based neural initializations.

Only predicted range routes the expensive CPU search. Coverage is held at the
baseline for this mechanism study; no refusal is introduced to hide failures.
This is not an operational model or an inference-time claim.
"""
import argparse,json,time,copy,collections,hashlib
from pathlib import Path
import numpy as np,cv2
from armor_native.data import rgb
from armor_native.broad import statistics
from evaluate_broad import score_frame,summarize
from probe_lamp_visibility import Renderer
from probe_mesh_refinement import image_refine

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--predictions',required=True);ap.add_argument('--geometry',required=True);ap.add_argument('--asset',required=True);ap.add_argument('--library',required=True);ap.add_argument('--output',required=True);ap.add_argument('--seconds',type=int,default=2400);a=ap.parse_args()
    out=Path(a.output);out.mkdir(exist_ok=False);cv2.setNumThreads(1)
    frames={(r['session'],r['frame_seq']):r for r in map(json.loads,Path(a.data).open())};assert frames and all(r['split']=='dev' for r in frames.values())
    template=json.loads(Path(a.geometry).read_text())['object_points_armor_m'];render=Renderer(a.library,a.asset,include_digit=True)
    source=[json.loads(l) for l in Path(a.predictions).open()];assert {(r['session'],r['frame_seq']) for r in source}==set(frames)
    plan={'scope':'DEV image-only geometric feasibility reference, not deployed network','predicted_range_route_m':[2.5,5.5],'input_predictions':a.predictions,'input_sha256':hashlib.sha256(Path(a.predictions).read_bytes()).hexdigest(),'camera_attitude':'public same-exposure metadata','target_truth_in_refinement':False,'baseline_coverage_held_fixed':True,'no_gating_by_error':True,'seconds':a.seconds}
    (out/'plan.json').write_text(json.dumps(plan,indent=2));start=time.monotonic();records=collections.defaultdict(list);calls=0;costs=[];timings=[]
    with (out/'image-only.jsonl').open('x') as log:
        for i,input_row in enumerate(source):
            if time.monotonic()-start>a.seconds:raise TimeoutError('bounded DEV mesh reference')
            key=(input_row['session'],input_row['frame_seq']);f=frames[key];base=input_row['methods']['plate_control'];level=copy.deepcopy(base);mesh=copy.deepcopy(base);im=None;diagnostics=[]
            # The following call receives camera metadata and model predictions,
            # never target corners, range, pose, slot, or target identity.
            K=np.array([[1303.675283386667,0,720],[0,1303.675283386667,540],[0,0,1.]])
            for j,q in enumerate(base):
                xyz=q.get('xyz')
                if xyz is None or not 2.5<=np.linalg.norm(xyz)<=5.5:continue
                if im is None:im=rgb(f['image_path'])
                result=image_refine(im,q['corners'],K,template,f['camera_quaternion_wxyz'],render,maxiter=40,popsize=8,seed=9001+i*11+j);calls+=1
                if result is None:diagnostics.append({'index':j,'status':'no_image_refinement','fallback':'baseline'});continue
                for target,pose in [(level,result['level_initial']),(mesh,result)]:
                    target[j].update(corners=pose['corners'],xyz=pose['xyz_m'],xyz_m=pose['xyz_m'],raw_xyz_m=pose['xyz_m'],center=((np.min(pose['corners'],0)+np.max(pose['corners'],0))/2).tolist())
                diagnostics.append({'index':j,**{k:result[k] for k in ['initial_loss','final_loss','observed_pixels','digit_used','evaluations','seconds']}})
                costs.append(result['final_loss']);timings.append(result['seconds'])
            methods={'plate_base':base,'level_reference':level,'mesh_reference':mesh}
            log.write(json.dumps({'session':key[0],'frame_seq':key[1],'methods':methods,'diagnostics':diagnostics})+'\n');log.flush()
            for name,values in methods.items():records[name].extend(score_frame(f,values,True))
            if (i+1)%64==0:print(json.dumps({'frames':i+1,'refinement_calls':calls,'elapsed_s':time.monotonic()-start}),flush=True)
    result={'complete':True,'development_only':True,'scope':plan['scope'],'frames':len(source),'refinement_calls':calls,'seconds':time.monotonic()-start,'methods':{name:{'all':summarize(rr),'primary_3_5':summarize([r for r in rr if 3<=r['range_m']<=5]),'by_range':{str(b):summarize([r for r in rr if r['range_bin']==b]) for b in range(8)}} for name,rr in records.items()},'refinement_seconds':statistics(timings),'final_image_cost':statistics(costs)}
    with (out/'records.jsonl').open('x') as log:
        for name,rr in records.items():
            for row in rr:log.write(json.dumps(dict(row,method=name))+'\n')
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({name:v['primary_3_5'] for name,v in result['methods'].items()}),flush=True)
if __name__=='__main__':main()
