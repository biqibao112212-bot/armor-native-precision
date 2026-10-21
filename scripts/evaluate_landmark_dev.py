"""Frozen DEV-only complete image inference; independent of GT anchors."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np,torch,cv2
from scipy.optimize import linear_sum_assignment
from armor_native.broad import load_localizer,statistics
from armor_native.data import rgb
from armor_native.proposals import NativeProposer
from evaluate_broad import score_frame,summarize

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--weights',required=True);ap.add_argument('--proposal',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();r=Path(a.run);out=Path(a.output);out.mkdir(exist_ok=False);p=Path(__file__).resolve().parents[1];torch.set_num_threads(3);cv2.setNumThreads(1);localizer=load_localizer(a.weights);coarse=NativeProposer(a.proposal,p/'configs/proposal_yolo11n_pose.json',mode='full_native');K=np.array([[1303.675283386667,0,720],[0,1303.675283386667,540],[0,0,1]]);records=[];predictions=[];start=time.monotonic()
 for domain,folder,stride in [('broad','observable-data',4),('legacy','observable-legacy',4)]:
  rows=[json.loads(l) for l in (r/folder/'dev.jsonl').open()][::stride]
  for i,f in enumerate(rows):
   if time.monotonic()-start>900:raise TimeoutError('DEV inference limit')
   im=rgb(f['image_path']);pp=coarse(im,threshold=localizer.proposal_threshold,include_geometry=True);qq=localizer(im,pp,K);pred=[dict(q,center=q.get('center',q['proposal_center']),xyz=q['xyz_m']) for q in qq];predictions.append({'domain':domain,'session':f['session'],'frame_seq':f['frame_seq'],'methods':pred})
   # Legacy manifests have no range-derived height; score_frame accepts None.
   for o in f['objects']:o.setdefault('height_difference_m',None)
   scored=score_frame(f,pred,True)
   for v in scored:v['domain']=domain
   records.extend(scored)
 def table(rr):return {'all':summarize(rr),'primary3_5':summarize([v for v in rr if 3<=v['range_m']<=5]),'by_range':{str(b):summarize([v for v in rr if v['range_bin']==b]) for b in range(8)}}
 result={'scope':'DEV only complete image inference; no oracle crop, no sealed data','candidate_sha256':hashlib.sha256(Path(a.weights).read_bytes()).hexdigest(),'proposal_sha256':hashlib.sha256(Path(a.proposal).read_bytes()).hexdigest(),'frames':len(predictions),'broad':table([v for v in records if v['domain']=='broad']),'legacy':table([v for v in records if v['domain']=='legacy']),'seconds':time.monotonic()-start};(out/'result.json').write_text(json.dumps(result,indent=2));(out/'records.json').write_text(json.dumps(records,indent=2));(out/'predictions.json').write_text(json.dumps(predictions,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
