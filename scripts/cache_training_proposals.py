"""Freeze image-only proposals on TRAIN, then join labels offline for sampling."""
import argparse,json,time,hashlib,collections,shutil
from pathlib import Path
import torch,cv2,numpy as np
from armor_native.proposals import NativeProposer
from armor_native.data import rgb
from armor_native.matching import match_centers

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--proposal',required=True);ap.add_argument('--output',required=True);ap.add_argument('--seconds',type=int,default=900);a=ap.parse_args();root=Path(a.data);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 rows=[json.loads(l) for l in (root/'train.jsonl').open()];assert all(r['split']=='train' for r in rows);torch.set_num_threads(3);cv2.setNumThreads(1)
 repo=Path(__file__).resolve().parents[1];net=NativeProposer(a.proposal,repo/'configs/proposal_yolo11n_pose.json',mode='full_native');start=time.monotonic()
 with (out/'image-only-proposals.jsonl').open('x') as f:
  for i,r in enumerate(rows):
   if time.monotonic()-start>a.seconds:raise TimeoutError('TRAIN coarse cache budget')
   pp=net(rgb(r['image_path']),threshold=.1,include_geometry=True)
   f.write(json.dumps({'session':r['session'],'frame_seq':r['frame_seq'],'proposals':pp})+'\n')
   if (i+1)%512==0:f.flush();print(json.dumps({'frames':i+1,'seconds':time.monotonic()-start}),flush=True)
 seal={'source_train_sha256':hashlib.sha256((root/'train.jsonl').read_bytes()).hexdigest(),'proposal_sha256':hashlib.sha256(Path(a.proposal).read_bytes()).hexdigest(),'predictions_sha256':hashlib.sha256((out/'image-only-proposals.jsonl').read_bytes()).hexdigest(),'threshold':.1,'RGB_resized':False,'GT_at_inference':False};(out/'seal.json').write_text(json.dumps(seal,indent=2))
 counts=collections.Counter();by_range=collections.defaultdict(collections.Counter);errors=collections.defaultdict(list)
 with (out/'train.jsonl').open('x') as dst:
  for r,line in zip(rows,(out/'image-only-proposals.jsonl').open()):
   pr=json.loads(line);assert (r['session'],r['frame_seq'])==(pr['session'],pr['frame_seq']);pp=[dict(p,xyz=[0,0,1]) for p in pr['proposals']];pairs=match_centers(pp,r['objects'])
   for j,o in enumerate(r['objects']):
    counts['objects']+=1;by_range[o['range_bin']]['objects']+=1
    if j in pairs:
     anchors=pp[pairs[j]]['coarse_corners'];o['training_proposal_corners']=anchors;o['training_proposal_score']=pp[pairs[j]]['score'];counts['matched']+=1;by_range[o['range_bin']]['matched']+=1
     errors[o['range_bin']].append(float(np.abs(np.asarray(anchors)-o['corners']).max()))
   dst.write(json.dumps(r)+'\n')
 shutil.copyfile(root/'geometry.json',out/'geometry.json')
 result={'complete':True,'frames':len(rows),'seconds':time.monotonic()-start,'counts':dict(counts),'by_range':{str(k):dict(v,anchor_max_linf_px_p95=float(np.quantile(errors[k],.95)) if errors[k] else None) for k,v in by_range.items()},'source_split':'TRAIN only, never DEV or test'}
 (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
