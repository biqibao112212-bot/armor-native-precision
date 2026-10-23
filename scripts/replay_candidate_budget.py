"""Replay score-ranked candidate budgets without rerunning or changing models."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from evaluate_broad import score_frame,summarize
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();n=Path(a.root)
 frames={(f['session'],f['frame_seq']):f for f in map(json.loads,(n/'observable-cyan/dev.jsonl').open())}
 data=[json.loads(l) for l in (n/'capacity-dev/image-only.jsonl').open()];scored={k:[] for k in [1,2,4,8,12]}
 for f in data:
  pp=sorted(f['coarse_by_model']['capacity_control'],key=lambda p:p['score'],reverse=True)
  qq=f['methods']['capacity_control']
  for count,rr in scored.items():
   centers={tuple(p['center']) for p in pp[:count]}
   chosen=[q for q in qq if tuple(q['proposal_center']) in centers]
   rr.extend(score_frame(frames[f['session'],f['frame_seq']],chosen,True))
 result={'scope':'Post hoc DEV replay of current fixed outputs, score-ranked top K; image-only ranking, same eligibility, no retraining. Timing not measured for this replay. Not deployment acceptance.','budgets':{str(k):{'all':summarize(rr),'primary_3_5':summarize([r for r in rr if 3<=r['range_m']<=5]),'by_range':{str(b):summarize([r for r in rr if r['range_bin']==b]) for b in range(8)}} for k,rr in scored.items()}}
 original=json.loads((n/'capacity-dev/result.json').read_text())['methods']['capacity_control']
 assert result['budgets']['12']['primary_3_5']==original['primary_3_5']
 assert result['budgets']['12']['all']==original['all']
 with Path(a.output).open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps({k:v['primary_3_5'] for k,v in result['budgets'].items()},indent=2))
if __name__=='__main__':main()
