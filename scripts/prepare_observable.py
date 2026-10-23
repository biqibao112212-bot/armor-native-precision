"""Annotate TRAIN/DEV only with independently specified image evidence."""
import argparse,json,hashlib,shutil,collections,concurrent.futures
from pathlib import Path
import numpy as np,cv2
from armor_native.observability import lamp_evidence,RULE,RULES
from functools import partial

def annotate(row,rule=RULE):
 im=cv2.cvtColor(cv2.imread(row['image_path']),cv2.COLOR_BGR2RGB);keep=[];refused=[]
 for obj in row['objects']:
  o=dict(obj);o['lamp_evidence']=lamp_evidence(im,o['corners'],rule=rule)
  (keep if o['lamp_evidence']['accepted'] else refused).append(o)
 r=dict(row,objects=keep,unobservable_objects=refused)
 r['coarse_ignored_boxes']=list(row.get('coarse_ignored_boxes',[]))+[[*np.min(o['corners'],0).tolist(),*np.max(o['corners'],0).tolist()] for o in refused]
 return r

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',required=True);ap.add_argument('--lamp-rule',choices=sorted(RULES),default=RULE['version']);a=ap.parse_args();src=Path(a.input);out=Path(a.output);out.mkdir(exist_ok=False);cv2.setNumThreads(1);audit={'rule':RULES[a.lamp_rule],'sealed_opened':False,'splits':{}}
 for split in ['train','dev']:
  rows=[json.loads(l) for l in (src/(split+'.jsonl')).open()];assert all(r['split'] in [split,'val'] for r in rows)
  counts=collections.Counter();bands=collections.defaultdict(collections.Counter)
  with (out/(split+'.jsonl')).open('x') as log,concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
   for i,r in enumerate(pool.map(partial(annotate,rule=RULES[a.lamp_rule]),rows)):
    log.write(json.dumps(r)+'\n')
    for label in ['objects','unobservable_objects']:
     counts[label]+=len(r[label])
     for o in r[label]:bands[str(o['range_bin'])][label]+=1
    if (i+1)%1000==0:print(json.dumps({'split':split,'frames':i+1}),flush=True)
  audit['splits'][split]={'frames':len(rows),'counts':dict(counts),'by_range':{k:dict(v) for k,v in bands.items()},'source_sha256':hashlib.sha256((src/(split+'.jsonl')).read_bytes()).hexdigest()}
 if (src/'geometry.json').exists():shutil.copy2(src/'geometry.json',out/'geometry.json')
 (out/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)
if __name__=='__main__':main()
