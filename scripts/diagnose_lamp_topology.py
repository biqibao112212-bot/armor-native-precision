"""Check an inactive lamp-topology prototype against all current TRAIN positives."""
import argparse, collections, concurrent.futures, hashlib, json, time
from pathlib import Path
import cv2
from armor_native.data import rgb
from armor_native.lamp_topology import separated_lamp_evidence, TOPOLOGY_RULE

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--seconds',type=int,default=240);a=ap.parse_args()
    out=Path(a.output);assert not out.exists();cv2.setNumThreads(1)
    frames=[json.loads(l) for l in Path(a.manifest).open()]
    assert all(f['split']=='train' for f in frames)
    start=time.monotonic()
    def evaluate(f):
        if not f['objects']:return []
        image=rgb(f['image_path']);rows=[]
        for o in f['objects']:
            evidence=separated_lamp_evidence(image,o['corners'])
            rows.append(dict(session=f['session'],frame_seq=f['frame_seq'],slot=o['slot'],
                range_m=o['range_m'],range_bin=o['range_bin'],light=f['light'],
                image_path=f['image_path'],corners=o['corners'],**evidence))
        return rows
    records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for rows in pool.map(evaluate,frames):
            if time.monotonic()-start>a.seconds:raise TimeoutError('TRAIN topology CPU bound')
            records.extend(rows)
    def summary(rows):
        return dict(objects=len(rows),retained=sum(r['accepted'] for r in rows),
                    rejected=sum(not r['accepted'] for r in rows))
    report=dict(complete=True,scope='TRAIN exact existing-positive corners; no change to eligibility or online outputs',
        seconds=time.monotonic()-start,rule=TOPOLOGY_RULE,all=summary(records),
        primary_3_5=summary([r for r in records if 3<=r['range_m']<=5]),
        by_range={str(i):summary([r for r in records if r['range_bin']==i]) for i in range(8)},
        by_light={k:summary([r for r in records if r['light']==k]) for k in ['dark','normal','bright']},
        rejections=[r for r in records if not r['accepted']],
        manifest_sha256=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with out.open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='rejections'}))
if __name__=='__main__':main()
