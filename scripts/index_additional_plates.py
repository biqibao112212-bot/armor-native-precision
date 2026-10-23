"""Index additional number3 plate projections from existing DEV exposure labels."""
import argparse, collections, hashlib, json, time
from pathlib import Path
import numpy as np
from prepare_broad import physical_order

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--labels',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--seconds',type=int,default=240)
    a=ap.parse_args();out=Path(a.output);assert not out.exists()
    wanted={}
    for line in Path(a.manifest).open():
        f=json.loads(line);assert f['split']=='dev'
        k=tuple(f[n] for n in ['producer_epoch','frame_seq','timestamp_ns'])
        assert k not in wanted;wanted[k]=f
    indexed=collections.defaultdict(list);source_hash=hashlib.sha256();start=time.monotonic()
    label_path=Path(a.labels);stat=label_path.stat()
    with label_path.open('rb') as stream:
        for i,line in enumerate(stream):
            if i%10000==0 and time.monotonic()-start>a.seconds:raise TimeoutError('bounded label-index pass')
            source_hash.update(line)
            l=json.loads(line);k=tuple(l[n] for n in ['producer_epoch','frame_seq','timestamp_ns'])
            if k not in wanted or l['target_number']!=3 or l['target_id']==wanted[k]['target_id']:continue
            v=l['visibility']
            if v['classification']!='fully_in_frame' or v['scene_hidden']:continue
            c=np.asarray(l['exact_corners_px'])[physical_order(l['plate_geometry']['object_corners_armor_m'])]
            indexed[k].append(dict(target_id=l['target_id'],slot=l['relative_slot'],
                corners=c.tolist(),center=((c.min(0)+c.max(0))/2).tolist(),
                extent_px=float(np.ptp(c,axis=0).max()),range_m=l['distance_m'],
                occlusion_tested=v['occlusion_tested']))
    assert label_path.stat().st_size==stat.st_size and label_path.stat().st_mtime_ns==stat.st_mtime_ns
    report=dict(complete=True,scope='Existing DEV other-number3 exact projections only; not primary performance population',
                seconds=time.monotonic()-start,source_path=str(label_path),source_sha256=source_hash.hexdigest(),
                manifest_sha256=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
                frames=[dict(session=f['session'],frame_seq=f['frame_seq'],objects=indexed[k])
                        for k,f in wanted.items()])
    with out.open('x') as stream:json.dump(report,stream)
    print(json.dumps(dict(complete=True,seconds=report['seconds'],frames=len(wanted),
                         additional_projections=sum(len(v) for v in indexed.values()),
                         source_sha256=report['source_sha256'])))

if __name__=='__main__':main()
