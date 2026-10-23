"""Diagnose full DEV glyph errors; never modifies gates or scoring population."""
import argparse,collections,json,hashlib
from pathlib import Path
import cv2,numpy as np
from armor_native.matching import match_centers
from armor_native.broad import solve_physical,statistics
from evaluate_broad import summarize

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--evaluation',required=True);ap.add_argument('--manifest',required=True)
    ap.add_argument('--geometry',required=True);ap.add_argument('--method',required=True)
    ap.add_argument('--output',required=True);a=ap.parse_args()
    root=Path(a.evaluation);assert json.loads((root/'result.json').read_text())['complete']
    source=Path(a.manifest);plan=json.loads((root/'plan.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest()==plan['manifest_sha256']
    frames={(f['session'],f['frame_seq']):f for f in map(json.loads,source.open())}
    template=np.array(json.loads(Path(a.geometry).read_text())['object_points'])
    rows=[];point_errors=[];rejects=[];counts=collections.Counter();seen=set()
    for pred in map(json.loads,(root/'image-only.jsonl').open()):
        key=pred['session'],pred['frame_seq'];assert key not in seen;seen.add(key)
        f=frames[key];pp=pred['methods'][a.method];pairs=match_centers(pp,f['objects'])
        counts['outputs']+=len(pp);counts['accepted_outputs']+=sum(p['xyz'] is not None for p in pp)
        counts['matched_eligible']+=len(pairs)
        for j,g in enumerate(f['objects']):
            if j not in pairs:counts['unmatched_eligible']+=1;continue
            p=pp[pairs[j]];gt=np.asarray(g['corners']);K=np.asarray(g['K'])
            r={'session':key[0],'frame_seq':key[1],'slot':g['slot'],'range_m':g['range_m'],'range_bin':g['range_bin'],'light':f['light'],'height_difference_m':g['height_difference_m'],'facing_cos':g['facing_cos'],'accepted':p['xyz'] is not None,'corner_rms_px':float(np.sqrt(np.mean((np.asarray(p['corners'])-gt)**2))),'decoder':p.get('decoder_source'),'used_points':p.get('used_points'),'reprojection_rms_px':p.get('reprojection_rms_px')}
            raw=p.get('raw_xyz_m',p.get('raw_xyz'))
            if raw is not None:
                d=(np.asarray(raw)-g['xyz_m'])*1000
                r.update(raw_depth_mm=float(abs(d[2])),raw_xy_mm=float(np.linalg.norm(d[:2])))
            if 'landmarks' in p:
                pose=solve_physical(gt,K,template[:4]);assert pose is not None
                truth=cv2.projectPoints(template,np.asarray(pose['rotation_vector']),np.asarray(pose['xyz_m']),K,np.zeros(5))[0].reshape(-1,2)
                delta=np.asarray(p['landmarks'])-truth
                r.update(direct_lamp_rms_px=float(np.sqrt(np.mean(delta[:4]**2))),digit_rms_px=float(np.sqrt(np.mean(delta[4:]**2))),point_probabilities=p['point_probabilities'])
                if 3<=g['range_m']<=5:point_errors.append(delta)
            if not r['accepted']:
                r['image_evidence']=p.get('image_evidence')
                r['image_path']=f['image_path'];rejects.append(r)
            rows.append(r)
    assert seen==set(frames)
    primary=[r for r in rows if 3<=r['range_m']<=5]
    def summarize_raw(rr):
        return {'n':len(rr),'accepted':sum(r['accepted'] for r in rr),**{k:statistics([r[k] for r in rr if r.get(k) is not None]) for k in ['corner_rms_px','raw_depth_mm','raw_xy_mm','direct_lamp_rms_px','digit_rms_px','reprojection_rms_px']}}
    result={'complete':True,'scope':'DEV mechanism diagnosis including rejected predictions; raw metrics are NOT operational acceptance scores','method':a.method,'counts':dict(counts),'all':summarize_raw(rows),'primary_3_5':summarize_raw(primary),'primary_by_light':{k:summarize_raw([r for r in primary if r['light']==k]) for k in ['dark','normal','bright']},'primary_by_view':{k:summarize_raw([r for r in primary if lo<=r['facing_cos']<hi]) for k,lo,hi in [('frontal',.866,1.001),('moderate',.5,.866),('oblique',.12,.5)]},'rejections':rejects,'primary_worst_xy':sorted([r for r in primary if 'raw_xy_mm' in r],key=lambda r:-r['raw_xy_mm'])[:20]}
    if point_errors:
        e=np.asarray(point_errors);result['primary_landmark_errors']=[{'point':i,'bias_uv_px':e[:,i].mean(0).tolist(),'rms_coordinate_px':statistics(np.sqrt(np.mean(e[:,i]**2,axis=1)))} for i in range(len(template))]
    out=Path(a.output);assert not out.exists();out.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ['method','counts','primary_3_5','primary_by_view']}))
if __name__=='__main__':main()
