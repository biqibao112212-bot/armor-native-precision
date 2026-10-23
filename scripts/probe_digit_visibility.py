"""TRAIN oracle-pose control for observed numeral and lamp contours."""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np
from armor_native.data import rgb
from armor_native.broad import solve_physical,statistics
from probe_training_convergence import selected_examples
from probe_lamp_visibility import Renderer
from digit_evidence import evidence_masks,projected_bounds

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--geometry',required=True);ap.add_argument('--asset',required=True);ap.add_argument('--library',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    out=Path(a.output);out.mkdir(exist_ok=False);cv2.setNumThreads(1);render=Renderer(a.library,a.asset,include_digit=True);template=json.loads(Path(a.geometry).read_text())['object_points_armor_m']
    records=[];start=time.monotonic()
    for r,o in selected_examples(a.manifest,24):
        im=rgb(r['image_path']);pose=solve_physical(o['corners'],o['K'],template);lo,hi=projected_bounds(render,pose['rotation_vector'],pose['xyz_m'],o['K'],im.shape)
        patch=im[lo[1]:hi[1],lo[0]:hi[0]];pred=render(pose['rotation_vector'],pose['xyz_m'],o['K'],lo,patch.shape[:2]);obs=evidence_masks(patch,pred);assert obs
        metrics={}
        for name,index in [('blue',1),('white',2)]:
            seen=obs[name];roi=obs[name+'_roi'];got=(pred==index)&roi
            metrics[name]={'iou':float((seen&got).sum()/max(1,(seen|got).sum())),'predicted_pixels':int(got.sum()),'observed_pixels':int(seen.sum())}
        records.append({'session':r['session'],'frame_seq':r['frame_seq'],'slot':o['slot'],'range_m':o['range_m'],'facing_cos':o['facing_cos'],'light':r['light'],'metrics':metrics,'threshold':obs['threshold']})
    result={'complete':True,'seconds':time.monotonic()-start,'scope':'TRAIN-only oracle-pose diagnostic; no online performance or bound claimed','objects':len(records),'summary':{k:statistics([r['metrics'][k]['iou'] for r in records]) for k in ['blue','white']},'geometry_parts':render.parts}
    (out/'records.json').write_text(json.dumps(records));(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
