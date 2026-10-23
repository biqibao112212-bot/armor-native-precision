"""Derive native #3 contour supervision from TRAIN labels and fixed asset."""
import argparse,json,time,hashlib,collections
from pathlib import Path
import numpy as np,cv2
from armor_native.broad import solve_physical,NATIVE_BUCKETS,statistics
from armor_native.data import rgb
from armor_native.glyph import translation_sensitivity
from probe_lamp_visibility import Renderer
from digit_evidence import evidence_masks,projected_bounds

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--geometry',required=True);ap.add_argument('--asset',required=True);ap.add_argument('--library',required=True);ap.add_argument('--output',required=True);ap.add_argument('--seconds',type=int,default=900);a=ap.parse_args()
    root=Path(a.data);out=Path(a.output);out.mkdir(exist_ok=False);cv2.setNumThreads(1);geo=json.loads(Path(a.geometry).read_text());template=np.asarray(geo['object_points'],float);physical=template[:4]
    renderer=Renderer(a.library,a.asset,include_digit=True);start=time.monotonic();counts=collections.Counter();sizes=collections.Counter();visibilities=[];residuals=[];max_jac=0.
    with (out/'train.jsonl').open('x') as writer:
        for i,r in enumerate(map(json.loads,(root/'train.jsonl').open())):
            assert r['split']=='train' and r['requested_level_target_verified']
            if time.monotonic()-start>a.seconds:raise TimeoutError('glyph label preparation budget')
            im=rgb(r['image_path']) if r['objects'] else None;objects=[]
            for o in r['objects']:
                K=np.asarray(o['K'],float);pose=solve_physical(o['corners'],K,physical);assert pose
                rv=np.asarray(pose['rotation_vector']);tv=np.asarray(pose['xyz_m']);points=cv2.projectPoints(template,rv,tv,K,np.zeros(5))[0].reshape(-1,2)
                residuals.append(float(np.sqrt(np.mean((points[:4]-o['corners'])**2))));points[:4]=o['corners']
                lo,hi=projected_bounds(renderer,rv,tv,K,im.shape);patch=im[lo[1]:hi[1],lo[0]:hi[0]]
                visible=np.ones(len(points),bool);visible[4:]=False
                if patch.size:
                    rendered=renderer(rv,tv,K,lo,patch.shape[:2]);obs=evidence_masks(patch,rendered)
                    if obs:
                        for j,p in enumerate(points[4:],4):
                            q=np.rint(p-lo).astype(int);x0=max(0,q[0]-2);x1=min(patch.shape[1],q[0]+3);y0=max(0,q[1]-2);y1=min(patch.shape[0],q[1]+3)
                            visible[j]=bool(0<=p[0]<im.shape[1] and 0<=p[1]<im.shape[0] and x1>x0 and y1>y0 and obs['white'][y0:y1,x0:x1].any())
                extent=float(np.max(np.ptp(points,axis=0)));size=next((s for s in NATIVE_BUCKETS if s>=extent+48),None);assert size is not None
                J=translation_sensitivity(template,rv,tv,K,r['camera_quaternion_wxyz'],visible);assert np.isfinite(J).all()
                max_jac=max(max_jac,float(np.abs(J).max()));visibilities.append(int(visible[4:].sum()));sizes[str(size)]+=1;counts[str(o['range_bin'])]+=1
                objects.append(dict(o,glyph_points_image=points.tolist(),glyph_visible=visible.tolist(),glyph_jacobian=J.tolist(),native_size=size))
            writer.write(json.dumps(dict(r,objects=objects))+'\n')
            if (i+1)%1024==0:writer.flush();print(json.dumps({'frames':i+1,'objects':sum(counts.values()),'elapsed_s':time.monotonic()-start}),flush=True)
    (out/'glyph-geometry.json').write_text(json.dumps(geo,indent=2))
    result={'complete':True,'seconds':time.monotonic()-start,'objects':sum(counts.values()),'by_range':dict(counts),'native_sizes':dict(sizes),'visible_numeral_points':statistics(visibilities),'physical_projection_residual_px':statistics(residuals),'maximum_pose_jacobian':max_jac,'manifest_sha256':hashlib.sha256((root/'train.jsonl').read_bytes()).hexdigest(),'scope':'TRAIN-derived additional visible geometric supervision; image files referenced in place, no RGB scaling, no new images copied','eligible_population_changed':False}
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
