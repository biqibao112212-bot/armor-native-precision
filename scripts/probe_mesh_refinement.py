"""Bounded image-only mesh-silhouette pose refinement on TRAIN.

Fixed public armor mesh and camera calibration are permitted inputs; target
pose/corners/slot/range are never passed to refinement. Labels only select and
score TRAIN examples. This is a mechanism probe, not a deployed model.
"""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np,torch
from scipy.spatial.transform import Rotation
from scipy.optimize import differential_evolution
from armor_native.broad import load_localizer,solve_physical,statistics
from armor_native.level_pose import fit_level_pose,LINK_TO_CV
from armor_native.data import rgb
from probe_lamp_visibility import Renderer
from probe_training_convergence import selected_examples
from digit_evidence import evidence_masks,projected_bounds

def image_refine(im,corners,K,template,camera_q,renderer,maxiter=40,popsize=8,seed=7):
    started=time.monotonic();K=np.asarray(K,float);template=np.asarray(template,float);corners=np.asarray(corners,float)
    level=fit_level_pose(corners,K,template,camera_q)
    if level is None:return None
    q=np.asarray(camera_q,float);rc=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();wc=LINK_TO_CV@rc.T;mount=Rotation.from_euler('x',-15,degrees=True).as_matrix()
    t0=np.asarray(level['xyz_m']);z0=t0[2];center=(K@t0)[:2]/z0;yaw0=level['yaw_world_rad']
    lo=np.maximum(np.floor(corners.min(0)-9).astype(int),0);hi=np.minimum(np.ceil(corners.max(0)+10).astype(int),[im.shape[1],im.shape[0]])
    use_digit=bool((renderer.material==2).any())
    patch=im[lo[1]:hi[1],lo[0]:hi[0]].astype(float)
    observed=(patch[:,:,2]-patch[:,:,0]>35)&(patch[:,:,2]>=patch[:,:,1]-10)&(patch.max(-1)>=80)
    def pose(x):
        z=z0*np.exp(x[2]*.01);uv=center+x[:2];tv=np.array([(uv[0]-K[0,2])*z/K[0,0],(uv[1]-K[1,2])*z/K[1,1],z])
        rot=wc@Rotation.from_euler('z',yaw0+np.deg2rad(x[3])).as_matrix()@mount
        return cv2.Rodrigues(rot)[0].ravel(),tv
    rv0,tv0=pose(np.zeros(4))
    if use_digit:
        lo,hi=projected_bounds(renderer,rv0,tv0,K,im.shape)
        patch=im[lo[1]:hi[1],lo[0]:hi[0]].astype(float);labels=renderer(rv0,tv0,K,lo,patch.shape[:2])
        evidence=evidence_masks(patch,labels)
        if evidence is None:return None
        observed=evidence['blue'];roi=evidence['blue_roi'];white=evidence['white'];white_roi=evidence['white_roi']
        use_digit=bool(white.sum()>=8)
    else:
        startmask=renderer(rv0,tv0,K,lo,observed.shape)>0
        roi=cv2.dilate(startmask.astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool)
    observed &= roi
    if observed.sum()<8:return None
    scale=max(1,int(observed.sum()));best=[float('inf'),None,None];evaluations=0
    def objective(x):
        nonlocal evaluations
        rv,tv=pose(x);labels=renderer(rv,tv,K,lo,observed.shape);pred=labels==1
        disagreement=int(((pred!=observed)&roi).sum());value=disagreement/scale
        if use_digit:value+=float((((labels==2)!=white)&white_roi).sum())/max(1,int(white.sum()))
        value+=1e-5*float(np.mean(np.square(x)))
        evaluations+=1
        if value<best[0]:best[:]=[value,x.copy(),disagreement]
        return value
    initial=objective(np.zeros(4));bounds=[(-2,2),(-2,2),(-7,7),(-10,10)]
    result=differential_evolution(objective,bounds,maxiter=maxiter,popsize=popsize,tol=0,atol=0,polish=False,seed=seed,workers=1,x0=np.zeros(4),updating='immediate')
    x=best[1];rv,tv=pose(x);points=cv2.projectPoints(template,rv,tv,K,np.zeros(5))[0].reshape(4,2)
    return {'corners':points.tolist(),'xyz_m':tv.tolist(),'rotation_vector':rv.tolist(),'level_initial':level,'initial_loss':initial,'final_loss':best[0],'mismatched_pixels':best[2],'observed_pixels':int(observed.sum()),'digit_used':use_digit,'parameters':x.tolist(),'evaluations':evaluations,'seconds':time.monotonic()-started,'scope':'RGB and fixed public mesh/camera attitude only'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--weights',required=True);ap.add_argument('--asset',required=True);ap.add_argument('--library',required=True);ap.add_argument('--output',required=True);ap.add_argument('--maxiter',type=int,default=40);ap.add_argument('--per-band',type=int,default=24);ap.add_argument('--seconds',type=int,default=300);ap.add_argument('--include-digit',action='store_true');a=ap.parse_args()
    out=Path(a.output);out.mkdir(exist_ok=False);torch.set_num_threads(1);cv2.setNumThreads(1);render=Renderer(a.library,a.asset,include_digit=a.include_digit);net=load_localizer(a.weights,device='cpu')
    pairs=selected_examples(a.manifest,a.per_band);start=time.monotonic();records=[]
    plan={'question':'Can the observed LED silhouette and known fixed geometry resolve endpoint length ambiguity from unchanged native RGB?','objects':len(pairs),'scope':'TRAIN only; no target truth at refinement; no output rejection changes','weights':a.weights,'maxiter':a.maxiter,'seconds':a.seconds,'no_local_backup':True,'simulator_read_only':True,'digit_outline_included':a.include_digit}
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    with torch.inference_mode():
        for i,(r,o) in enumerate(pairs):
            if time.monotonic()-start>a.seconds:raise TimeoutError('mesh probe budget')
            im=rgb(r['image_path']);pc=np.asarray(o['training_proposal_corners']);proposal={'score':1.,'center':((pc.min(0)+pc.max(0))/2).tolist(),'coarse_corners':pc.tolist(),'extent_px':float(np.max(np.ptp(pc,axis=0)))}
            base=net(im,[proposal],np.asarray(o['K']))[0];p=image_refine(im,base['corners'],o['K'],net.template,r['camera_quaternion_wxyz'],render,maxiter=a.maxiter,seed=701+i)
            row={'session':r['session'],'frame_seq':r['frame_seq'],'slot':o['slot'],'range_m':o['range_m'],'facing_cos':o['facing_cos'],'light':r['light'],'refinement':p,'methods':{}}
            variants={'base':base,'level':None if p is None else p['level_initial'],'mesh':p}
            for name,v in variants.items():
                if v is None:continue
                c=np.asarray(v['corners']);pose=solve_physical(c,o['K'],net.template) if name=='base' else v
                if pose is None:continue
                delta=(np.asarray(pose['xyz_m'])-o['xyz_m'])*1000
                row['methods'][name]={'corner_rms_px':float(np.sqrt(np.mean((c-o['corners'])**2))),'depth_mm':abs(float(delta[2])),'xy_mm':float(np.linalg.norm(delta[:2]))}
            records.append(row)
            if (i+1)%8==0:print(json.dumps({'objects':i+1,'elapsed_s':time.monotonic()-start}),flush=True)
    result={'complete':True,'objects':len(records),'seconds':time.monotonic()-start,'scope':plan['scope'],'methods':{name:{k:statistics([r['methods'][name][k] for r in records if name in r['methods']]) for k in ['corner_rms_px','depth_mm','xy_mm']} for name in ['base','level','mesh']},'compute_seconds':statistics([r['refinement']['seconds'] for r in records if r['refinement']])}
    (out/'records.json').write_text(json.dumps(records));(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
