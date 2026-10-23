"""TRAIN-only bounded direct image alignment probe for endpoint observability.

Uses a native near-frontal TRAIN template and native target pixels. Template
sampling is continuous bilinear; target RGB is never resized. Labels are used
only to define the template and to score saved network/image outputs.
"""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np,torch
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from armor_native.data import rgb
from armor_native.geometry import crop_native
from armor_native.quad import NativeQuadPrecision
from armor_native.broad import solve_physical,statistics

def feature(im):
    x=im.astype(float)/255
    return np.stack([x.mean(-1),np.maximum(x[...,1],x[...,2])-x[...,0]],-1)

def refine(target,guess,template,template_corners):
    truthless=np.asarray(guess,float)
    lo=np.maximum(np.floor(truthless.min(0)-4).astype(int),0);hi=np.minimum(np.ceil(truthless.max(0)+5).astype(int),[target.shape[1],target.shape[0]])
    yy,xx=np.mgrid[lo[1]:hi[1],lo[0]:hi[0]];grid=np.c_[xx.ravel(),yy.ravel(),np.ones(xx.size)]
    inv=cv2.getPerspectiveTransform(truthless.astype(np.float32),np.array([[0,1],[0,0],[1,0],[1,1]],np.float32));uv=grid@inv.T;uv=uv[:,:2]/uv[:,2:3]
    valid=(uv[:,0]>-.025)&(uv[:,0]<1.025)&(uv[:,1]>-.05)&(uv[:,1]<1.05)
    grid=grid[valid]
    if len(grid)<40:return truthless,{'reason':'too few pixels'}
    observed=feature(target)[grid[:,1].astype(int),grid[:,0].astype(int)]
    temp=feature(template)
    def residual(delta,return_cost=False):
        q=truthless+delta.reshape(4,2)
        H=cv2.getPerspectiveTransform(q.astype(np.float32),np.asarray(template_corners,np.float32));p=grid@H.T;p=p[:,:2]/p[:,2:3]
        predicted=np.stack([map_coordinates(temp[:,:,k],[p[:,1],p[:,0]],order=1,mode='nearest',prefilter=False) for k in range(2)],1)
        errors=[]
        for k in range(2):
            x=predicted[:,k];y=observed[:,k];xc=x-x.mean();yc=y-y.mean();s=np.std(y)
            if s<.015:continue
            slope=np.clip(np.dot(xc,yc)/(np.dot(xc,xc)+1e-8),0,4)
            errors.append((yc-slope*xc)/max(s,.03))
        if not errors:return np.zeros(len(grid)*2+8)
        residual=np.concatenate(errors)/np.sqrt(len(grid));return np.r_[residual,.001*delta]
    # Explicit finite differences avoid default steps smaller than float32
    # homography solve precision; no gradient through target labels.
    def jac(x):
        eps=.01;eye=np.eye(8)*eps
        return np.stack([(residual(x+d)-residual(x-d))/(2*eps) for d in eye],1)
    start=np.zeros(8);base=np.linalg.norm(residual(start));fit=least_squares(residual,start,jac=jac,bounds=(-2.,2.),max_nfev=20,ftol=1e-5,xtol=1e-5,gtol=1e-5)
    return truthless+fit.x.reshape(4,2),{'initial_cost':float(base),'final_cost':float(np.linalg.norm(residual(fit.x))),'nfev':fit.nfev,'maximum_change_px':float(abs(fit.x).max())}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--weights',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();out=Path(a.output);out.mkdir(exist_ok=False);torch.set_num_threads(1);cv2.setNumThreads(1)
    rows=[json.loads(l) for l in Path(a.data).open()];assert all(r['split']=='train' for r in rows)
    refs=[(r,o) for r in rows for o in r['objects'] if 1<o['range_m']<2 and o['facing_cos']>.97 and r['light']=='normal']
    assert refs;r0,g0=min(refs,key=lambda pair:abs(pair[1]['height_difference_m']-.1));ref=rgb(r0['image_path']);corners0=np.array(g0['corners']);lo=np.maximum(np.floor(corners0.min(0)-8).astype(int),0);hi=np.minimum(np.ceil(corners0.max(0)+9).astype(int),[1440,1080]);template=ref[lo[1]:hi[1],lo[0]:hi[0]];tc=corners0-lo
    ck=torch.load(a.weights,map_location='cpu',weights_only=False);model=NativeQuadPrecision(ck['config']['joint']).eval();model.load_state_dict(ck['model']);groups={l:[] for l in ['dark','normal','bright']};seen=set()
    for r in rows:
        for o in r['objects']:
            if 3<=o['range_m']<=5 and 'training_proposal_corners' in o and (r['session'],o['slot']) not in seen:
                groups[r['light']].append((r,o));seen.add((r['session'],o['slot']))
    rng=np.random.default_rng(914);selected=[]
    for l,pairs in groups.items():selected.extend(pairs[i] for i in rng.permutation(len(pairs))[:16])
    plan={'question':'Can a fixed TRAIN appearance template recover the endpoint length signal from unchanged native pixels?', 'reference':{'session':r0['session'],'frame_seq':r0['frame_seq'],'slot':g0['slot']},'objects':len(selected),'time_cap_s':240,'control':'fixed neural corners','candidate':'image photometric alignment, no target truth at refinement','limits':'one template is a conditional mechanism probe, not an architecture bound; TRAIN only, not adoption evidence'}
    (out/'plan.json').write_text(json.dumps(plan,indent=2));records=[];started=time.monotonic()
    with torch.inference_mode():
        for r,o in selected:
            if time.monotonic()-started>240:raise TimeoutError('template probe budget')
            im=rgb(r['image_path']);anchors=np.array(o['training_proposal_corners'],np.float32);crops=[crop_native(im,p,128) for p in anchors];patches=np.stack([c[0] for c in crops]).transpose(0,3,1,2).copy();origins=np.array([c[1] for c in crops]);p=model(torch.from_numpy(patches[None]).float()/255,torch.from_numpy(anchors[None]));base=p['corners'][0].numpy()+origins
            refined,meta=refine(im,base,template,tc);record={'session':r['session'],'frame_seq':r['frame_seq'],'slot':o['slot'],'range_m':o['range_m'],'light':r['light'],'alignment':meta}
            for name,c in [('base',base),('aligned',refined)]:
                pose=solve_physical(c,o['K'],ck['geometry']);err=(np.array(pose['xyz_m'])-o['xyz_m'])*1000
                record[name]={'corner_rms_px':float(np.sqrt(np.mean((c-o['corners'])**2))),'depth_mm':float(abs(err[2])),'xy_mm':float(np.linalg.norm(err[:2]))}
            records.append(record)
    result={'complete':True,'seconds':time.monotonic()-started,'objects':len(records),'scope':'TRAIN-only fixed-template mechanism probe','methods':{m:{k:statistics([r[m][k] for r in records]) for k in ['corner_rms_px','depth_mm','xy_mm']} for m in ['base','aligned']}}
    (out/'records.json').write_text(json.dumps(records));(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
