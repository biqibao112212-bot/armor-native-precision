"""Fixed TRAIN examples for convergence diagnosis; not a deployment score.

Selection uses TRAIN metadata only. Inference receives RGB, intrinsics and saved
image-derived proposals. No ground-truth corner coordinates are inference inputs.
"""
import argparse,json,time,hashlib
from pathlib import Path
import cv2,numpy as np,torch
from armor_native.broad import load_localizer,solve_physical,statistics
from armor_native.data import rgb

def selected_examples(manifest,per_band=96):
    groups={3:[],4:[]};seen=set()
    for r in map(json.loads,Path(manifest).open()):
        assert r['split']=='train'
        for o in r['objects']:
            key=(r['session'],o['slot'])
            if o['range_bin'] not in groups or not 3<=o['range_m']<=5 or 'training_proposal_corners' not in o or key in seen:continue
            groups[o['range_bin']].append((r,o));seen.add(key)
    rng=np.random.default_rng(7070)
    return [v[int(i)] for v in groups.values() for i in rng.permutation(len(v))[:per_band]]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--models',nargs='+',required=True);ap.add_argument('--names',nargs='+',required=True);ap.add_argument('--output',required=True);ap.add_argument('--seconds',type=int,default=600);ap.add_argument('--device',default='cpu');ap.add_argument('--per-band',type=int,default=96);ap.add_argument('--decoders',nargs='+');a=ap.parse_args()
    decoders=a.decoders or [None]*len(a.models);assert len(decoders)==len(a.models)
    assert len(a.models)==len(a.names) and len(set(a.names))==len(a.names)
    out=Path(a.output);out.mkdir(exist_ok=False);torch.set_num_threads(1);cv2.setNumThreads(1)
    pairs=selected_examples(a.manifest,a.per_band);assert pairs
    plan={'scope':'TRAIN convergence diagnostic, fixed associated image proposal, no end-to-end recall claim','objects':len(pairs),'models':{n:{'path':p,'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()} for n,p in zip(a.names,a.models)},'manifest_sha256':hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),'device':a.device,'native_pixels':True,'decoder_overrides':dict(zip(a.names,decoders))}
    (out/'plan.json').write_text(json.dumps(plan,indent=2));start=time.monotonic();records=[]
    with torch.inference_mode():
        for name,path,decoder in zip(a.names,a.models,decoders):
            net=load_localizer(path,device=a.device)
            if decoder is not None:
                if hasattr(net,'mode'):
                    assert decoder in ['direct','lamp_level','glyph_pnp','glyph_level'];net.mode=decoder
                else:
                    assert hasattr(net,'decoder') and decoder in ['corners','dense','blend'];net.decoder=decoder
            for r,o in pairs:
                if time.monotonic()-start>a.seconds:raise TimeoutError('TRAIN convergence probe budget')
                im=rgb(r['image_path']);pc=np.asarray(o['training_proposal_corners']);center=(pc.min(0)+pc.max(0))/2
                proposal={'score':1.,'center':center.tolist(),'coarse_corners':pc.tolist(),'extent_px':float(np.max(np.ptp(pc,axis=0)))}
                got=net(im,[proposal],np.asarray(o['K']),r['camera_quaternion_wxyz']) if getattr(net,'supports_camera_attitude',False) else net(im,[proposal],np.asarray(o['K']));assert len(got)==1
                pred=got[0];corn=np.asarray(pred['corners']);truth=np.asarray(o['corners']);err=corn-truth
                pose=solve_physical(corn,o['K'],net.template);delta=None if pose is None else (np.asarray(pose['xyz_m'])-o['xyz_m'])*1000
                lengths=[]
                for bottom,top in [(0,1),(3,2)]:
                    u=truth[top]-truth[bottom];u/=np.linalg.norm(u);lengths.append(float((err[top]-err[bottom])@u))
                records.append({'method':name,'session':r['session'],'frame_seq':r['frame_seq'],'slot':o['slot'],'range_m':o['range_m'],'light':r['light'],'facing_cos':o['facing_cos'],'accepted':pred['xyz_m'] is not None,'decoder_source':pred.get('decoder_source','quad'),'valid_pose':pose is not None,'corner_rms_px':float(np.sqrt(np.mean(err**2))),'depth_mm':None if delta is None else abs(float(delta[2])),'xy_mm':None if delta is None else float(np.linalg.norm(delta[:2])),'length_error_px':lengths,'corner_error_px':err.tolist()})
            del net
    report={'complete':True,'seconds':time.monotonic()-start,'scope':plan['scope'],'methods':{}}
    for name in a.names:
        rows=[r for r in records if r['method']==name]
        report['methods'][name]={'objects':len(rows),'accepted':sum(r['accepted'] for r in rows),'valid_pose':sum(r['valid_pose'] for r in rows),'dense_used':sum(r['decoder_source'] in ['dense','blend'] for r in rows),**{k:statistics([r[k] for r in rows if r[k] is not None]) for k in ['corner_rms_px','depth_mm','xy_mm']}}
    (out/'records.json').write_text(json.dumps(records));(out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
