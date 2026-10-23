"""TRAIN-only native integer-crop consistency probe at fixed model weights."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from armor_native.broad import solve_physical, statistics
from armor_native.data import rgb
from armor_native.geometry import crop_native
from armor_native.quad import NativeQuadPrecision

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights',required=True);ap.add_argument('--manifest',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--seconds',type=int,default=180)
    a=ap.parse_args();out=Path(a.output);assert not out.exists();torch.set_num_threads(2)
    ck=torch.load(a.weights,map_location='cpu',weights_only=False)
    net=NativeQuadPrecision(ck['config']['joint'],ck['config'].get('coordinate_mode','legacy'),ck['config'].get('width_multiplier',1),ck['config'].get('refinement_mode','none')).cuda().eval()
    net.load_state_dict(ck['model'])
    groups={3:[],4:[]};seen=set()
    for line in Path(a.manifest).open():
        f=json.loads(line);assert f['split']=='train'
        for o in f['objects']:
            band=o['range_bin'];identity=f['session'],o['slot']
            if band not in groups or not 3<=o['range_m']<=5 or 'training_proposal_corners' not in o or identity in seen:continue
            seen.add(identity);groups[band].append((f,o))
    rng=np.random.default_rng(7070);chosen=[]
    for band,pairs in groups.items():
        chosen.extend(pairs[i] for i in rng.permutation(len(pairs))[:96])
    shifts=np.asarray([[0,0],[-4,0],[4,0],[0,-4],[0,4]],np.float32)
    rows=[];start=time.monotonic()
    for f,o in chosen:
        if time.monotonic()-start>a.seconds:raise TimeoutError('bounded crop probe')
        image=rgb(f['image_path']);anchors=np.asarray(o['training_proposal_corners'],np.float32)[None]+shifts[:,None]
        native=[[crop_native(image,a,128) for a in aa] for aa in anchors]
        patches=np.asarray([[v[0] for v in items] for items in native]).transpose(0,1,4,2,3).copy()
        origins=np.asarray([[v[1] for v in items] for items in native])
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            pred=net(torch.from_numpy(patches).float().cuda()/255,torch.from_numpy(anchors).cuda())
        points=pred['corners'].float().cpu().numpy()+origins
        record={'session':f['session'],'frame_seq':f['frame_seq'],'slot':o['slot'],'light':f['light'],'range_m':o['range_m'],
                'crop_prediction_rms_spread_px':float(np.sqrt(np.mean((points-points.mean(0))**2)))}
        for name,c in [('baseline',points[0]),('five_crop_mean',points.mean(0))]:
            pose=solve_physical(c,o['K'],ck['geometry'])
            item={'corner_rms_px':float(np.sqrt(np.mean((c-o['corners'])**2))),'valid_pnp':pose is not None}
            if pose is not None:
                delta=(np.asarray(pose['xyz_m'])-o['xyz_m'])*1000
                item.update(depth_mm=float(abs(delta[2])),xy_mm=float(np.linalg.norm(delta[:2])))
            record[name]=item
        rows.append(record)
    report={'complete':True,'scope':'192 TRAIN objects, fixed checkpoint; identical image information, five native integer crops; no recall or deployment claim',
            'seconds':time.monotonic()-start,'shifts_px':shifts.tolist(),'GPU_T4_AMP':True,'weights_changed':False,
            'summary':{name:{m:statistics([r[name][m] for r in rows if m in r[name]]) for m in ['corner_rms_px','depth_mm','xy_mm']} for name in ['baseline','five_crop_mean']},
            'crop_prediction_spread_px':statistics([r['crop_prediction_rms_spread_px'] for r in rows]),
            'records':rows,'source_sha256':{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [a.weights,a.manifest,__file__]}}
    with out.open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='records'}))
if __name__=='__main__':main()
