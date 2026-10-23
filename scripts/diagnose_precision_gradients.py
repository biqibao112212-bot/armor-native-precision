"""TRAIN-only gradient alignment probe for the retained precision checkpoint."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from train_native_quad import Samples, read
from armor_native.model import precision_loss
from armor_native.quad import NativeQuadPrecision

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights',required=True)
    ap.add_argument('--data',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--seconds',type=int,default=180)
    a=ap.parse_args();output=Path(a.output);assert not output.exists()
    torch.set_num_threads(2);torch.manual_seed(2026102203)
    ck=torch.load(a.weights,map_location='cpu',weights_only=False)
    model=NativeQuadPrecision(ck['config']['joint'],ck['config'].get('coordinate_mode','legacy'),ck['config'].get('width_multiplier',1),ck['config'].get('refinement_mode','none')).eval()
    model.load_state_dict(ck['model'])
    rows=read(Path(a.data)/'train.jsonl')
    rows=[dict(r,objects=[o for o in r['objects'] if 3<=o['range_m']<=5]) for r in rows]
    rows=[r for r in rows if r['objects']]
    ds=Samples(rows,8,ck['geometry'],2026102203,'image_coarse',16000,'distance')
    params=[p for p in model.parameters() if p.requires_grad]
    records=[];start=time.monotonic()
    for i in range(8):
        if time.monotonic()-start>a.seconds:raise TimeoutError('TRAIN CPU probe')
        b=ds[i];pred=model(b['image'],b['anchors'])
        base,_=precision_loss({'heatmap':pred['heatmap'],'offset':pred['offset']},b)
        e=pred['corners']-b['corners']
        coord=F.smooth_l1_loss(e/.236,torch.zeros_like(e))
        good=e.detach().abs().amax((1,2))<4
        d=torch.bmm(b['jacobian'],e.reshape(-1,8,1)).squeeze(-1)
        geo=(F.smooth_l1_loss(d/torch.tensor([.0106,.0106,.0743]),torch.zeros_like(d),reduction='none').mean(1)*good).sum()/good.sum().clamp_min(1)
        shape=F.smooth_l1_loss((e-e.mean(1,keepdim=True))/.236,torch.zeros_like(e))
        losses={'detection':base,'coordinate':.1*coord,'geometry_shape':.2*(geo+.25*shape)}
        grads={}
        for k,loss in losses.items():
            gg=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
            grads[k]=torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for p,g in zip(params,gg)]).detach()
        accuracy=grads['coordinate']+grads['geometry_shape']
        total=grads['detection']+accuracy
        def cosine(a,b):
            a,b=a.double(),b.double()
            return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-12))
        records.append({'batch':i,'objects':len(e),'losses':{k:float(v.detach()) for k,v in losses.items()},
          'gradient_norms':{k:float(v.norm()) for k,v in grads.items()},
          'detection_accuracy_cosine':cosine(grads['detection'],accuracy),
          'total_accuracy_cosine':cosine(total,accuracy),
          'total_coordinate_cosine':cosine(total,grads['coordinate']),
          'total_geometry_cosine':cosine(total,grads['geometry_shape'])})
        print(json.dumps(records[-1]),flush=True)
    report={'complete':True,'scope':'32 TRAIN objects, 8 actual augmented batches; CPU FP32; no weights changed',
      'seconds':time.monotonic()-start,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'weights_sha256':hashlib.sha256(Path(a.weights).read_bytes()).hexdigest(),
      'manifest_sha256':hashlib.sha256((Path(a.data)/'train.jsonl').read_bytes()).hexdigest(),
      'records':records,'total_gradient_opposes_accuracy_batches':sum(r['total_accuracy_cosine']<0 for r in records),
      'interpretation_limit':'Local first-order probe; no claim about Adam updates, global convergence or deployment precision.'}
    with output.open('x') as f:json.dump(report,f,indent=2)
if __name__=='__main__':main()
