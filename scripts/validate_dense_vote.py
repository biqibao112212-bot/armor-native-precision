"""Real TRAIN engineering checks for coherent dense endpoint supervision."""
import argparse,importlib.util,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from train_native_quad import Samples,read
from armor_native.quad import NativeQuadPrecision,QuadLocalizer
from armor_native.model import precision_loss
from armor_native.dense_vote import dense_vote_loss
from export_quad import DecodedQuad,sha256
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();n=Path(a.root);out=n/'dense-vote-validation';out.mkdir(exist_ok=False);torch.set_num_threads(2);torch.manual_seed(2026102303);start=time.monotonic()
 initial=n/'capacity-extra-control/checkpoint-004000.pt';ck=torch.load(initial,map_location='cpu',weights_only=False)
 spec=importlib.util.spec_from_file_location('armor_native._dense_reference',n/'capacity-source/src/armor_native/quad.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 old=mod.NativeQuadPrecision(True).cuda().eval();old.load_state_dict(ck['model'])
 m=NativeQuadPrecision(True,'dense_votes').cuda().eval();missing,unexpected=m.load_state_dict(ck['model'],strict=False);assert set(missing)=={'dense_head.weight','dense_head.bias'} and not unexpected
 ds=Samples(read(n/'retrain-anchors-cyan/train.jsonl'),2,ck['geometry'],2026102203,'image_coarse',20000,'distance');batches=[{k:v.cuda() for k,v in ds[i].items()} for i in range(2)]
 differences=[]
 for b in batches:
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
   legacy=old(b['image'],b['anchors']);new=m(b['image'],b['anchors'])
  assert torch.equal(legacy['corners'],new['corners']);assert torch.equal(legacy['heatmap'],new['heatmap'])
  differences.append(float((legacy['corners']-new['corners']).abs().max()))
 opt=torch.optim.AdamW(m.parameters(),lr=1e-4);scaler=torch.amp.GradScaler('cuda',init_scale=128);rows=[]
 for b in batches:
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.float16):pred=m(b['image'],b['anchors'])
  base,_=precision_loss({'heatmap':pred['heatmap'],'offset':pred['offset']},b)
  dense=dense_vote_loss(pred['vote_vectors'],b['corners']);e=pred['corners']-b['corners'];coord=F.smooth_l1_loss(e/.236,torch.zeros_like(e))
  good=e.detach().abs().amax((1,2))<4;d=torch.bmm(b['jacobian'],e.reshape(-1,8,1)).squeeze(-1)
  geo=(F.smooth_l1_loss(d/torch.tensor([.0106,.0106,.0743],device='cuda'),torch.zeros_like(d),reduction='none').mean(1)*good).sum()/good.sum().clamp_min(1)
  shape=F.smooth_l1_loss((e-e.mean(1,keepdim=True))/.236,torch.zeros_like(e));loss=base+dense+.1*coord+.2*(geo+.25*shape)
  assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt)
  headnorm=float(m.dense_head.weight.grad.norm());assert headnorm>0
  gn=torch.nn.utils.clip_grad_norm_(m.parameters(),5);assert torch.isfinite(gn)
  scale=scaler.get_scale();scaler.step(opt);scaler.update();assert scaler.get_scale()==scale
  rows.append({'loss':float(loss.detach()),'dense_loss':float(dense.detach()),'head_grad_norm':headnorm,'total_grad_norm':float(gn)})
 path=out/'smoke-only.pt';cfg={**ck['config'],'coordinate_mode':'dense_votes','refinement_mode':'none','dense_vote_weight':1.}
 torch.save({**ck,'config':cfg,'model':m.state_dict(),'validation_only':True},path);loaded=QuadLocalizer(path).model;b=batches[0]
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
  ref=m(b['image'],b['anchors']);got=DecodedQuad(loaded)(b['image'],b['anchors'])
 assert torch.equal(ref['corners'],got[0])
 result={'complete':True,'scope':'8 real TRAIN objects; engineering initialization, gradients, AMP and load only','initial_corner_max_delta_px':max(differences),'finite_effective_updates':2,'rows':rows,'loader_decoded_wrapper_exact':True,'source_initial_sha256':sha256(initial),'seconds':time.monotonic()-start}
 (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
