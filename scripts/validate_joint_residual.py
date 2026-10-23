"""Validate a new joint residual readout on real TRAIN native crops."""
import argparse,importlib.util,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from armor_native.quad import NativeQuadPrecision,QuadLocalizer
from train_native_quad import Samples,read
from export_quad import DecodedQuad,sha256
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--mode',choices=['joint_residual','joint_raw'],default='joint_residual');ap.add_argument('--output',default='joint-residual-validation');a=ap.parse_args();n=Path(a.root);out=n/a.output;out.mkdir(exist_ok=False)
 torch.set_num_threads(2);torch.manual_seed(2026102302);start=time.monotonic()
 initial=n/'capacity-extra-control/checkpoint-004000.pt';ck=torch.load(initial,map_location='cpu',weights_only=False)
 reference_file=n/'capacity-source/src/armor_native/quad.py'
 spec=importlib.util.spec_from_file_location('armor_native._frozen_quad',reference_file);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 reference=mod.NativeQuadPrecision(True,'legacy',1).cuda().eval();reference.load_state_dict(ck['model'])
 base=NativeQuadPrecision(True,'legacy',1).cuda().eval();base.load_state_dict(ck['model'])
 candidate=NativeQuadPrecision(True,'legacy',1,a.mode).cuda().eval()
 missing,unexpected=candidate.load_state_dict(ck['model'],strict=False)
 assert set(missing)=={k for k in candidate.state_dict() if k.startswith('refiner.')} and not unexpected
 for key,p in candidate.named_parameters():p.requires_grad_(key.startswith('refiner.'))
 ds=Samples(read(n/'retrain-anchors-cyan/train.jsonl'),2,ck['geometry'],2026102203,'image_coarse',20000,'distance')
 batches=[{k:v.cuda() for k,v in ds[i].items()} for i in range(2)]
 parity=[]
 for b in batches:
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
   old=reference(b['image'],b['anchors']);legacy=base(b['image'],b['anchors']);new=candidate(b['image'],b['anchors'])
  assert torch.equal(old['corners'],legacy['corners']) and torch.equal(legacy['corners'],new['corners'])
  assert torch.equal(old['heatmap'],new['heatmap'])
  parity.append(float((old['corners']-new['corners']).abs().max()))
 snapshot={k:p.detach().clone() for k,p in candidate.named_parameters() if not p.requires_grad}
 opt=torch.optim.AdamW([p for p in candidate.parameters() if p.requires_grad],lr=1e-4);losses=[];gradnorms=[]
 for b in batches:
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.float16):pred=candidate(b['image'],b['anchors'])
  loss=F.smooth_l1_loss((pred['corners']-b['corners'])/.236,torch.zeros_like(pred['corners']))
  assert torch.isfinite(loss);loss.backward()
  gn=torch.nn.utils.clip_grad_norm_([p for p in candidate.parameters() if p.requires_grad],5)
  assert torch.isfinite(gn) and gn>0;gradnorms.append(float(gn));opt.step();losses.append(float(loss.detach()))
 assert all(torch.equal(p,snapshot[k]) for k,p in candidate.named_parameters() if k in snapshot)
 assert candidate.refiner.joint[-1].weight.abs().max()>0
 cfg={**ck['config'],'refinement_mode':a.mode,'width_multiplier':1}
 path=out/'smoke-only.pt';torch.save({**ck,'model':candidate.state_dict(),'config':cfg,'validation_only':True},path)
 loaded=QuadLocalizer(path).model;b=batches[0]
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
  original=candidate(b['image'],b['anchors']);again=loaded(b['image'],b['anchors']);decoded=DecodedQuad(loaded)(b['image'],b['anchors'])
 assert torch.equal(original['corners'],again['corners']) and torch.equal(again['corners'],decoded[0])
 assert original['corner_correction'].abs().max()<=4
 result={'complete':True,'mode':a.mode,'scope':'8 real augmented TRAIN objects; engineering parity/freeze/finite-gradient/loader only, not efficacy','legacy_frozen_source_max_delta_px':max(parity),'zero_head_initial_max_delta_px':max(parity),'base_parameters_unchanged':True,'gradient_norms':gradnorms,'losses':losses,'loader_decoded_wrapper_exact':True,'initial_sha256':sha256(initial),'source_reference_sha256':sha256(reference_file),'seconds':time.monotonic()-start,'local_backup':False}
 (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
