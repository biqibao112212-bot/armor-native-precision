from pathlib import Path
import sys,json,time,math,hashlib,copy,collections
import torch,numpy as np,cv2
from torch.utils.data import DataLoader
import argparse
ap=argparse.ArgumentParser()
ap.add_argument("--manifest",required=True);ap.add_argument("--output",required=True);ap.add_argument("--initial")
ap.add_argument("--steps",type=int,default=8000);ap.add_argument("--seconds",type=int,default=2400)
ap.add_argument("--seed",type=int,default=2026102405);ap.add_argument("--sample-offset",type=int,default=8000);ap.add_argument("--learning-rate",type=float,default=.0003)
a=ap.parse_args()
from armor_native.efficient_proposal import FastProposalNetwork
from train_proposal_observable import Batches
from train_proposal_broad import loss_fn
torch.set_num_threads(3);cv2.setNumThreads(1)
seed=a.seed;torch.manual_seed(seed);np.random.seed(seed)
out=Path(a.output);out.mkdir(exist_ok=False)
manifest=Path(a.manifest);rows=list(map(json.loads,manifest.open()))
assert all(r["split"]=="train" and r["requested_level_target_verified"] for r in rows)
steps=a.steps;budget=a.seconds
model=FastProposalNetwork().cuda()
if a.initial:model.load_state_dict(torch.load(a.initial,map_location="cpu",weights_only=False)["model"])
ema=copy.deepcopy(model).eval();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001);scaler=torch.amp.GradScaler("cuda",init_scale=128)
config={"seed":seed,"steps":steps,"seconds":budget,"source_sha256":hashlib.sha256(Path(__import__("armor_native.efficient_proposal",fromlist=[""]).__file__).read_bytes()).hexdigest(),"manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),"native_rgb_resize":False,"pixel_unshuffle":4,"sampling":"distance-balanced existing Batches","initial":a.initial,"mask_policy":"coarse-confirmed-negative-train.audit.json","normalization":"BN batch-one TRAIN; equivalent frame statistics inference mandatory","selection":"fixed final and EMA on two existing DEV suites; final fine model unchanged","fine_model_sha256":"942ea92547e6f9d9554a8c249c819c8fc5fe0d532b57ac1a9c7e1348f7a43c54","missing_labels":"existing coarse ignored boxes; positive overrides ignore"}
(out/"config.json").write_text(json.dumps(config,indent=2))
ds=Batches(rows,[],steps,seed,sampling="distance",legacy_fraction=0,gain_mode="legacy",sample_offset=a.sample_offset)
loader=iter(DataLoader(ds,batch_size=1,num_workers=2,pin_memory=True,prefetch_factor=2))
start=time.monotonic();skipped=0;draws=collections.Counter()
def save(step):
 common={"head":"fast_native_proposal","architecture":"pixel-unshuffle-depthwise-v1","updates":step,"optimizer_updates":step-skipped,"config":config}
 torch.save(dict(common,model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),out/f"checkpoint-{step:06d}.pt")
 torch.save(dict(common,model=ema.state_dict(),weight_average="EMA .999 over effective optimizer steps"),out/f"ema-{step:06d}.pt")
with (out/"train.jsonl").open("x") as log:
 for step in range(steps):
  if time.monotonic()-start>budget:save(step);raise TimeoutError("bounded 2400s lightweight proposal training")
  model.train();draw=next(loader);draws.update(map(str,draw["sample_range"].tolist()));b=[v.cuda(non_blocking=True) for v in draw["tensors"]]
  lr=a.learning_rate*min(1.,(step+1)/200)*(.1+.9*.5*(1+math.cos(math.pi*step/(steps-1))));opt.param_groups[0]["lr"]=lr;opt.zero_grad(set_to_none=True)
  with torch.autocast("cuda",dtype=torch.float16):loss,terms=loss_fn(model(b[0]),*b[1:])
  if not torch.isfinite(loss):raise RuntimeError("nonfinite training loss")
  scaler.scale(loss).backward();scaler.unscale_(opt);gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5);old=scaler.get_scale();scaler.step(opt);scaler.update();skip=scaler.get_scale()<old;skipped+=int(skip)
  if not skip:
   with torch.no_grad():
    for k,v in model.state_dict().items():
     ev=ema.state_dict()[k]
     if v.is_floating_point():ev.lerp_(v,.001)
     else:ev.copy_(v)
  done=step+1;row={"step":done,"loss":float(loss.detach()),"skipped":skipped,"grad_norm":float(gn) if torch.isfinite(gn) else None,"elapsed_s":time.monotonic()-start,**terms};log.write(json.dumps(row)+"\n")
  if done%100==0:log.flush()
  if done%500==0:print(json.dumps(row),flush=True)
  if done%2000==0:save(done)
result={"complete":True,"steps":steps,"effective_updates":steps-skipped,"seconds":time.monotonic()-start,"draws":dict(draws),"peak_gpu_bytes":torch.cuda.max_memory_allocated()}
(out/"result.json").write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
