"""Portable form of the validated fixed2000-step ROI recipe."""
import argparse,json,math,random,time
from contextlib import nullcontext
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
from .data import BalancedRois,read_manifest
from .model import RoiPrecision,precision_loss
from .geometry import solve_geometry

def dump(path,value):
 with Path(path).open('x') as out:json.dump(value,out,indent=2,allow_nan=False);out.write('\n')

@torch.no_grad()
def diagnostic(model,ds,device,amp):
 model.eval();errors=[];depth=[]
 for start in range(0,min(64,len(ds)),16):
  samples=[ds[i] for i in range(start,min(start+16,len(ds),64))];image=torch.stack([s['image'] for s in samples]).to(device)
  with torch.autocast('cuda',dtype=torch.float16,enabled=amp):prediction=model(image)
  points=model.points(prediction).cpu().numpy()
  for c,s in zip(points,samples):
   errors.append(float(np.sqrt(np.mean((c-s['corners'].numpy())**2))));pose=solve_geometry(c,s['K'].numpy())
   if pose:depth.append(abs(pose['xyz_m'][2]-float(s['xyz'][2]))*1000)
 model.train()
 return {'scope':'GT-centered jittered ROI diagnostic; not detector-level acceptance','n':len(errors),'corner_rms_p95_px':float(np.quantile(errors,.95)),'valid_pnp':len(depth),'pnp_depth_p95_mm':float(np.quantile(depth,.95)) if depth else None}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--train',nargs='+',required=True);ap.add_argument('--val',nargs='*',default=[]);ap.add_argument('--output',required=True);ap.add_argument('--steps',type=int,default=2000);ap.add_argument('--batch-size',type=int,default=16);ap.add_argument('--head',choices=['spatial','center_regression'],default='spatial');ap.add_argument('--device',default='cuda');ap.add_argument('--seed',type=int,default=20261024);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--eval-interval',type=int,default=200);ap.add_argument('--checkpoint-interval',type=int,default=200);ap.add_argument('--resume');ap.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True);a=ap.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);device=torch.device(a.device);amp=a.amp and device.type=='cuda';torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed);model=RoiPrecision(a.head).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001);scaler=torch.amp.GradScaler('cuda',init_scale=256,enabled=amp);start_step=0;skipped=0;resume=None
 if a.resume:
  resume=torch.load(a.resume,map_location='cpu',weights_only=False);model.load_state_dict(resume['model']);opt.load_state_dict(resume['optimizer']);scaler.load_state_dict(resume['scaler']);start_step=resume['updates'];skipped=start_step-resume.get('optimizer_updates',start_step)
 if not 0<=start_step<a.steps:raise ValueError('steps must exceed resumed update count')
 rows=read_manifest(a.train);val=read_manifest(a.val);train_sessions={r['session'] for r in rows};assert not train_sessions&{r['session'] for r in val},'Train and validation sessions overlap'
 ds=BalancedRois(rows,(a.steps-start_step)*a.batch_size,seed=a.seed,start=start_step*a.batch_size);probe=BalancedRois(rows,64,seed=a.seed);dev=BalancedRois(val,64,seed=20261025) if val else None;loader=DataLoader(ds,batch_size=a.batch_size,num_workers=a.workers,shuffle=False,pin_memory=device.type=='cuda');iterator=iter(loader)
 if resume:
  torch.set_rng_state(resume['cpu_rng']);np.random.set_state(resume['numpy_rng']);random.setstate(resume['python_rng'])
  if device.type=='cuda' and 'cuda_rng' in resume:torch.cuda.set_rng_state_all(resume['cuda_rng'])
 dump(out/'config.json',{**vars(a),'lr':.002,'final_lr':.0002,'weight_decay':.0001,'gradient_clip':5,'jitter_px':8,'input_rgb_resized':False,'sampler':'equal distance/facing bins then equal session then object','train_sessions':sorted(train_sessions),'validation_scope':'oracle ROI diagnostic only'})
 def checkpoint(step):
  payload={'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'updates':step,'optimizer_updates':step-skipped,'head':a.head,'stage':'full','next_draw_index':step*a.batch_size,'cpu_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate()}
  if device.type=='cuda':payload['cuda_rng']=torch.cuda.get_rng_state_all()
  torch.save(payload,out/f'checkpoint-{step:06d}.pt');return payload
 def evaluate(step):dump(out/f'diagnostic-{step:06d}.json',{'train':diagnostic(model,probe,device,amp),'validation':diagnostic(model,dev,device,amp) if dev else None})
 if a.eval_interval:evaluate(start_step)
 start=time.monotonic();model.train()
 with (out/'train.jsonl').open('x') as log:
  for step in range(start_step,a.steps):
   lr=.0002+.5*(.002-.0002)*(1+math.cos(math.pi*step/max(a.steps-1,1)))
   for group in opt.param_groups:group['lr']=lr
   batch={k:v.to(device,non_blocking=True) for k,v in next(iterator).items()};opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16,enabled=amp):prediction=model(batch['image']);loss,terms=precision_loss(prediction,batch)
   if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
   scaler.scale(loss).backward();scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);old_scale=scaler.get_scale();scaler.step(opt);scaler.update();skipped+=int(scaler.get_scale()<old_scale);completed=step+1
   log.write(json.dumps({'update':completed,'loss':float(loss.detach()),'grad_norm':float(norm) if torch.isfinite(norm) else None,'skipped_updates':skipped,'lr':lr,**terms})+'\n');log.flush()
   if completed%a.checkpoint_interval==0 or completed==a.steps:checkpoint(completed)
   if a.eval_interval and completed%a.eval_interval==0:evaluate(completed)
 dump(out/'result.json',{'complete':True,'updates':a.steps,'optimizer_updates':a.steps-skipped,'skipped_updates':skipped,'seconds':time.monotonic()-start,'parameters':sum(p.numel() for p in model.parameters())});print(json.dumps({'complete':True,'output':str(out)}))
if __name__=='__main__':main()
