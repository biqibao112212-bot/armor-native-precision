"""Validated spatial model: broad roll0 data + old TRAIN rehearsal, native pixels."""
import argparse,json,math,time,hashlib,collections
from pathlib import Path
import numpy as np,torch,cv2
from torch.utils.data import Dataset,DataLoader
from armor_native.broad import NativeBucketPrecision
from armor_native.model import precision_loss
from train_broad import pool,read,sample,diagnostic,dump

class Batches(Dataset):
 def __init__(self,broad,legacy,steps,seed):
  self.steps=steps;self.seed=seed;self.pools=[pool(broad),pool(legacy)];self.keys=[sorted(p) for p in self.pools]
 def __len__(self):return self.steps
 def __getitem__(self,index):
  rng=np.random.default_rng(self.seed+index);domain=int(rng.random()<.35);p=self.pools[domain];keys=self.keys[domain];key=keys[int(rng.integers(len(keys)))];size=key[-1];ss=sorted(p[key]);n=max(1,min(16,262144//(size*size)));samples=[]
  for _ in range(n):
   session=ss[int(rng.integers(len(ss)))];pairs=p[key][session];r,o=pairs[int(rng.integers(len(pairs)))];samples.append(sample(r,o,rng,index<int(self.steps*.75) and domain==0))
  return {**{k:torch.stack([s[k] for s in samples]) for k in ['image','heatmap','offset','point_mask']},'native_size':size,'domain':'legacy' if domain else 'broad','range_bin':key[0],'batch_size':n}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--legacy',required=True);ap.add_argument('--initial',required=True);ap.add_argument('--output',required=True);ap.add_argument('--steps',type=int,default=10000);ap.add_argument('--seconds',type=int,default=6000);ap.add_argument('--workers',type=int,default=2);a=ap.parse_args();out=Path(a.output);out.mkdir(exist_ok=False,parents=True);root=Path(a.data);legacy=Path(a.legacy);seed=20261113;torch.manual_seed(seed);torch.set_num_threads(4);cv2.setNumThreads(1)
 train=read(root/'train.jsonl');dev=read(root/'dev.jsonl');oldtrain=read(legacy/'train.jsonl');olddev=read(legacy/'dev.jsonl');assert not {r['session'] for r in train}&{r['session'] for r in dev};assert all(r['split']=='train' for r in train+oldtrain);assert all(r['requested_roll_zero_verified'] for r in train)
 oldtrain=[dict(r,objects=[o for o in r['objects'] if 3<=o['xyz_m'][2]<=5]) for r in oldtrain];olddev=[dict(r,objects=[o for o in r['objects'] if 3<=o['xyz_m'][2]<=5]) for r in olddev]
 template=json.loads((root/'geometry.json').read_text())['object_points_armor_m'];model=NativeBucketPrecision().cuda();model.load_state_dict(torch.load(a.initial,map_location='cpu',weights_only=False)['model']);opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.0001);scaler=torch.amp.GradScaler('cuda',init_scale=128)
 cfg={**vars(a),'seed':seed,'architecture':'unchanged original spatial head with native-size coordinate grids','RGB_resized':False,'template':template,'legacy_update_fraction':.35,'broad_update_fraction':.65,'selection':'fixed final; intermediate DEV only, old report benchmark and sealed data never used for fitting','training_frames':len(train),'legacy_training_frames':len(oldtrain),'training_configs':len(set(r['session'] for r in train)),'schedule':'first75% cosine5e-4 to3e-5;last25% clean precision consolidation1e-4 to1e-5; no legacy augmentation','source_sha256':{str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in [Path(__file__),Path(__file__).with_name('train_broad.py')]},'manifest_sha256':{str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in [root/'train.jsonl',root/'dev.jsonl',legacy/'train.jsonl',legacy/'dev.jsonl']},'initial_sha256':hashlib.sha256(Path(a.initial).read_bytes()).hexdigest()};dump(out/'config.json',cfg)
 ds=Batches(train,oldtrain,a.steps,seed);loader=DataLoader(ds,batch_size=None,num_workers=a.workers,pin_memory=True,**({'prefetch_factor':2} if a.workers else {}));started=time.monotonic();skipped=0;draws=collections.Counter();completed=0
 def save():
  path=out/f'checkpoint-{completed:06d}.pt';torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'updates':completed,'optimizer_updates':completed-skipped,'head':'native_bucket_physical','geometry':template,'config':cfg,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},path)
 def evaluate():
  dump(out/f'dev-{completed:06d}.json',{'broad':diagnostic(model,dev,template,'cuda',n=256),'legacy':diagnostic(model,olddev,template,'cuda',n=128)});model.train()
 evaluate()
 with (out/'train.jsonl').open('x') as log:
  for step,b in enumerate(loader):
   if time.monotonic()-started>a.seconds:save();raise TimeoutError('fixed training time budget reached')
   pivot=int(a.steps*.75)
   if step<pivot:lr=.00003+.5*(.0005-.00003)*(1+math.cos(math.pi*step/max(1,pivot-1)))
   else:lr=.00001+.5*(.0001-.00001)*(1+math.cos(math.pi*(step-pivot)/max(1,a.steps-pivot-1)))
   for g in opt.param_groups:g['lr']=lr
   batch={k:b[k].cuda(non_blocking=True) for k in ['image','heatmap','offset','point_mask']};opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16):pred=model(batch['image']);loss,terms=precision_loss(pred,batch)
   if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
   scaler.scale(loss).backward();scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);old=scaler.get_scale();scaler.step(opt);scaler.update();skipped+=int(scaler.get_scale()<old);completed=step+1;draws[b['domain']+'/'+str(b['range_bin'])]+=b['batch_size'];row={'update':completed,'loss':float(loss.detach()),'grad_norm':float(norm) if torch.isfinite(norm) else None,'lr':lr,'domain':b['domain'],'native_size':b['native_size'],'batch_size':b['batch_size'],'skipped':skipped,'elapsed_s':time.monotonic()-started,**terms};log.write(json.dumps(row)+'\n');log.flush()
   if completed%1000==0 or completed==a.steps:save();evaluate();print(json.dumps(row),flush=True)
 dump(out/'result.json',{'complete':True,'updates':completed,'optimizer_updates':completed-skipped,'seconds':time.monotonic()-started,'draws':dict(draws),'sealed_used':False})
if __name__=='__main__':main()
