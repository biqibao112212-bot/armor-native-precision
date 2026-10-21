"""Full original-pixel proposal training; same existing convolutional architecture."""
import argparse,json,time,math,hashlib,collections
from pathlib import Path
import numpy as np,torch,cv2
from torch.utils.data import Dataset,DataLoader
from armor_native.proposals import ProposalNetwork
from train_broad import image,pool,read,dump
from train_proposal_broad import loss_fn

def targets(row,photometric,rng):
 im=image(row['image_path']);patch=np.zeros((1088,1440,3),np.uint8);patch[:1080]=im;h,w=272,360;yy,xx=np.ogrid[:h,:w];hm=np.zeros((1,h,w),np.float32);geo=np.zeros((11,h,w),np.float32);mask=np.zeros((1,h,w),np.float32);geomask=mask.copy();neg=np.ones((1,h,w),np.float32);neg[:,270:]=0
 for box in row.get('coarse_ignored_boxes',[]):
  x0,y0,x1,y1=np.array(box)/4;x0,y0=np.clip(np.floor([x0,y0]).astype(int),[0,0],[w,h]);x1,y1=np.clip(np.ceil([x1,y1]).astype(int)+1,[0,0],[w,h])
  if x1>x0 and y1>y0:neg[:,y0:y1,x0:x1]=0
 for o in row['objects']:
  c=np.array(o['center']);ix,iy=np.floor(c/4).astype(int)
  if not(0<=ix<w and 0<=iy<270):continue
  hm[0]=np.maximum(hm[0],np.exp(-((xx-ix)**2+(yy-iy)**2)/2));mask[0,iy,ix]=1;neg[0,iy,ix]=1;geo[:2,iy,ix]=c/4-[ix,iy];geo[2:10,iy,ix]=((np.array(o['screen_corners'])-c)/32).reshape(-1);geomask[0,iy,ix]=1
 if photometric:patch[:1080]=np.clip(patch[:1080].astype(np.float32)*rng.uniform(.8,1.2)+rng.normal(0,rng.uniform(0,2),patch[:1080].shape),0,255).astype(np.uint8)
 return [torch.from_numpy(patch.transpose(2,0,1).copy()).float()/255,*[torch.from_numpy(x) for x in [hm,geo,mask,geomask,neg]]]

class Batches(Dataset):
 def __init__(self,broad,legacy,steps,seed):self.p=[pool(broad),pool(legacy)];self.keys=[sorted(p) for p in self.p];self.steps=steps;self.seed=seed
 def __len__(self):return self.steps*2
 def __getitem__(self,i):
  rng=np.random.default_rng(self.seed+i);domain=int(rng.random()<.3);p=self.p[domain];keys=self.keys[domain];key=keys[int(rng.integers(len(keys)))];ss=sorted(p[key]);session=ss[int(rng.integers(len(ss)))];pairs=p[key][session];r,o=pairs[int(rng.integers(len(pairs)))];return targets(r,domain==0 and rng.random()<.2,rng)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--legacy',required=True);ap.add_argument('--initial',required=True);ap.add_argument('--config',required=True);ap.add_argument('--output',required=True);ap.add_argument('--steps',type=int,default=3000);ap.add_argument('--seconds',type=int,default=5400);a=ap.parse_args();out=Path(a.output);out.mkdir(exist_ok=False,parents=True);rows=read(a.data);legacy=read(a.legacy);assert all(r['split']=='train' for r in rows+legacy);assert all(r['requested_roll_zero_verified'] for r in rows);seed=20261114;torch.manual_seed(seed);torch.set_num_threads(4);cv2.setNumThreads(1);model=ProposalNetwork(json.loads(Path(a.config).read_text())).cuda();model.load_state_dict(torch.load(a.initial,map_location='cpu',weights_only=False)['model']);model.train();model.backbone.eval();back=list(model.backbone.parameters());ids={id(v) for v in back};head=[v for v in model.parameters() if id(v) not in ids]
 for v in back:v.requires_grad=False
 opt=torch.optim.AdamW([{'params':head,'lr':.0003},{'params':back,'lr':.00003}],weight_decay=.0001);scaler=torch.amp.GradScaler('cuda',init_scale=128);cfg={**vars(a),'seed':seed,'input':'original1440x1080 pixels with8 zero rows at bottom;K unchanged','rgb_resized':False,'microbatch':1,'accumulation':2,'architecture':'same YOLO11n feature backbone+native RGB stride4 stem+existing center/geometry heads;variable feature map only','legacy_draw_fraction':.3,'unknown_regions':'ignored, positive overrides mask','depth_loss':False,'selection':'fixed final;no sealed data','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'initial_sha256':hashlib.sha256(Path(a.initial).read_bytes()).hexdigest()};dump(out/'config.json',cfg);loader=iter(DataLoader(Batches(rows,legacy,a.steps,seed),batch_size=1,num_workers=2,pin_memory=True,prefetch_factor=2));start=time.monotonic();skipped=0
 def save(step):torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'updates':step,'optimizer_updates':step-skipped,'head':'broad_native_proposal','config':cfg,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},out/f'checkpoint-{step:06d}.pt')
 with (out/'train.jsonl').open('x') as log:
  for step in range(a.steps):
   if time.monotonic()-start>a.seconds:save(step);raise TimeoutError('bounded full native proposal training')
   if step==200:
    for v in back:v.requires_grad=True
    model.backbone.train()
   frac=.1+.9*.5*(1+math.cos(math.pi*step/max(1,a.steps-1)));opt.param_groups[0]['lr']=.0003*frac;opt.param_groups[1]['lr']=.00003*frac;opt.zero_grad(set_to_none=True);value=0;agg={}
   for micro in range(2):
    batch=[v.cuda(non_blocking=True) for v in next(loader)]
    with torch.autocast('cuda',dtype=torch.float16):loss,terms=loss_fn(model(batch[0]),*batch[1:])
    if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
    scaler.scale(loss/2).backward();value+=float(loss.detach())/2
    for k,v in terms.items():agg[k]=agg.get(k,0)+v/2
   scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);old=scaler.get_scale();scaler.step(opt);scaler.update();skipped+=int(scaler.get_scale()<old);done=step+1;row={'update':done,'loss':value,'grad_norm':float(norm) if torch.isfinite(norm) else None,'skipped':skipped,'elapsed_s':time.monotonic()-start,**agg};log.write(json.dumps(row)+'\n');log.flush()
   if done%500==0 or done==a.steps:save(done);print(json.dumps(row),flush=True)
 dump(out/'result.json',{'complete':True,'updates':a.steps,'optimizer_updates':a.steps-skipped,'seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated()})
if __name__=='__main__':main()
