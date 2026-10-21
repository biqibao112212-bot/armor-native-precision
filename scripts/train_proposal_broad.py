"""One bounded broad-domain native448 proposal adaptation; no direct-depth head use."""
import argparse,json,time,math,hashlib,collections
from pathlib import Path
import cv2,numpy as np,torch
from torch.nn import functional as F
from armor_native.proposals import ProposalNetwork
from armor_native.geometry import crop_native
from train_broad import image,pool

def dump(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def sample(rows,p,keys,rng):
 key=keys[int(rng.integers(len(keys)))];sessions=sorted(p[key]);ss=sessions[int(rng.integers(len(sessions)))];r,o=p[key][ss][int(rng.integers(len(p[key][ss])))];jitter=max(0,min(128,(448-o['extent_px'])/2-12));anchor=np.array(o['center'])+rng.uniform(-jitter,jitter,2)
 if rng.random()<.2:anchor=rng.uniform([224,224],[1216,856])
 patch,origin,valid=crop_native(image(r['image_path']),anchor,448);yy,xx=np.ogrid[:112,:112];hm=np.zeros((1,112,112),np.float32);geo=np.zeros((11,112,112),np.float32);mask=np.zeros((1,112,112),np.float32);geomask=mask.copy();neg=valid.reshape(112,4,112,4).all(axis=(1,3))[None].astype(np.float32)
 for box in r.get('coarse_ignored_boxes',[]):
  x0,y0,x1,y1=(np.array(box)-np.tile(origin,2))/4;x0,y0=np.clip(np.floor([x0,y0]).astype(int),0,112);x1,y1=np.clip(np.ceil([x1,y1]).astype(int)+1,0,112)
  if x1>x0 and y1>y0:neg[:,y0:y1,x0:x1]=0
 for obj in r['objects']:
  c=np.array(obj['center'])-origin;ix,iy=np.floor(c/4).astype(int)
  if not(0<=ix<112 and 0<=iy<112):continue
  hm[0]=np.maximum(hm[0],np.exp(-((xx-ix)**2+(yy-iy)**2)/2));mask[0,iy,ix]=1;neg[0,iy,ix]=1;geo[:2,iy,ix]=c/4-[ix,iy];corners=np.array(obj['screen_corners'])-origin
  if (corners>=0).all() and (corners<448).all():geomask[0,iy,ix]=1;geo[2:10,iy,ix]=((np.array(obj['screen_corners'])-obj['center'])/32).reshape(-1)
 if rng.random()<.25:patch=np.clip(patch.astype(float)*rng.uniform(.75,1.25)+rng.normal(0,rng.uniform(0,3),patch.shape),0,255).astype(np.uint8)
 return [torch.from_numpy(patch.transpose(2,0,1).copy()).float()/255,*[torch.from_numpy(x) for x in [hm,geo,mask,geomask,neg]]]
def loss_fn(pred,hm,gt,mask,geomask,neg):
 logits,geo=pred;p=logits.float().sigmoid().clamp(1e-5,1-1e-5);n=mask.sum().clamp_min(1);focal=-(torch.log(p)*(1-p)**2*(hm==1)+torch.log(1-p)*p**2*(1-hm)**4*(hm<1)*neg).sum()/n;off=(F.smooth_l1_loss(geo[:,:2].float().sigmoid(),gt[:,:2],reduction='none')*mask).sum()/n;corner=(F.smooth_l1_loss(geo[:,2:10].float(),gt[:,2:10],reduction='none')*geomask).sum()/geomask.sum().clamp_min(1);return focal+.2*off+corner,{'center':float(focal.detach()),'offset':float(off.detach()),'corner_extent':float(corner.detach())}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--initial',required=True);ap.add_argument('--config',required=True);ap.add_argument('--output',required=True);ap.add_argument('--steps',type=int,default=1500);ap.add_argument('--seconds',type=int,default=1800);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);rows=[json.loads(l) for l in Path(a.data).open()];assert all(r['split']=='train' and r['requested_roll_zero_verified'] for r in rows);p=pool(rows);keys=sorted(p);seed=20261028;torch.manual_seed(seed);rng=np.random.default_rng(seed);torch.set_num_threads(4);cv2.setNumThreads(1);model=ProposalNetwork(json.loads(Path(a.config).read_text())).cuda();model.load_state_dict(torch.load(a.initial,map_location='cpu',weights_only=False)['model']);model.train();model.backbone.eval();back=list(model.backbone.parameters());ids={id(v) for v in back};head=[v for v in model.parameters() if id(v) not in ids]
 for v in back:v.requires_grad=False
 opt=torch.optim.AdamW([{'params':head,'lr':.0003},{'params':back,'lr':.00003}],weight_decay=.0001);scaler=torch.amp.GradScaler('cuda',init_scale=128);started=time.monotonic();skipped=0;draws=collections.Counter();cfg={**vars(a),'seed':seed,'initial_sha256':hashlib.sha256(Path(a.initial).read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),'native_rgb_resized':False,'native_tile':448,'direct_depth_loss':False,'missing_labels_masked':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};dump(out/'config.json',cfg)
 def save(step):torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'updates':step,'optimizer_updates':step-skipped,'head':'broad_native_proposal','config':cfg,'rng_state':rng.bit_generator.state},out/f'checkpoint-{step:06d}.pt')
 with (out/'train.jsonl').open('x') as log:
  for step in range(a.steps):
   if time.monotonic()-started>a.seconds:save(step);raise TimeoutError('bounded proposal adaptation')
   if step==200:
    for v in back:v.requires_grad=True
    model.backbone.train()
   samples=[sample(rows,p,keys,rng) for _ in range(4)];batch=[torch.stack([s[i] for s in samples]).cuda() for i in range(6)];fraction=.1+.9*.5*(1+math.cos(math.pi*step/(a.steps-1)));opt.param_groups[0]['lr']=.0003*fraction;opt.param_groups[1]['lr']=.00003*fraction;opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16):pred=model(batch[0]);loss,terms=loss_fn(pred,*batch[1:])
   if not torch.isfinite(loss):raise RuntimeError('nonfinite proposal loss')
   scaler.scale(loss).backward();scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);old=scaler.get_scale();scaler.step(opt);scaler.update();skipped+=int(scaler.get_scale()<old);done=step+1;log.write(json.dumps({'update':done,'loss':float(loss.detach()),'grad_norm':float(norm) if torch.isfinite(norm) else None,'skipped_updates':skipped,'elapsed_s':time.monotonic()-started,**terms})+'\n');log.flush()
   if done%500==0 or done==a.steps:save(done)
 dump(out/'result.json',{'complete':True,'updates':a.steps,'optimizer_updates':a.steps-skipped,'seconds':time.monotonic()-started,'rgb_resized':False});print(json.dumps({'complete':True,'output':str(out)}),flush=True)
if __name__=='__main__':main()
