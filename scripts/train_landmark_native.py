"""Native128 per-landmark conditioning for large plates; no resized pixels.

Image-only coarse corners define anchors online. Training uses jittered labels;
this distinction is explicit, and held-out coarse-anchor evaluation is mandatory.
"""
import argparse,json,time,math,hashlib
from pathlib import Path
import numpy as np,torch,cv2
from torch.utils.data import Dataset,DataLoader
from train_broad import read,pool,image
from armor_native.broad import NativeBucketPrecision
from armor_native.geometry import crop_native,corner_targets
from armor_native.model import precision_loss
class Samples(Dataset):
 def __init__(self,rows,steps):self.p=pool(rows);self.keys=sorted(self.p);self.steps=steps
 def __len__(self):return self.steps
 def __getitem__(self,i):
  rng=np.random.default_rng(20261115+i);samples=[]
  for _ in range(24):
   key=self.keys[int(rng.integers(len(self.keys)))];ss=sorted(self.p[key]);pairs=self.p[key][ss[int(rng.integers(len(ss)))]];r,o=pairs[int(rng.integers(len(pairs)))];k=int(rng.integers(4));anchor=np.array(o['corners'][k])+rng.uniform(-48 if o['native_size']>128 else -16,48 if o['native_size']>128 else 16,2);patch,origin,_=crop_native(image(r['image_path']),anchor,128);c=np.array(o['corners'])-origin;hm,off,mask=corner_targets(np.repeat(c[k:k+1],4,axis=0),128);samples.append({'image':torch.from_numpy(patch.transpose(2,0,1).copy()).float()/255,'heatmap':torch.from_numpy(hm[k:k+1]),'offset':torch.from_numpy(off[2*k:2*k+2]),'point_mask':torch.from_numpy(mask[k:k+1]),'corner_id':k})
  return {k:torch.stack([s[k] for s in samples]) if k!='corner_id' else torch.tensor([s[k] for s in samples]) for k in samples[0]}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--initial',required=True);ap.add_argument('--output',required=True);ap.add_argument('--steps',type=int,default=4000);ap.add_argument('--seconds',type=int,default=1800);a=ap.parse_args();root=Path(a.data);out=Path(a.output);out.mkdir(exist_ok=False);torch.manual_seed(20261115);torch.set_num_threads(3);cv2.setNumThreads(1);rows=read(root/'train.jsonl');assert all(r['split']=='train' for r in rows);m=NativeBucketPrecision().cuda();m.load_state_dict(torch.load(a.initial,map_location='cpu',weights_only=False)['model']);opt=torch.optim.AdamW(m.parameters(),lr=.0001,weight_decay=.0001);scaler=torch.amp.GradScaler('cuda',init_scale=128);cfg={**vars(a),'initial_sha256':hashlib.sha256(Path(a.initial).read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256((root/'train.jsonl').read_bytes()).hexdigest(),'RGB_resized':False,'native_size':128,'jitter_px':{'source_bucket128':16,'source_bucket_gt128':48},'batch':24,'sampler':'equal distance/facing/light/source native size groups then sessions','online_anchor':'image-only coarse corner; must be tested independently of training jitter','output_representation':'one heatmap and subpixel offset per semantic corner, batched four native patches','selection':'fixed final4000, DEV only before sealed; report expert may be retained for128 ROI when image-supported','no_test_used':True};(out/'config.json').write_text(json.dumps(cfg,indent=2));loader=DataLoader(Samples(rows,a.steps),batch_size=None,num_workers=2,pin_memory=True,prefetch_factor=2);start=time.monotonic();skipped=0
 with (out/'train.jsonl').open('x') as log:
  for i,b in enumerate(loader):
   if time.monotonic()-start>a.seconds:raise TimeoutError('landmark training budget')
   b={k:v.cuda(non_blocking=True) for k,v in b.items()};idx=b['corner_id'];batchidx=torch.arange(len(idx),device='cuda');lr=.00001+.5*.00009*(1+math.cos(math.pi*i/max(1,a.steps-1)))
   for g in opt.param_groups:g['lr']=lr
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16):
    pred=m(b['image']);hm=pred['heatmap'][batchidx,idx][:,None];off=pred['offset'].reshape(-1,4,2,128,128)[batchidx,idx];loss,terms=precision_loss({'heatmap':hm,'offset':off},b)
   assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(m.parameters(),5);scale=scaler.get_scale();scaler.step(opt);scaler.update();skipped+=int(scaler.get_scale()<scale);row={'update':i+1,'loss':float(loss.detach()),'lr':lr,'elapsed_s':time.monotonic()-start,'skipped':skipped,**terms};log.write(json.dumps(row)+'\n');log.flush()
   if (i+1)%1000==0 or i+1==a.steps:
    torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'updates':i+1,'head':'native_landmark128','geometry':json.loads((root/'geometry.json').read_text())['object_points_armor_m'],'config':cfg},out/f'checkpoint-{i+1:06d}.pt');print(json.dumps(row),flush=True)
 (out/'result.json').write_text(json.dumps({'complete':True,'updates':a.steps,'optimizer_updates':a.steps-skipped,'seconds':time.monotonic()-start,'sealed_used':False}))
if __name__=='__main__':main()
