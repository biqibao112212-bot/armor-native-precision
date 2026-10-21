"""Bounded native-size number3 training. No test access or RGB interpolation."""
import argparse,json,random,time,math,collections,functools,hashlib
from pathlib import Path
import cv2,numpy as np,torch
from armor_native.broad import NativeBucketPrecision,NativeGlobalPrecision,coarse_context_loss,solve_physical,statistics
from armor_native.model import precision_loss
from armor_native.geometry import crop_native,corner_targets,local_camera

def dump(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def read(path):return [json.loads(l) for l in Path(path).open()]
@functools.lru_cache(maxsize=192)
def image(path):
 a=cv2.imread(path);assert a is not None;return cv2.cvtColor(a,cv2.COLOR_BGR2RGB)
def sample(r,o,rng,augmentation):
 size=o['native_size'];anchor=np.asarray(o['center'])+rng.uniform(-16,16,2);patch,origin,valid=crop_native(image(r['image_path']),anchor,size);c=np.asarray(o['corners'])-origin
 aug='none'
 if augmentation:
  # Pixel-value noise/blur only, no spatial scaling or perspective relabeling.
  if rng.random()<.2:
   length=int(rng.choice([3,5,7]));kernel=np.zeros((length,length),np.float32);direction=int(rng.integers(4))
   if direction==0:kernel[length//2,:]=1
   elif direction==1:kernel[:,length//2]=1
   elif direction==2:np.fill_diagonal(kernel,1)
   else:kernel[np.arange(length),np.arange(length-1,-1,-1)]=1
   patch=cv2.filter2D(patch,-1,kernel/kernel.sum());aug=f'synthetic_blur_{length}_{direction}'
  if rng.random()<.35:
   gain=float(rng.uniform(.65,1.35));sigma=float(rng.uniform(0,5));patch=np.clip(patch.astype(np.float32)*gain+rng.normal(0,sigma,patch.shape),0,255).astype(np.uint8);aug+='+photometric_noise'
 hm,off,mask=corner_targets(c,size)
 return {'image':torch.from_numpy(patch.transpose(2,0,1).copy()).float()/255,'heatmap':torch.from_numpy(hm),'offset':torch.from_numpy(off),'point_mask':torch.from_numpy(mask),'corners':c,'K':local_camera(o['K'],origin),'xyz':np.array(o['xyz_m']),'r':r,'o':o,'augmentation':aug}
def pool(rows):
 p={}
 for r in rows:
  for o in r['objects']:
   k=(o['range_bin'],min(2,int(max(0,o['facing_cos'])*3)),r['light'],o['native_size']);p.setdefault(k,{}).setdefault(r['session'],[]).append((r,o))
 return p

def diagnostic(model,rows,template,device,seed=1042,n=192):
 model.eval();p=pool(rows);keys=sorted(p);rng=np.random.default_rng(seed);records=[];started=time.monotonic()
 with torch.inference_mode():
  for i in range(n):
   key=keys[i%len(keys)];sessions=sorted(p[key]);ss=sessions[int(rng.integers(len(sessions)))];pairs=p[key][ss];r,o=pairs[int(rng.integers(len(pairs)))];s=sample(r,o,rng,False);x=s['image'][None].to(device)
   with torch.autocast('cuda',dtype=torch.float16):pred=model(x)
   c=model.points(pred).cpu().numpy()[0];pose=solve_physical(c,s['K'],template);rec={'session':r['session'],'range_bin':o['range_bin'],'range_m':o['range_m'],'native_size':o['native_size'],'corner_rms_px':float(np.sqrt(np.mean((c-s['corners'])**2))),'valid_pnp':pose is not None}
   if pose:
    delta=(np.array(pose['xyz_m'])-s['xyz'])*1000;rec.update(depth_mm=float(abs(delta[2])),xy_mm=float(np.linalg.norm(delta[:2])),signed_xyz_mm=delta.tolist())
   records.append(rec)
 model.train()
 def summarize(rr):return {'objects':len(rr),'corner_rms_px':statistics([r['corner_rms_px'] for r in rr]),'valid_pnp':sum(r['valid_pnp'] for r in rr),'depth_mm':statistics([r['depth_mm'] for r in rr if r['valid_pnp']]),'xy_mm':statistics([r['xy_mm'] for r in rr if r['valid_pnp']])}
 return {'scope':'GT-centered jittered native-ROI development diagnostic, not detector or prediction acceptance','all':summarize(records),'by_range':{str(k):summarize([r for r in records if r['range_bin']==k]) for k in sorted(set(r['range_bin'] for r in records))},'seconds':time.monotonic()-started,'records':records}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--architecture',choices=['spatial','global'],default='spatial');ap.add_argument('--output',required=True);ap.add_argument('--initial',required=True);ap.add_argument('--steps',type=int,default=4000);ap.add_argument('--seconds',type=int,default=2700);a=ap.parse_args();root=Path(a.data);out=Path(a.output);out.mkdir(exist_ok=False,parents=True);seed=20261022;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4);cv2.setNumThreads(1);device='cuda';train=read(root/'train.jsonl');dev=read(root/'dev.jsonl');assert not {r['session'] for r in train}&{r['session'] for r in dev};template=json.loads((root/'geometry.json').read_text())['object_points_armor_m'];p=pool(train);keys=sorted(p);assert len(keys)>0
 model=(NativeGlobalPrecision() if a.architecture=='global' else NativeBucketPrecision()).to(device);initial=torch.load(a.initial,map_location='cpu',weights_only=False);incompat=model.load_state_dict(initial['model'],strict=a.architecture=='spatial');assert not incompat.unexpected_keys;assert all(k.startswith(('global_context.','global_heatmap.')) for k in incompat.missing_keys);opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.0001);scaler=torch.amp.GradScaler('cuda',init_scale=128);rng=np.random.default_rng(seed);skipped=0;draw_counts=collections.Counter();aug_counts=collections.Counter();started=time.monotonic()
 config={**vars(a),'seed':seed,'coarse_auxiliary_weight':.2 if a.architecture=='global' else None,'global_feature_grid':[8,8] if a.architecture=='global' else None,'execution_source_sha256':{str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).parents[1]/'src/armor_native/broad.py']},'initial_sha256':hashlib.sha256(Path(a.initial).read_bytes()).hexdigest(),'sampler':'equal nonempty physical-distance/facing/light/native-size groups, then equal sessions, then object','native_pixel_budget_per_update':16*128*128,'native_sizes':sorted(set(k[-1] for k in keys)),'input_rgb_resized':False,'semantic_order':'physical armor coordinates; fixed training-only asset template','training_frames':len(train),'training_objects':sum(len(r['objects']) for r in train),'dev_frames':len(dev),'sealed_test_opened':False,'augmentation':{'motion_blur_probability':.2,'motion_blur_lengths_px':[3,5,7],'physical_exposure_integration':False,'gain_and_noise_probability':.35},'selection':'fixed final4000; dev diagnostics only; no tuning on sealed test','plan_sha256':hashlib.sha256((root.parent/'plan.json').read_bytes()).hexdigest()};dump(out/'config.json',config)
 def checkpoint(step):
  path=out/f'checkpoint-{step:06d}.pt';torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'updates':step,'optimizer_updates':step-skipped,'head':'global_bucket_physical' if a.architecture=='global' else 'native_bucket_physical','geometry':template,'config':config,'rng_state':rng.bit_generator.state,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},path)
 dump(out/'diagnostic-000000.json',diagnostic(model,dev,template,device));model.train();completed=0
 with (out/'train.jsonl').open('x') as log:
  for step in range(a.steps):
   if time.monotonic()-started>a.seconds:checkpoint(completed);raise TimeoutError('bounded training time exceeded; partial checkpoint retained')
   key=keys[int(rng.integers(len(keys)))];ss=sorted(p[key]);size=key[-1];batch_size=max(1,min(16,(16*128*128)//(size*size)));samples=[]
   for j in range(batch_size):
    session=ss[int(rng.integers(len(ss)))];pairs=p[key][session];r,o=pairs[int(rng.integers(len(pairs)))];samples.append(sample(r,o,rng,True));draw_counts[str(o['range_bin'])]+=1;aug_counts[samples[-1]['augmentation']]+=1
   batch={k:torch.stack([s[k] for s in samples]).to(device) for k in ['image','heatmap','offset','point_mask']};lr=.00005+.5*(.0005-.00005)*(1+math.cos(math.pi*step/(a.steps-1)))
   for g in opt.param_groups:g['lr']=lr
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16):pred=model(batch['image']);loss,terms=precision_loss(pred,batch)
   if a.architecture=='global':
    gt_corners=torch.tensor(np.stack([s['corners'] for s in samples]),dtype=torch.float32,device=device);aux=coarse_context_loss(pred['coarse_heatmap'],gt_corners,size);loss=loss+.2*aux;terms['coarse_context']=float(aux.detach())
   if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
   scaler.scale(loss).backward();scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);old=scaler.get_scale();scaler.step(opt);scaler.update();skipped+=int(scaler.get_scale()<old);completed=step+1
   log.write(json.dumps({'update':completed,'loss':float(loss.detach()),'grad_norm':float(norm) if torch.isfinite(norm) else None,'lr':lr,'native_size':size,'batch_size':batch_size,'skipped_updates':skipped,'elapsed_s':time.monotonic()-started,**terms})+'\n');log.flush()
   if completed%500==0:checkpoint(completed)
   if completed%1000==0:dump(out/f'diagnostic-{completed:06d}.json',diagnostic(model,dev,template,device))
 result={'complete':True,'updates':completed,'optimizer_updates':completed-skipped,'seconds':time.monotonic()-started,'range_draw_counts':dict(draw_counts),'synthetic_augmentation_counts':dict(aug_counts),'sealed_test_used':False};dump(out/'result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
