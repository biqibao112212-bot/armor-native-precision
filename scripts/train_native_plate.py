"""Paired native whole-plate training with/without dense planar supervision."""
import argparse,json,time,math,hashlib,collections,random,copy
from pathlib import Path
import cv2,numpy as np,torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from armor_native.plate import NativePlatePrecision,dense_targets
from armor_native.geometry import crop_native,corner_targets
from armor_native.model import precision_loss
from armor_native.height_scope import height_pool,DistanceHeightSelector
from armor_native.observability import RULES
from train_broad import read,image
from train_native_quad import sensitivity

class PlateSamples(Dataset):
    def __init__(self,rows,steps,template,seed,offset):
        assert all(r['split']=='train' for r in rows)
        self.pool=height_pool(rows);self.keys=sorted(self.pool);self.selector=DistanceHeightSelector(self.keys)
        self.steps=steps;self.template=template;self.seed=seed;self.offset=offset;self.jac={}
    def __len__(self):return self.steps
    def __getitem__(self,i):
        rng=np.random.default_rng(self.seed+i+self.offset)
        while True:
            key=self.selector.draw(rng);size=key[3]
            if size<=512:break
        sessions=sorted(self.pool[key]);samples=[]
        batch=max(1,min(4,65536//(size*size)))
        for _ in range(batch):
            ss=sessions[int(rng.integers(len(sessions)))];choices=self.pool[key][ss];r,o=choices[int(rng.integers(len(choices)))];truth=np.array(o['corners'])
            if 'training_proposal_corners' in o and rng.random()<.8:
                pc=np.array(o['training_proposal_corners']);center=(pc.min(0)+pc.max(0))/2+rng.uniform(-4,4,2)
            else:center=np.array(o['center'])+rng.uniform(-12,12,2)
            local=truth-(np.floor(center)-size//2)
            if np.min(local)<4 or np.max(local)>size-4:center=np.array(o['center'])+rng.uniform(-6,6,2)
            patch,origin,_=crop_native(image(r['image_path']),center,size);local=truth-origin
            xx=patch.astype(np.float32)/255
            if rng.random()<.35:xx=np.clip(xx*rng.uniform(.75,1.25)+rng.normal(0,rng.uniform(0,.012),xx.shape),0,1)
            hm,off,mask=corner_targets(local,size);uv,fg=dense_targets(local,size)
            ident=(r['session'],r['frame_seq'],o['slot'])
            if ident not in self.jac:self.jac[ident]=sensitivity(truth,o['K'],self.template)
            width=np.mean([np.linalg.norm(truth[2]-truth[1]),np.linalg.norm(truth[3]-truth[0])]);height=np.mean([np.linalg.norm(truth[1]-truth[0]),np.linalg.norm(truth[2]-truth[3])])
            samples.append({'image':torch.from_numpy(xx.transpose(2,0,1).copy()).float(),'heatmap':torch.from_numpy(hm),'offset':torch.from_numpy(off),'point_mask':torch.from_numpy(mask),'corners':torch.tensor(local,dtype=torch.float32),'uv':torch.from_numpy(uv),'foreground':torch.from_numpy(fg),'projected_size':torch.tensor([width,height],dtype=torch.float32),'jacobian':torch.tensor(self.jac[ident],dtype=torch.float32)})
        return {**{k:torch.stack([s[k] for s in samples]) for k in samples[0]},'range_bin':int(key[0]),'native_size':int(size),'batch_objects':batch}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--initial',required=True);ap.add_argument('--fallback',required=True);ap.add_argument('--output',required=True);ap.add_argument('--steps',type=int,default=8000);ap.add_argument('--seconds',type=int,default=1800);ap.add_argument('--dense-weight',type=float,default=1);ap.add_argument('--sample-offset',type=int,default=32000);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--device',default='cuda');ap.add_argument('--learning-rate',type=float,default=1e-4);ap.add_argument('--minimum-learning-rate',type=float,default=1e-5);ap.add_argument('--warmup',type=int,default=0);ap.add_argument('--ema-decay',type=float,default=0.);ap.add_argument('--save-every',type=int,default=1000);ap.add_argument('--context-dilations',default='');a=ap.parse_args()
    a.context_dilations=tuple(int(v) for v in a.context_dilations.split(',') if v);assert all(v>0 for v in a.context_dilations)
    assert 0<a.minimum_learning_rate<=a.learning_rate and 0<=a.ema_decay<1 and a.save_every>0 and 0<=a.warmup<a.steps
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);seed=2026102307;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(3);cv2.setNumThreads(1)
    root=Path(a.data);rows=read(root/'train.jsonl');assert all(r.get('requested_level_target_verified') for r in rows)
    ck=torch.load(a.initial,map_location='cpu',weights_only=False);model=NativePlatePrecision(a.context_dilations)
    if ck['head']=='native_quad_geometry':
        state={k.removeprefix('backbone.'):v for k,v in ck['model'].items() if k.startswith('backbone.')}
    else:state=ck['model']
    missing,extra=model.load_state_dict(state,strict=False);assert not extra and all(k.startswith(('dense.','wide_context.')) for k in missing)
    model.to(a.device);torch.manual_seed(seed)
    ds=PlateSamples(rows,a.steps,ck['geometry'],seed,a.sample_offset)
    config={**vars(a),'seed':seed,'initial_sha256':hashlib.sha256(Path(a.initial).read_bytes()).hexdigest(),'train_sha256':hashlib.sha256((root/'train.jsonl').read_bytes()).hexdigest(),'input_rgb_resized':False,'split':'TRAIN only','training':'shared full native plate, same corner/geometry targets; candidate differs only dense supervision weight','dense_target':'inverse homography to unit square from physical four corners; no extra independent measurement claimed','fallback':'fixed quad model for coarse extent requiring>512 pixels','selection':'fixed final weights on DEV; sealed confirmation untouched'}
    (out/'config.json').write_text(json.dumps(config,indent=2));loader=DataLoader(ds,batch_size=None,num_workers=a.workers,pin_memory=a.device=='cuda',**({'prefetch_factor':2} if a.workers else {}))
    ema=copy.deepcopy(model).eval() if a.ema_decay else None
    opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=1e-4);scaler=torch.amp.GradScaler('cuda',enabled=a.device=='cuda',init_scale=128)
    start=time.monotonic();skipped=0;draws=collections.Counter()
    def save(step):
        torch.save({'head':'native_dense_plate','model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'updates':step,'geometry':ck['geometry'],'config':config,'inference_config':{'proposal_threshold':.1,'minimum_corner_probability':.1,'duplicate_max_px':3,'observability_rule':dict(RULES['two-lamp-cyan-v2']),'decoder':'corners','fallback_weights':a.fallback}},out/f'checkpoint-{step:06d}.pt')
    def save_ema(step):
        if ema is None:return
        torch.save({'head':'native_dense_plate','model':ema.state_dict(),'updates':step,'geometry':ck['geometry'],'config':dict(config,weight_average='EMA of successful optimizer steps'),'inference_config':{'proposal_threshold':.1,'minimum_corner_probability':.1,'duplicate_max_px':3,'observability_rule':dict(RULES['two-lamp-cyan-v2']),'decoder':'corners','fallback_weights':a.fallback}},out/f'ema-{step:06d}.pt')
    with (out/'train.jsonl').open('x') as log:
        for i,b in enumerate(loader):
            if time.monotonic()-start>a.seconds:save(i);raise TimeoutError('plate training budget')
            draws[str(b['range_bin'])]+=b['batch_objects'];size=b['native_size'];b={k:v.to(a.device,non_blocking=True) for k,v in b.items() if torch.is_tensor(v)}
            lr=a.minimum_learning_rate+.5*(a.learning_rate-a.minimum_learning_rate)*(1+math.cos(math.pi*max(0,i-a.warmup)/max(1,a.steps-a.warmup-1)))
            if a.warmup and i<a.warmup:lr=a.minimum_learning_rate+(a.learning_rate-a.minimum_learning_rate)*(i+1)/a.warmup
            for g in opt.param_groups:g['lr']=lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16,enabled=a.device=='cuda'):pred=model(b['image'])
            base,terms=precision_loss({k:pred[k] for k in ['heatmap','offset']},b)
            err=pred['corners']-b['corners'];coord=F.smooth_l1_loss(err/.236,torch.zeros_like(err));shape=F.smooth_l1_loss((err-err.mean(1,keepdim=True))/.236,torch.zeros_like(err))
            good=err.detach().abs().amax((1,2))<4;d=torch.bmm(b['jacobian'],err.reshape(-1,8,1)).squeeze(-1)
            norm=err.new_tensor([.0106,.0106,.0743]);geo=(F.smooth_l1_loss(d/norm,torch.zeros_like(d),reduction='none').mean(1)*good).sum()/good.sum().clamp_min(1)
            mask=b['foreground'];count=mask.sum().clamp_min(1)
            uv_error=(pred['uv']-b['uv'])*b['projected_size'][:,:,None,None]/.5
            dense=(F.smooth_l1_loss(uv_error,torch.zeros_like(uv_error),reduction='none')*mask).sum()/(2*count)
            pos_weight=((mask.numel()-count)/count).clamp(1,30)
            foreground=F.binary_cross_entropy_with_logits(pred['foreground'],mask,pos_weight=pos_weight)
            loss=base+.1*coord+.2*(geo+.25*shape)+a.dense_weight*(.2*dense+.5*foreground)
            assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt);gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5);old=scaler.get_scale();scaler.step(opt);scaler.update();skipped+=int(scaler.get_scale()<old)
            if ema is not None and scaler.get_scale()>=old:
                with torch.no_grad():
                    values=model.state_dict()
                    for name,value in ema.state_dict().items():
                        if value.is_floating_point():value.lerp_(values[name],1-a.ema_decay)
                        else:value.copy_(values[name])
            row={'update':i+1,'loss':float(loss.detach()),'coordinate':float(coord.detach()),'geometry':float(geo.detach()),'dense':float(dense.detach()),'foreground':float(foreground.detach()),'gradient_norm':float(gn),'elapsed_s':time.monotonic()-start,'skipped':skipped,'native_size':size,'learning_rate':lr,**terms};log.write(json.dumps(row)+'\n');log.flush()
            if (i+1)%a.save_every==0:save(i+1);save_ema(i+1);print(json.dumps(row),flush=True)
    if a.steps%a.save_every:save(a.steps);save_ema(a.steps)
    (out/'result.json').write_text(json.dumps({'complete':True,'updates':a.steps,'optimizer_updates':a.steps-skipped,'seconds':time.monotonic()-start,'range_draws':dict(draws),'sealed_used':False},indent=2))
if __name__=='__main__':main()
