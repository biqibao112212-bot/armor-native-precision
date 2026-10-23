"""Train native 4-lamp + 12-numeral landmarks with pose-sensitive supervision."""
import argparse,json,time,math,random,collections,hashlib,copy
from pathlib import Path
import numpy as np,torch,cv2
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from armor_native.glyph import NativeGlyphPrecision,point_targets
from armor_native.height_scope import height_pool,DistanceHeightSelector
from armor_native.geometry import crop_native
from armor_native.model import precision_loss
from armor_native.observability import RULES
from train_broad import read,image

class GlyphSamples(Dataset):
    def __init__(self,rows,steps,seed,offset):
        assert all(r['split']=='train' and r['requested_level_target_verified'] for r in rows)
        self.pool=height_pool(rows);self.keys=sorted(self.pool);self.selector=DistanceHeightSelector(self.keys);self.steps=steps;self.seed=seed;self.offset=offset
    def __len__(self):return self.steps
    def __getitem__(self,i):
        rng=np.random.default_rng(self.seed+i+self.offset)
        while True:
            key=self.selector.draw(rng);size=key[3]
            if size<=512:break
        sessions=sorted(self.pool[key]);batch=max(1,min(4,65536//(size*size)));samples=[]
        for _ in range(batch):
            ss=sessions[int(rng.integers(len(sessions)))];choices=self.pool[key][ss];r,o=choices[int(rng.integers(len(choices)))];points=np.asarray(o['glyph_points_image']);visible=np.asarray(o['glyph_visible'],bool)
            if 'training_proposal_corners' in o and rng.random()<.8:
                pc=np.asarray(o['training_proposal_corners']);center=(pc.min(0)+pc.max(0))/2+rng.uniform(-4,4,2)
            else:center=np.asarray(o['center'])+rng.uniform(-12,12,2)
            local=points-(np.floor(center)-size//2)
            if local[visible].min()<4 or local[visible].max()>size-4:center=(points.min(0)+points.max(0))/2+rng.uniform(-6,6,2)
            patch,origin,_=crop_native(image(r['image_path']),center,size);local=points-origin
            xx=patch.astype(np.float32)/255
            if rng.random()<.35:xx=np.clip(xx*rng.uniform(.75,1.25)+rng.normal(0,rng.uniform(0,.012),xx.shape),0,1)
            hm,off,mask=point_targets(local,visible,size)
            samples.append({'image':torch.from_numpy(xx.transpose(2,0,1).copy()).float(),'heatmap':torch.from_numpy(hm),'offset':torch.from_numpy(off),'point_mask':torch.from_numpy(mask),'corners':torch.tensor(local,dtype=torch.float32),'visible':torch.tensor(visible),'jacobian':torch.tensor(o['glyph_jacobian'],dtype=torch.float32)})
        return {**{k:torch.stack([s[k] for s in samples]) for k in samples[0]},'range_bin':int(key[0]),'native_size':int(size),'batch_objects':batch}

def head_loss(pred,b,start,end):
    return precision_loss({'heatmap':pred['heatmap'][:,start:end],'offset':pred['offset'][:,2*start:2*end]}, {'heatmap':b['heatmap'][:,start:end],'offset':b['offset'][:,2*start:2*end],'point_mask':b['point_mask'][:,start:end]})[0]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--initial',required=True);ap.add_argument('--fallback',required=True);ap.add_argument('--output',required=True);ap.add_argument('--steps',type=int,default=32000);ap.add_argument('--seconds',type=int,default=2400);ap.add_argument('--sample-offset',type=int,default=64000);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--device',default='cuda');ap.add_argument('--learning-rate',type=float,default=3e-4);ap.add_argument('--warmup',type=int,default=1000);ap.add_argument('--ema-decay',type=float,default=.999);ap.add_argument('--save-every',type=int,default=8000);ap.add_argument('--observability-rule',choices=sorted(RULES));a=ap.parse_args()
    assert 0<=a.warmup<a.steps and 0<=a.ema_decay<1
    root=Path(a.data);assert json.loads((root/'result.json').read_text())['complete'];out=Path(a.output);out.mkdir(exist_ok=False)
    torch.set_num_threads(3);cv2.setNumThreads(1);seed=2026102313;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    geo=json.loads((root/'glyph-geometry.json').read_text());rows=read(root/'train.jsonl');ck=torch.load(a.initial,map_location='cpu',weights_only=False);assert ck['head'] in {'native_dense_plate','native_glyph_geometry'}
    model=NativeGlyphPrecision(len(geo['object_points']),(2,4,8))
    if ck['head']=='native_glyph_geometry':
        assert tuple(ck['config']['context_dilations'])==(2,4,8)
        np.testing.assert_allclose(ck['glyph_geometry']['object_points'],geo['object_points'],atol=1e-8,rtol=0)
        model.load_state_dict(ck['model'])
    else:model.initialize_from_plate(ck['model'])
    model.to(a.device);torch.manual_seed(seed);ema=copy.deepcopy(model).eval()
    ds=GlyphSamples(rows,a.steps,seed,a.sample_offset);loader=DataLoader(ds,batch_size=None,num_workers=a.workers,pin_memory=a.device=='cuda',**({'prefetch_factor':2} if a.workers else {}))
    config={**vars(a),'seed':seed,'context_dilations':[2,4,8],'geometry_source':'fixed public #3 numeral contour vertices; physical lamp template from first TRAIN annotation','initial_head':ck['head'],'optimizer_policy':'new AdamW state; weight initialization is not exact optimizer resume','initial_sha256':hashlib.sha256(Path(a.initial).read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256((root/'train.jsonl').read_bytes()).hexdigest(),'native_rgb':True,'split':'TRAIN only','geometry_supervision':'linearized calibrated translation sensitivity of visible physical/numeral points; no target metadata at inference','selection':'fixed final and EMA weights on DEV; independent confirmation unused'}
    inference={'proposal_threshold':.1,'minimum_point_probability':.1,'minimum_glyph_points':4,'minimum_total_points':8,'observability_rule':dict(RULES[a.observability_rule] if a.observability_rule else ck.get('inference_config',{}).get('observability_rule',RULES['two-lamp-cyan-v2'])),'decoder':'glyph_level','fallback_weights':a.fallback,'camera_attitude':'same-exposure public sensor input; no target orientation input'}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=1e-4);scaler=torch.amp.GradScaler('cuda',enabled=a.device=='cuda',init_scale=128)
    skipped=0;draws=collections.Counter();started=time.monotonic()
    def save(step):
        common={'head':'native_glyph_geometry','geometry':geo['physical_points'],'glyph_geometry':geo,'updates':step,'config':config,'inference_config':inference}
        torch.save(dict(common,model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict()),out/f'checkpoint-{step:06d}.pt')
        torch.save(dict(common,model=ema.state_dict(),weight_average='EMA over successful optimizer steps'),out/f'ema-{step:06d}.pt')
    with (out/'train.jsonl').open('x') as log:
        for i,b in enumerate(loader):
            if time.monotonic()-started>a.seconds:save(i);raise TimeoutError('glyph training budget')
            draws[str(b['range_bin'])]+=b['batch_objects'];size=b['native_size'];b={k:v.to(a.device,non_blocking=True) for k,v in b.items() if torch.is_tensor(v)}
            lr=1e-5+.5*(a.learning_rate-1e-5)*(1+math.cos(math.pi*max(0,i-a.warmup)/max(1,a.steps-a.warmup-1)))
            if a.warmup and i<a.warmup:lr=1e-5+(a.learning_rate-1e-5)*(i+1)/a.warmup
            for g in opt.param_groups:g['lr']=lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16,enabled=a.device=='cuda'):pred=model(b['image'])
            lamp=head_loss(pred,b,0,4);glyph=head_loss(pred,b,4,len(geo['object_points']))
            err=pred['corners']-b['corners'];visible=b['visible'];near=(err.detach().abs().amax(2)<6)&visible;near_count=near.sum().clamp_min(1)
            coord=(F.smooth_l1_loss(err/.236,torch.zeros_like(err),reduction='none')*near[...,None]).sum()/(2*near_count)
            good=torch.where(visible,err.detach().abs().amax(2),0).amax(1)<4
            delta=torch.bmm(b['jacobian'],err.reshape(-1,2*len(geo['object_points']),1)).squeeze(-1);norm=err.new_tensor([.0106,.0106,.0743])
            geometry=(F.smooth_l1_loss(delta/norm,torch.zeros_like(delta),reduction='none').mean(1)*good).sum()/good.sum().clamp_min(1)
            loss=lamp+glyph+.1*coord+.2*geometry;assert torch.isfinite(loss)
            scaler.scale(loss).backward();scaler.unscale_(opt);gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5);old=scaler.get_scale();scaler.step(opt);scaler.update();did_skip=scaler.get_scale()<old;skipped+=int(did_skip)
            if not did_skip:
                with torch.no_grad():
                    current=model.state_dict()
                    for k,v in ema.state_dict().items():
                        if v.is_floating_point():v.lerp_(current[k],1-a.ema_decay)
                        else:v.copy_(current[k])
            row={'update':i+1,'loss':float(loss.detach()),'lamp_loss':float(lamp.detach()),'glyph_loss':float(glyph.detach()),'coordinate':float(coord.detach()),'geometry':float(geometry.detach()),'near_visible_points':int(near.sum()),'visible_points':int(visible.sum()),'gradient_norm':float(gn),'learning_rate':lr,'native_size':size,'elapsed_s':time.monotonic()-started,'skipped':skipped}
            log.write(json.dumps(row)+'\n');log.flush()
            if (i+1)%1000==0:print(json.dumps(row),flush=True)
            if (i+1)%a.save_every==0:save(i+1)
    if a.steps%a.save_every:save(a.steps)
    result={'complete':True,'seconds':time.monotonic()-started,'updates':a.steps,'optimizer_updates':a.steps-skipped,'range_draws':dict(draws),'sealed_used':False};(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
