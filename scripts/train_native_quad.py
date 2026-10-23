"""Bounded paired four-crop training; geometry labels are TRAIN-only."""
import argparse,json,time,math,hashlib,random,collections
from pathlib import Path
import cv2,numpy as np,torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from train_broad import read,pool,image
from armor_native.quad import NativeQuadPrecision, gaussian_coordinate_loss
from armor_native.dense_vote import dense_vote_loss
from armor_native.sampling import GroupSelector
from armor_native.height_scope import height_pool, height_region, DistanceHeightSelector
from armor_native.observability import RULE,RULES
from armor_native.broad import solve_physical
from armor_native.geometry import crop_native,corner_targets
from armor_native.model import precision_loss

def sensitivity(c,K,template):
    J=np.zeros((3,8));epsilon=.002
    for k in range(8):
        p=c.copy();m=c.copy();p.flat[k]+=epsilon;m.flat[k]-=epsilon
        sp=solve_physical(p,K,template);sm=solve_physical(m,K,template)
        if sp is None or sm is None:raise ValueError('TRAIN Jacobian has invalid oracle pose')
        J[:,k]=(np.asarray(sp['xyz_m'])-sm['xyz_m'])/(2*epsilon)
    return J

class Samples(Dataset):
    def __init__(self,rows,steps,template,seed,anchor_mode='uniform',sample_offset=0,sampling='groups'):
        assert all(r['split']=='train' for r in rows)
        self.p=height_pool(rows) if sampling=='distance_height' else pool(rows);self.keys=sorted(self.p);self.steps=steps;self.template=template;self.seed=seed;self.jac={};self.anchor_mode=anchor_mode;self.sample_offset=sample_offset;self.selector=DistanceHeightSelector(self.keys) if sampling=='distance_height' else GroupSelector(self.keys,sampling)
    def __len__(self):return self.steps
    def __getitem__(self,i):
        i+=self.sample_offset; rng=np.random.default_rng(self.seed+i);samples=[]
        for sample_index in range(4):
            key=self.selector.draw(rng);ss=sorted(self.p[key]);pairs=self.p[key][ss[int(rng.integers(len(ss)))]];r,o=pairs[int(rng.integers(len(pairs)))]
            corners=np.asarray(o['corners']);jitter=48 if o['native_size']>128 else 16
            anchors=corners+rng.uniform(-jitter,jitter,(4,2))
            if self.anchor_mode=='image_coarse':
                arng=np.random.default_rng(self.seed+11113+i*4+sample_index)
                if 'training_proposal_corners' in o and arng.random()<.8:
                    anchors=np.asarray(o['training_proposal_corners'])+arng.uniform(-4,4,2)+arng.uniform(-1,1,(4,2))
                else:
                    span=24 if o['native_size']>128 else 8
                    anchors=corners+arng.uniform(-span,span,2)+arng.uniform(-3,3,(4,2))
                local_check=corners-(np.floor(anchors)-64)
                if (local_check<6).any() or (local_check>122).any():
                    span=24 if o['native_size']>128 else 8
                    anchors=corners+arng.uniform(-span,span,2)+arng.uniform(-3,3,(4,2))
            patches=[];origins=[]
            for anchor in anchors:
                patch,origin,_=crop_native(image(r['image_path']),anchor,128);patches.append(patch);origins.append(origin)
            # The same photometric augmentation applies to all four crops.
            xx=np.asarray(patches).astype(np.float32)/255
            if rng.random()<.35:xx=np.clip(xx*rng.uniform(.75,1.25)+rng.normal(0,rng.uniform(0,.012),xx.shape),0,1)
            local=corners-np.asarray(origins);hm,off,mask=corner_targets(local,128)
            ident=(r['session'],r['frame_seq'],o['slot'])
            if ident not in self.jac:self.jac[ident]=sensitivity(corners,o['K'],self.template)
            samples.append({'image':torch.from_numpy(xx.transpose(0,3,1,2).copy()).float(),'heatmap':torch.from_numpy(hm),'offset':torch.from_numpy(off),'point_mask':torch.from_numpy(mask),'corners':torch.from_numpy(local).float(),'anchors':torch.from_numpy(anchors).float(),'jacobian':torch.from_numpy(self.jac[ident]).float(),'range_bin':torch.tensor(o['range_bin'],dtype=torch.int64),'height_region':torch.tensor(height_region(o['height_difference_m']) if isinstance(o.get('height_difference_m'),(int,float)) and abs(o['height_difference_m'])<=.4+1e-9 else -1,dtype=torch.int64)})
        return {k:torch.stack([s[k] for s in samples]) for k in samples[0]}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--initial',required=True);ap.add_argument('--output',required=True);ap.add_argument('--joint',type=int,required=True);ap.add_argument('--geometry-weight',type=float,required=True);ap.add_argument('--steps',type=int,default=2000);ap.add_argument('--seconds',type=int,default=900);ap.add_argument('--anchor-mode',choices=['uniform','image_coarse'],default='uniform');ap.add_argument('--sample-offset',type=int,default=0);ap.add_argument('--sampling',choices=['groups','distance','distance_height'],default='groups');ap.add_argument('--require-level-body',action='store_true');ap.add_argument('--lamp-rule',choices=sorted(RULES),default=RULE['version']);ap.add_argument('--width-multiplier',type=int,choices=[1,2],default=1);ap.add_argument('--coordinate-mode',choices=['legacy','continuous_gaussian','dense_votes'],default='legacy');ap.add_argument('--refinement-mode',choices=['none','joint_residual','joint_raw'],default='none');ap.add_argument('--dense-vote-weight',type=float,default=0.0);ap.add_argument('--precision-weight',type=float,default=1.0);a=ap.parse_args()
    assert a.dense_vote_weight>=0 and (a.coordinate_mode=='dense_votes' or a.dense_vote_weight==0)
    assert a.precision_weight>0
    root=Path(a.data);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);seed=2026102203
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(3);cv2.setNumThreads(1)
    ck=torch.load(a.initial,map_location='cpu',weights_only=False);m=NativeQuadPrecision(bool(a.joint),a.coordinate_mode,a.width_multiplier,a.refinement_mode).cuda()
    if ck['head']=='native_quad_geometry':
        assert ck['config']['joint']==bool(a.joint)
        source_mode=ck['config'].get('coordinate_mode','legacy')
        source_width=ck['config'].get('width_multiplier',1)
        source_refinement=ck['config'].get('refinement_mode','none')
        if source_refinement!=a.refinement_mode:
            assert source_refinement=='none' and a.refinement_mode in {'joint_residual','joint_raw'}
            assert source_width==a.width_multiplier and source_mode==a.coordinate_mode=='legacy'
            missing,unexpected=m.load_state_dict(ck['model'],strict=False)
            assert set(missing)=={k for k in m.state_dict() if k.startswith('refiner.')} and not unexpected
        elif source_width!=a.width_multiplier:
            assert source_width==1 and a.width_multiplier==2 and source_mode==a.coordinate_mode=='legacy'
            from armor_native.widen import widen_quad_state
            m.load_state_dict(widen_quad_state(ck['model'],m))
        elif source_mode==a.coordinate_mode:m.load_state_dict(ck['model'])
        else:
            assert source_mode=='legacy' and a.coordinate_mode in {'continuous_gaussian','dense_votes'}
            missing,unexpected=m.load_state_dict(ck['model'],strict=False)
            head='coordinate_head' if a.coordinate_mode=='continuous_gaussian' else 'dense_head'
            assert set(missing)=={head+'.weight',head+'.bias'} and not unexpected
            if a.coordinate_mode=='continuous_gaussian':m.coordinate_head.load_state_dict(m.backbone.heatmap.state_dict())
    else:
        assert a.width_multiplier==1
        m.backbone.load_state_dict(ck['model'])
        if a.coordinate_mode=='continuous_gaussian':m.coordinate_head.load_state_dict(m.backbone.heatmap.state_dict())
    if a.refinement_mode in {'joint_residual','joint_raw'}:
        for name,param in m.named_parameters():param.requires_grad_(name.startswith('refiner.'))
    # Reset data-loader RNG independently from added parameter initialization.
    torch.manual_seed(seed);template=ck['geometry'];rows=read(root/'train.jsonl')
    assert all(o.get('lamp_evidence',{}).get('rule_version',RULE['version'])==a.lamp_rule for r in rows for o in r['objects']), 'Training visibility rule mismatch'
    if a.require_level_body:assert all(r.get('requested_level_target_verified') and abs(r['actual_target_pitch_deg'])<.01 and abs(r['actual_target_roll_deg'])<.01 and abs(r['actual_camera_roll_deg'])<.01 for r in rows)
    dataset=Samples(rows,a.steps,template,seed,a.anchor_mode,a.sample_offset,a.sampling)
    cfg={**vars(a),'joint':bool(a.joint),'seed':seed,'initial_sha256':hashlib.sha256(Path(a.initial).read_bytes()).hexdigest(),'train_manifest_sha256':hashlib.sha256((root/'train.jsonl').read_bytes()).hexdigest(),'batch_objects':4,'batch_native_patches':16,'native_rgb_resized':False,'train_only_geometry_loss':'central-difference IPPE sensitivity at labeled corners, no GT online','primary_absolute_goals':{'corner_p95_px':.236,'depth_p95_mm':74.3,'xy_p95_mm':10.6},'selection':'fixed final, DEV assessment only; previously opened static data diagnostic only','continuous_decoder':('9x9 expectation from separate continuous Gaussian head; sigma1.25 full-map KL' if a.coordinate_mode=='continuous_gaussian' else ('7x7 local softmax of coherent endpoint vectors' if a.coordinate_mode=='dense_votes' else '7x7 local softmax plus learned offsets')),'data_order_reproducibility':'same seed+manifest+offset+sampler config only; differing samplers intentionally draw different examples','trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),'frozen_base':a.refinement_mode in {'joint_residual','joint_raw'},'expected_range_draw_probabilities':dataset.selector.range_probabilities(),'expected_height_probabilities_by_range':dataset.selector.probabilities() if a.sampling=='distance_height' else None}
    (out/'config.json').write_text(json.dumps(cfg,indent=2));opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4);scaler=torch.amp.GradScaler('cuda',init_scale=128)
    loader=DataLoader(dataset,batch_size=None,num_workers=2,pin_memory=True,prefetch_factor=2);start=time.monotonic();skipped=0;draws=collections.Counter();height_draws=collections.Counter()
    def save(step):
        torch.save({'head':'native_quad_geometry','model':m.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'updates':step,'geometry':template,'config':cfg,'inference_config':{'proposal_threshold':.1,'minimum_corner_probability':.1,'duplicate_max_px':3,'observability_rule':dict(RULES[a.lamp_rule])}},out/f'checkpoint-{step:06d}.pt')
    with (out/'train.jsonl').open('x') as log:
        for i,b in enumerate(loader):
            if time.monotonic()-start>a.seconds:save(i);raise TimeoutError('bounded paired training')
            draws.update(map(str,b['range_bin'].tolist()));height_draws.update(str((int(d),int(h))) for d,h in zip(b['range_bin'].tolist(),b['height_region'].tolist()));b={k:v.cuda(non_blocking=True) for k,v in b.items()};opt.zero_grad(set_to_none=True)
            lr=1e-5+.5*9e-5*(1+math.cos(math.pi*i/max(1,a.steps-1)))
            for g in opt.param_groups:g['lr']=lr
            with torch.autocast('cuda',dtype=torch.float16):pred=m(b['image'],b['anchors'])
            hmoff={'heatmap':pred['heatmap'],'offset':pred['offset']};base,terms=precision_loss(hmoff,b)
            coordinate_distribution=gaussian_coordinate_loss(pred['coordinate_heatmap'],b['corners']) if a.coordinate_mode=='continuous_gaussian' else base.new_zeros(())
            dense_supervision=dense_vote_loss(pred['vote_vectors'],b['corners']) if a.coordinate_mode=='dense_votes' else base.new_zeros(())
            terms['dense_vote_loss']=float(dense_supervision.detach())
            terms['coordinate_distribution_kl']=float(coordinate_distribution.detach())
            e=pred['corners'].float()-b['corners'];coord=F.smooth_l1_loss(e/.236,torch.zeros_like(e),beta=1)
            # This approximation is used only within its local regime.
            good=(e.detach().abs().amax((1,2))<4)
            d=torch.bmm(b['jacobian'],e.reshape(-1,8,1)).squeeze(-1)
            norm=torch.tensor([.0106,.0106,.0743],device=e.device)
            geo=(F.smooth_l1_loss(d/norm,torch.zeros_like(d),reduction='none',beta=1).mean(1)*good).sum()/good.sum().clamp_min(1)
            shape=F.smooth_l1_loss((e-e.mean(1,keepdim=True))/.236,torch.zeros_like(e),beta=1)
            loss=base+coordinate_distribution+a.dense_vote_weight*dense_supervision+(.1*a.precision_weight)*coord+(a.geometry_weight*a.precision_weight)*(geo+.25*shape)
            assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt);gn=torch.nn.utils.clip_grad_norm_(m.parameters(),5);scale=scaler.get_scale();scaler.step(opt);scaler.update();skipped+=int(scaler.get_scale()<scale)
            row={'update':i+1,'loss':float(loss.detach()),'coordinate':float(coord.detach()),'geometry':float(geo.detach()),'shape':float(shape.detach()),'local_geometry_objects':int(good.sum()),'elapsed_s':time.monotonic()-start,'skipped':skipped,'lr':lr,'gradient_norm_before_clip':float(gn),'gradient_clip_factor':min(1.0,5.0/(float(gn)+1e-6)),**terms};log.write(json.dumps(row)+'\n');log.flush()
            if (i+1)%500==0:save(i+1);print(json.dumps(row),flush=True)
    if a.steps%500:save(a.steps)
    (out/'result.json').write_text(json.dumps({'complete':True,'updates':a.steps,'optimizer_updates':a.steps-skipped,'seconds':time.monotonic()-start,'sealed_used':False,'actual_range_draw_counts':dict(draws),'actual_range_height_draw_counts':dict(height_draws)}))
if __name__=='__main__':main()
