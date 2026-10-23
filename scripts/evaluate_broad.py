"""Frozen broad-domain single-frame comparison; never selects on sealed episodes.

Oracle ROI diagnosis and image-only detector results are separate. No future
prediction or physical exposure/hit-rate acceptance is inferred from this file.
"""
import argparse,json,time,sys,hashlib,collections
from pathlib import Path
import numpy as np,cv2,torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from armor_native.model import RoiPrecision
from armor_native.broad import NativeBucketPrecision,NativeGlobalPrecision,solve_physical,statistics,load_localizer
from armor_native.geometry import crop_native,solve_geometry
from armor_native.data import rgb
from armor_native.observability import lamp_evidence,RULE
from armor_native.proposals import NativeProposer

def dump(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def summarize(rows):
 valid=[r for r in rows if r.get('valid_pnp')];out={'objects':len(rows),'detected':sum(r.get('detected',False) for r in rows),'valid_pnp':len(valid),'coverage':len(valid)/max(1,len(rows)),'corner_rms_px':statistics([r['corner_rms_px'] for r in valid]),'proposed_corner_rms_px':statistics([r['corner_rms_px'] for r in rows if r.get('detected')]),'refused_or_invalid_count':sum(r.get('detected',False) and not r.get('valid_pnp',False) for r in rows),'depth_mm':statistics([r['depth_mm'] for r in valid]),'xy_mm':statistics([r['xy_mm'] for r in valid]),'detected_and_xy_under50_fraction':sum(r['xy_mm']<50 for r in valid)/max(1,len(rows))}
 if valid:out['signed_xyz_bias_mm']=np.mean([r['signed_xyz_mm'] for r in valid],axis=0).tolist()
 return out
@torch.inference_mode()
def refine(model,image,proposals,K,template,adaptive):
 if adaptive and not isinstance(model,torch.nn.Module):
  return [dict(q,center=q.get('center',q['proposal_center']),xyz=q['xyz_m']) for q in model(image,proposals,K)]
 result=[]
 for p in proposals:
  # Only image-based coarse corner extent selects a native bucket.
  size=128 if not adaptive else next((s for s in [128,192,256,384,512,768,1024,1536] if s>=p.get('extent_px',64)+40),1536)
  patch,origin,_=crop_native(image,p['center'],size);x=torch.from_numpy(patch.transpose(2,0,1).copy())[None].float().cuda()/255
  with torch.autocast('cuda',dtype=torch.float16):pred=model(x)
  corners=model.points(pred).cpu().numpy()[0]+origin;pose=solve_physical(corners,K,template) if adaptive else solve_geometry(corners,K)
  evidence=lamp_evidence(image,corners);raw_xyz=None if pose is None else pose['xyz_m']
  result.append({'image_evidence':evidence,'raw_xyz':raw_xyz,'center':p['center'],'proposal_center':p['center'],'score':p['score'],'corners':corners.tolist(),'xyz':raw_xyz if (not adaptive or evidence['accepted']) else None,'native_size':size})
 return result
@torch.inference_mode()
def proposals_with_extent(proposer,image):
 return proposer(image,include_geometry=True)

def score_frame(frame,predictions,physical):
 from armor_native.matching import match_centers
 gt=frame['objects'];pairs=match_centers(predictions,gt)
 rows=[]
 for j,g in enumerate(gt):
  r={'session':frame['session'],'frame_seq':frame['frame_seq'],'slot':g['slot'],'range_bin':g['range_bin'],'range_m':g['range_m'],'facing_cos':g['facing_cos'],'light':frame['light'],'mode':frame['mode'],'camera_roll_deg':frame['camera_roll_deg'],'target_roll_deg':frame['target_roll_deg'],'target_pitch_deg':frame['target_pitch_deg'],'height_difference_m':g['height_difference_m'],'observable':g.get('lamp_evidence',{}).get('accepted'),'detected':j in pairs,'valid_pnp':False}
  if j in pairs:
   p=predictions[pairs[j]];want=np.array(g['corners'] if physical else g['screen_corners']);r['corner_rms_px']=float(np.sqrt(np.mean((np.array(p['corners'])-want)**2)));r['image_rejected']=p.get('image_evidence',{}).get('accepted') is False
   if p['xyz'] is not None:
    delta=(np.array(p['xyz'])-g['xyz_m'])*1000;r.update(valid_pnp=True,depth_mm=float(abs(delta[2])),xy_mm=float(np.linalg.norm(delta[:2])),signed_xyz_mm=delta.tolist())
  rows.append(r)
 return rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--old',required=True);ap.add_argument('--output',required=True);ap.add_argument('--proposal-weights',required=True);ap.add_argument('--proposal-config',required=True);ap.add_argument('--proposal-mode',choices=['tiles448','full_native'],default='tiles448');ap.add_argument('--old-proposal-weights',help='preserve complete old pipeline while adapting the new proposer');ap.add_argument('--seconds',type=int,default=1500);ap.add_argument('--consumer-source',help='optional historical SZU decoder source, private assets stay outside this repository');a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);root=Path(a.data);torch.set_num_threads(4);cv2.setNumThreads(1);ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);assert ck['updates']==ck.get('config',{}).get('steps',4000);localizer_head=ck.get('head') in {'native_landmark_fusion','native_quad_geometry'};model=load_localizer(a.checkpoint) if localizer_head else (NativeGlobalPrecision() if ck.get('head')=='global_bucket_physical' else NativeBucketPrecision()).cuda().eval();
 if not localizer_head:model.load_state_dict(ck['model'])
 population_rule=ck.get('inference_config',{}).get('observability_rule',RULE);old=RoiPrecision().cuda().eval();old.load_state_dict(torch.load(a.old,map_location='cpu',weights_only=False)['model']);template=ck['geometry'];K=np.array([[1303.675283386667,0,720],[0,1303.675283386667,540],[0,0,1]])
 plan={'scope':'sealed independent simulator episodes, number3, single-frame; oracle ROI and full image evaluation separated','candidate_sha256':sha(a.checkpoint),'old_sha256':sha(a.old),'proposal_sha256':sha(a.proposal_weights),'proposal_mode':a.proposal_mode,'old_proposal_sha256':sha(a.old_proposal_weights) if a.old_proposal_weights else sha(a.proposal_weights),'szu_frozen_scalar':1.04221015588693,'sealed_manifest_sha256':sha(root/'sealed.jsonl'),'source_sha256':sha(__file__),'threshold':getattr(model,'proposal_threshold',.5),'matching':'Hungarian image center distance <=max(8px, .15*GT native extent); GT only used after inference','native_adaptive_route':({'native_landmark_fusion':'four native128 landmark crops and agreed report128','native_quad_geometry':'four native128 crops with cross-corner features and continuous coordinates'}.get(ck.get('head'),'frozen coarse image-only predicted extent+40, smallest sufficient native bucket')),'rgb_resized':False,'selection':'fixed completed final training checkpoint, no test selection','roll_strata':'requested camera and target world roll0; historical grouping keys retained, nonzero groups must be empty','unknown_slots':'excluded from confirmed-negative and recall denominators; explicit audit retained'};dump(out/'plan.json',plan)
 from train_broad import diagnostic
 train_rows=[json.loads(l) for l in (root/'train.jsonl').open()];dump(out/'train-fit-final.json',diagnostic(model,train_rows,template,'cuda',seed=1042,n=192) if train_rows and not localizer_head else {'not_applicable':'four-crop localizer uses a separate fine-anchor diagnostic; no oracle coarse corners substituted for image proposals'});
 if not localizer_head:model.eval()
 frames=[json.loads(l) for l in (root/'sealed.jsonl').open()];proposer=NativeProposer(a.proposal_weights,a.proposal_config,mode=a.proposal_mode);old_proposer=NativeProposer(a.old_proposal_weights,a.proposal_config) if a.old_proposal_weights else None;szu=None
 if a.consumer_source:
  sys.path.insert(0,a.consumer_source);from diagnose_baselines import detector,MODEL
  import onnxruntime as ort
  szu=ort.InferenceSession(str(MODEL),providers=[('CUDAExecutionProvider',{'cudnn_conv_algo_search':'HEURISTIC'}),'CPUExecutionProvider']);assert szu.get_providers()[0]=='CUDAExecutionProvider';dump(out/'szu-provenance.json',{'model_sha256':sha(MODEL),'native_tiles':9,'tile_size':640,'threshold':.5})
 started=time.monotonic();times=[]
 with (out/'image-only-predictions.jsonl').open('x') as log:
  for i,r in enumerate(frames):
   if time.monotonic()-started>a.seconds:raise TimeoutError('bounded evaluation budget')
   im=rgb(r['image_path']);torch.cuda.synchronize();t=time.perf_counter();pp=proposer(im,threshold=getattr(model,'proposal_threshold',.5),include_geometry=True);t1=time.perf_counter();new=refine(model,im,pp,K,template,True);t2=time.perf_counter();pp0=proposals_with_extent(old_proposer,im) if old_proposer else pp;t2a=time.perf_counter();legacy=refine(old,im,pp0,K,template,False);t3=time.perf_counter();methods={'broad_native':new,'old128':legacy}
   if old_proposer:methods['new_fine_old_proposals']=refine(model,im,pp0,K,template,True)
   t3a=time.perf_counter()
   if szu is not None:
    methods['szu_native']=detector(szu,im,True);methods['szu_frozen_scalar']=[dict(p,xyz=None if p['xyz'] is None else (np.array(p['xyz'])*1.04221015588693).tolist()) for p in methods['szu_native']]
   torch.cuda.synchronize();t4=time.perf_counter();lat={'coarse_ms':(t1-t)*1000,'new_fine_ms':(t2-t1)*1000,'new_total_ms':(t2-t)*1000,'old_coarse_ms':((t2a-t2) if old_proposer else (t1-t))*1000,'old_fine_ms':(t3-t2a)*1000,'old_total_ms':((t3-t2) if old_proposer else ((t1-t)+(t3-t2a)))*1000,'new_fine_old_proposals_ms':(t3a-t3)*1000,'szu_ms':(t4-t3a)*1000};times.append(lat)
   # No labels are placed in this artifact; score only after its hash is sealed.
   log.write(json.dumps({'session':r['session'],'frame_seq':r['frame_seq'],'methods':methods,'latency_ms':lat})+'\n');log.flush()
   if i%64==0:print(json.dumps({'image_only_frames':i+1,'total':len(frames),'seconds':time.monotonic()-started}),flush=True)
 dump(out/'image-prediction-seal.json',{'sha256':sha(out/'image-only-predictions.jsonl'),'frames':len(frames),'seconds':time.monotonic()-started,'target_truth_in_inference':False})
 predictions=[json.loads(l) for l in (out/'image-only-predictions.jsonl').open()];scored=collections.defaultdict(list)
 for f,p in zip(frames,predictions):
  assert (f['session'],f['frame_seq'])==(p['session'],p['frame_seq'])
  im=rgb(f['image_path'])
  for g in f['objects']:g['lamp_evidence']=lamp_evidence(im,g['corners'],rule=population_rule)
  for method,pp in p['methods'].items():scored[method].extend(score_frame(f,pp,method in ['broad_native','new_fine_old_proposals']))
 # Separate GT-centered oracle diagnosis of the fine head: every test object,
 # same integer anchors, old128 retained, no detector coverage claim.
 oracle=collections.defaultdict(list)
 for f in ([] if localizer_head else frames):
  if time.monotonic()-started>a.seconds:raise TimeoutError('bounded oracle evaluation budget')
  im=rgb(f['image_path'])
  for g in f['objects']:
   for name,net,size,physical in [('broad_native',model,g['native_size'],True),('old128',old,128,False)]:
    patch,origin,_=crop_native(im,g['center'],size);x=torch.from_numpy(patch.transpose(2,0,1).copy())[None].float().cuda()/255
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):pred=net(x)
    c=net.points(pred).cpu().numpy()[0]+origin;pose=solve_physical(c,K,template) if physical else solve_geometry(c,K);p={'center':g['center'],'corners':c.tolist(),'xyz':None if pose is None else pose['xyz_m']};single={**f,'objects':[g]};oracle[name].extend(score_frame(single,[p],physical))
 def report(rows):
  return {'all':summarize(rows),'primary_3_5m':summarize([r for r in rows if 3<=r['range_m']<=5]),'by_session':{k:summarize([r for r in rows if r['session']==k]) for k in sorted(set(r['session'] for r in rows))},'by_facing':{k:summarize([r for r in rows if lo<=r['facing_cos']<hi]) for k,lo,hi in [('grazing_12_25',.12,.25),('25_50',.25,.5),('50_75',.5,.75),('75_100',.75,1.001)]},'by_range':{str(k):summarize([r for r in rows if r['range_bin']==k]) for k in sorted(set(r['range_bin'] for r in rows))},'by_light':{k:summarize([r for r in rows if r['light']==k]) for k in ['dark','normal','bright']},'by_motion':{k:summarize([r for r in rows if r['mode']==k]) for k in ['stationary','linear','spin','combined']},'by_pitch':{k:summarize([r for r in rows if lo<=abs(r['target_pitch_deg'])<hi]) for k,lo,hi in [('0-5deg',0,5),('5-15deg',5,15),('15deg+',15,90)]},'height_pitch_oblique_combo':summarize([r for r in rows if abs(r['height_difference_m'])>=.3 and abs(r['target_pitch_deg'])>=10 and r['facing_cos']<.5]),'by_camera_roll':{k:summarize([r for r in rows if lo<=abs(r['camera_roll_deg'])<hi]) for k,lo,hi in [('0-4deg',0,4),('4-8deg',4,8),('8-12deg',8,12.001)]},'roll_height_oblique_combo':summarize([r for r in rows if abs(r['camera_roll_deg'])>=8 and abs(r['height_difference_m'])>=.3 and abs(r['target_pitch_deg'])>=10]),'by_target_roll':{k:summarize([r for r in rows if lo<=abs(r['target_roll_deg'])<hi]) for k,lo,hi in [('0-4deg',0,4),('4-8deg',4,8),('8-12deg',8,12.001)]},'by_height':{k:summarize([r for r in rows if lo<=abs(r['height_difference_m'])<hi]) for k,lo,hi in [('0-.2m',0,.2),('.2-.5m',.2,.5),('.5m+',.5,10)]}}
 for scope,methods in [('image_only',scored),('oracle_roi',oracle)]:
  with (out/(scope+'-scored.jsonl')).open('x') as log:
   for name,rows in methods.items():
    for r in rows:log.write(json.dumps({'method':name,**r})+'\n')
 result={'complete':True,'frames':len(frames),'single_frame_only':True,'oracle_scope':'not run for four-crop localizers; image proposals are scored directly' if localizer_head else 'GT-centered native ROI','future_prediction_accepted':False,'candidate_counts':{k:{'predictions_per_frame':statistics([len(p['methods'][k]) for p in predictions]),'total_predictions':sum(len(p['methods'][k]) for p in predictions),'accepted_total_predictions':sum(q.get('xyz') is not None for p in predictions for q in p['methods'][k]),'refused_or_invalid_total':sum(q.get('xyz') is None for p in predictions for q in p['methods'][k]),'outside_filtered_labels':sum(len(p['methods'][k]) for p in predictions)-sum(r['detected'] for r in v),'outside_is_not_confirmed_false_positive':True} for k,v in scored.items()},'observability_rule':population_rule,'eligibility_counts':{'two_lamp_supported':sum(g['lamp_evidence']['accepted'] for f in frames for g in f['objects']),'unsupported':sum(not g['lamp_evidence']['accepted'] for f in frames for g in f['objects'])},'image_only_observable':{k:report([r for r in v if r['observable']]) for k,v in scored.items()},'image_only_unsupported':{k:report([r for r in v if not r['observable']]) for k,v in scored.items()},'image_only':{k:report(v) for k,v in scored.items()},'oracle_roi':{k:report(v) for k,v in oracle.items()},'latency_ms':{k:statistics([r[k] for r in times]) for k in times[0]},'seconds':time.monotonic()-started};dump(out/'result.json',result);print(json.dumps({'complete':True,'frames':len(frames),'output':str(out)}),flush=True)
if __name__=='__main__':main()
