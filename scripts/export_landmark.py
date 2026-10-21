"""Export landmark/report128 branches and full-native proposer, verify on TRAIN."""
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np,torch
from torch import nn
from armor_native.broad import NativeBucketPrecision
from armor_native.proposals import ProposalNetwork
from armor_native.geometry import crop_native
from armor_native.data import rgb
class Decoded(nn.Module):
 def __init__(self,m):super().__init__();self.m=m
 def forward(self,image):
  pred=self.m(image);return self.m.points(pred),pred['heatmap'].float().sigmoid().flatten(2).max(-1).values

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fine',required=True);ap.add_argument('--proposal',required=True);ap.add_argument('--training',required=True);ap.add_argument('--manifest',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();out=Path(a.output);out.mkdir(exist_ok=False);p=Path(__file__).resolve().parents[1];torch.set_num_threads(2);ck=torch.load(a.fine,map_location='cpu',weights_only=False);coarse=torch.load(a.proposal,map_location='cpu',weights_only=False);assert ck['head']=='native_landmark_fusion';shutil.copy2(a.fine,out/'number3_roll0_fine.pt');shutil.copy2(a.training,out/'number3_roll0_landmark_training.pt');torch.save({'model':coarse['model'],'head':'broad_native_proposal','updates':coarse['updates'],'mode':'full_native','original_size':[1440,1080],'padded_size':[1440,1088],'source_training_sha256':sha(a.proposal),'training_config':coarse['config']},out/'number3_roll0_proposal.pt');(out/'inference_config.json').write_text(json.dumps({'geometry':ck['geometry'],**ck['inference_config']},indent=2))
 models={}
 for key,name in [('model','landmark128'),('report_model','report128')]:
  m=NativeBucketPrecision().eval();m.load_state_dict(ck[key]);models[name]=Decoded(m).eval();torch.onnx.export(models[name],torch.zeros(4,3,128,128),out/(name+'.onnx'),input_names=['rgb_native'],output_names=['corners_px','peak_probabilities'],dynamic_axes={'rgb_native':{0:'batch'},'corners_px':{0:'batch'},'peak_probabilities':{0:'batch'}},opset_version=17)
 net=ProposalNetwork(json.loads((p/'configs/proposal_yolo11n_pose.json').read_text())).eval();net.load_state_dict(coarse['model']);torch.onnx.export(net,torch.zeros(1,3,1088,1440),out/'proposal_fullnative.onnx',input_names=['rgb_padded'],output_names=['center_logits','geometry'],opset_version=17)
 import onnx,onnxruntime as ort
 for f in out.glob('*.onnx'):onnx.checker.check_model(onnx.load(f))
 opts=ort.SessionOptions();opts.intra_op_num_threads=2;sessions={k:ort.InferenceSession(str(out/(k+'.onnx')),sess_options=opts,providers=['CPUExecutionProvider']) for k in models};rows=[json.loads(l) for l in Path(a.manifest).open()];assert all(r['split']=='train' for r in rows);checks=[]
 with torch.inference_mode():
  for band in range(8):
   r,o=next((r,o) for r in rows for o in r['objects'] if o['range_bin']==band);im=rgb(r['image_path']);patches=[crop_native(im,c,128)[0] for c in o['corners']];x=np.stack(patches).transpose(0,3,1,2).astype(np.float32)/255
   for name,model in models.items():
    pt=[v.numpy() for v in model(torch.from_numpy(x))];oo=sessions[name].run(None,{'rgb_native':x});delta=[float(np.abs(v-ref).max()) for v,ref in zip(oo,pt)];assert delta[0]<.01 and delta[1]<1e-4,(name,band,delta);checks.append({'branch':name,'source_range_bin':band,'batch':4,'max_coordinate_delta_px':delta[0],'max_probability_delta':delta[1]})
  ps=ort.InferenceSession(str(out/'proposal_fullnative.onnx'),sess_options=opts,providers=['CPUExecutionProvider']);im=rgb(rows[0]['image_path']);x=np.zeros((1,3,1088,1440),np.float32);x[0,:,:1080]=im.transpose(2,0,1)/255;pt=[v.numpy() for v in net(torch.from_numpy(x))];oo=ps.run(None,{'rgb_padded':x});delta=[float(np.abs(v-ref).max()) for v,ref in zip(oo,pt)];assert max(delta)<.02,delta
 manifest={'schema':'armor-native.landmark-roll0-assets/1','source_training':{'fine_sha256':sha(a.fine),'proposal_sha256':sha(a.proposal),'landmark_training_sha256':sha(a.training)},'input':'RGB original1440x1080; proposer pads8 rows; fine four128 integer crops; no RGB resize','fine_decode':'For four landmark patches i=0..3 select output corners_px[i,i] then add integer crop origin; use report branch only by image evidence and agreement rule in Python reference','fine_parity':checks,'coarse_parity_max_abs_delta':delta,'validation':'CPU FP32 PyTorch/ONNX Runtime on TRAIN native crops; not TensorRT/NX acceptance','files':{f.name:{'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(out.iterdir())}}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
