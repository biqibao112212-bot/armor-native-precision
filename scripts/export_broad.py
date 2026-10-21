"""Export portable fine/proposal weights and ONNX; never overwrite existing assets."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,torch
from torch import nn
from armor_native.broad import NativeBucketPrecision
from armor_native.proposals import ProposalNetwork
from armor_native.geometry import crop_native
from armor_native.data import rgb
class Decoded(nn.Module):
 def __init__(self,model):super().__init__();self.model=model
 def forward(self,image):return self.model.points(self.model(image))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fine',required=True);ap.add_argument('--proposal',required=True);ap.add_argument('--proposal-config',required=True);ap.add_argument('--output',required=True);ap.add_argument('--manifest',required=True,help='TRAIN-only rows for numerical export checks');a=ap.parse_args();out=Path(a.output);out.mkdir(exist_ok=False,parents=True);torch.set_num_threads(2);fine=torch.load(a.fine,map_location='cpu',weights_only=False);proposal=torch.load(a.proposal,map_location='cpu',weights_only=False);assert fine['head']=='native_bucket_physical';model=NativeBucketPrecision().eval();model.load_state_dict(fine['model']);net=ProposalNetwork(json.loads(Path(a.proposal_config).read_text())).eval();net.load_state_dict(proposal['model']);fpath=out/'number3_roll0_fine.pt';ppath=out/'number3_roll0_proposal.pt'
 torch.save({'model':fine['model'],'head':'native_bucket_physical','geometry':fine['geometry'],'updates':fine['updates'],'native_buckets':[128,192,256,384,512,768,1024,1536],'RGB_resized':False,'source_training_sha256':sha(a.fine)},fpath);torch.save({'model':proposal['model'],'head':'broad_native_proposal','updates':proposal['updates'],'mode':'full_native','input_size':[1440,1088],'original_size':[1440,1080],'RGB_resized':False,'source_training_sha256':sha(a.proposal)},ppath)
 # Trace192 deliberately: it executes the dynamic coordinate grid instead of the128 buffer branch.
 torch.onnx.export(Decoded(model),torch.zeros(1,3,192,192),out/'number3_roll0_fine.onnx',input_names=['rgb'],output_names=['corners_px'],dynamic_axes={'rgb':{0:'batch',2:'height',3:'width'},'corners_px':{0:'batch'}},opset_version=17)
 torch.onnx.export(net,torch.zeros(1,3,1088,1440),out/'number3_roll0_proposal.onnx',input_names=['rgb_padded'],output_names=['center_logits','geometry'],opset_version=17)
 import onnx,onnxruntime as ort
 for name in ['number3_roll0_fine.onnx','number3_roll0_proposal.onnx']:onnx.checker.check_model(onnx.load(out/name))
 opts=ort.SessionOptions();opts.intra_op_num_threads=2;session=ort.InferenceSession(str(out/'number3_roll0_fine.onnx'),sess_options=opts,providers=['CPUExecutionProvider']);ps=ort.InferenceSession(str(out/'number3_roll0_proposal.onnx'),sess_options=opts,providers=['CPUExecutionProvider']);rows=[json.loads(l) for l in Path(a.manifest).open()];checks=[]
 with torch.inference_mode():
  for size in [128,192,256,384,512,768]:
   candidates=[(r,o) for r in rows for o in r['objects'] if o['native_size']==size]
   if not candidates:candidates=[(r,o) for r in rows for o in r['objects']][:1]
   r,o=candidates[0];patch,_,_=crop_native(rgb(r['image_path']),o['center'],size);x=patch.transpose(2,0,1)[None].astype(np.float32)/255;pt=model.points(model(torch.from_numpy(x))).numpy();onnx_points=session.run(None,{'rgb':x})[0];delta=float(np.abs(pt-onnx_points).max());checks.append({'native_size':size,'max_corner_delta_px':delta});assert delta<.01,(size,delta)
  r=rows[0];im=rgb(r['image_path']);x=np.zeros((1,3,1088,1440),np.float32);x[0,:,:1080]=im.transpose(2,0,1)/255;pt=[v.numpy() for v in net(torch.from_numpy(x))];onnx_values=ps.run(None,{'rgb_padded':x});proposal_delta=[float(np.abs(v-ref).max()) for v,ref in zip(onnx_values,pt)];assert all(v<.01 for v in proposal_delta),proposal_delta
 manifest={'schema':'armor-native.roll0-assets/1','files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.iterdir())},'fine_native_pixel_parity':checks,'proposal_output_max_abs_delta':proposal_delta,'validation_device':'CPU FP32 PyTorch versus ONNX Runtime; not TensorRT/NX acceptance','source_training':{'fine_sha256':sha(a.fine),'proposal_sha256':sha(a.proposal)},'input':'original pixels only; proposal adds8 bottom rows; fine uses integer crops/padding'}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
