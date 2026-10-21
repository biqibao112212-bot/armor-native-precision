"""Check exported dynamic batch corner coordinates against FP32 PyTorch."""
import argparse,json
from pathlib import Path
import numpy as np,torch,onnx,onnxruntime as ort
from armor_native.model import RoiPrecision
from armor_native.geometry import crop_native
from armor_native.data import rgb

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--image',required=True);ap.add_argument('--centers',required=True,help='JSON predictions emitted by verify_original.py');ap.add_argument('--output',required=True);a=ap.parse_args();torch.set_num_threads(4);model=RoiPrecision();model.load_state_dict(torch.load('weights/native_roi_spatial.pt',map_location='cpu',weights_only=True)['model']);model.eval();onnx.checker.check_model(onnx.load('weights/native_roi_spatial.onnx'));session=ort.InferenceSession('weights/native_roi_spatial.onnx',providers=['CPUExecutionProvider']);image=rgb(a.image);center=np.array(json.loads(Path(a.centers).read_text())[0]['proposal_center']);errors={}
 for n in [1,2,4]:
  patches=[crop_native(image,center+[j-1,j%2])[0] for j in range(n)];x=np.stack([q.transpose(2,0,1) for q in patches]).astype(np.float32)/255
  with torch.no_grad():expected=model.points(model(torch.from_numpy(x))).numpy()
  actual=session.run(None,{'rgb':x})[0];assert actual.shape==(n,4,2);delta=float(np.max(np.abs(expected-actual)));assert delta<.003,(n,delta);errors[str(n)]=delta
 result={'onnx_checker':True,'batch_max_corner_delta_px':errors,'provider':'CPUExecutionProvider','opset':17,'fixed_input_spatial_shape':[3,128,128],'dynamic_batch_verified':[1,2,4],'not_tensorRT_or_NX_validation':True};Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
