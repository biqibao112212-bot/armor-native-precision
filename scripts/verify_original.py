"""Maintainer parity check against the frozen parent implementation (CUDA)."""
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
from armor_native.proposals import NativeProposer
from armor_native.infer import NativeLocalizer
from armor_native.geometry import crop_native
from armor_native.data import rgb

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source-root',required=True);ap.add_argument('--image',required=True);ap.add_argument('--original-proposal',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();sys.path.insert(0,str(Path(a.source_root).resolve()/'training/armor_native'));from model import NativeArmor;from roi_precision import RoiPrecision as OriginalRoi;from evaluate import infer as original_infer
 torch.set_num_threads(4);image=rgb(a.image);new=NativeProposer('weights/proposal_yolo3200.pt','configs/proposal_yolo11n_pose.json','cuda');old=NativeArmor('yolo').cuda();old.load_state_dict(torch.load(a.original_proposal,map_location='cpu',weights_only=False)['model']);old.eval();pp=new(image);qq,_=original_infer(old,image);qq=[p for p in qq if p['score']>=.5];assert len(pp)==len(qq)>0;pd=float(np.max(np.abs(np.array([p['center'] for p in pp])-np.array([p['center'] for p in qq]))));assert pd<1e-6
 local=NativeLocalizer('weights/native_roi_spatial.pt','cuda');reference=OriginalRoi('spatial').cuda();reference.load_state_dict(torch.load('weights/native_roi_training_002000.pt',map_location='cpu',weights_only=False)['model']);reference.eval();crops=[crop_native(image,p['center']) for p in pp];batch=torch.stack([torch.from_numpy(q[0].transpose(2,0,1).copy()).float()/255 for q in crops]).cuda()
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):expected=reference.points(reference(batch)).cpu().numpy()
 predictions=local(image,pp,json.loads(Path('configs/camera_simulator.json').read_text())['K']);actual=np.array([p['corners'] for p in predictions]);expected=np.array([c+q[1] for c,q in zip(expected,crops)]);rd=float(np.max(np.abs(actual-expected)));assert rd<1e-6
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);(out/'prediction.json').write_text(json.dumps(predictions,indent=2)+'\n');result={'proposal_center_max_delta_px':pd,'roi_corner_max_delta_px':rd,'objects':len(pp),'weights_loaded_strictly':True,'original_inference_amp_parity':True};(out/'parity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
