"""Image-only broad native candidate inference; no GT crop or range input."""
import argparse,json
from pathlib import Path
import torch
from armor_native.broad import BroadNativeLocalizer
from armor_native.proposals import NativeProposer
from armor_native.data import rgb

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--image',required=True);ap.add_argument('--weights',required=True);ap.add_argument('--camera',default='configs/camera_simulator.json');ap.add_argument('--proposal-weights',default='weights/proposal_yolo3200.pt');ap.add_argument('--proposal-config',default='configs/proposal_yolo11n_pose.json');ap.add_argument('--device',default='cuda');ap.add_argument('--proposal-mode',choices=['tiles448','full_native'],default='full_native');ap.add_argument('--output',required=True);a=ap.parse_args();torch.set_num_threads(4);im=rgb(a.image);K=json.loads(Path(a.camera).read_text())['K'];proposer=NativeProposer(a.proposal_weights,a.proposal_config,a.device,mode=a.proposal_mode);pp=proposer(im,include_geometry=True);result=BroadNativeLocalizer(a.weights,a.device)(im,pp,K)
 with Path(a.output).open('x') as f:json.dump({'input_rgb_resized':False,'proposals_from_image_only':True,'predictions':result},f,indent=2)
 print(json.dumps({'objects':len(result),'output':a.output}))
if __name__=='__main__':main()
