"""RGB + causal proposal centers -> four original-pixel corners -> calibrated IPPE."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from .model import RoiPrecision
from .geometry import crop_native,solve_geometry
from .data import rgb

class NativeLocalizer:
 def __init__(self,weights,device='cuda'):
  self.device=torch.device(device);self.model=RoiPrecision('spatial').to(self.device);self.model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True)['model']);self.model.eval()
 @torch.inference_mode()
 def __call__(self,image,proposals,K):
  if not proposals:return []
  crops=[crop_native(image,p['center']) for p in proposals];batch=torch.stack([torch.from_numpy(c[0].transpose(2,0,1).copy()).float()/255 for c in crops]).to(self.device)
  with torch.autocast('cuda',dtype=torch.float16,enabled=self.device.type=='cuda'):output=self.model(batch)
  corners=self.model.points(output).cpu().numpy();result=[]
  for p,c,(_,origin,_) in zip(proposals,corners,crops):
   c=c+origin;pose=solve_geometry(c,K);result.append({'score':p.get('score',1.),'proposal_center':p['center'],'corners':c.tolist(),'corner_order':['bottom_left','top_left','top_right','bottom_right'],'xyz_m':None if pose is None else pose['xyz_m'],'reprojection_rms_px':None if pose is None else pose['reprojection_rms_px']})
  return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--image',required=True);ap.add_argument('--weights',default='weights/native_roi_spatial.pt');ap.add_argument('--camera',default='configs/camera_simulator.json');ap.add_argument('--centers',help='JSON list of [u,v] causal detector/tracker centers');ap.add_argument('--proposal-weights',default='weights/proposal_yolo3200.pt');ap.add_argument('--proposal-config',default='configs/proposal_yolo11n_pose.json');ap.add_argument('--device',default='cuda');ap.add_argument('--output',required=True);a=ap.parse_args();torch.set_num_threads(4);image=rgb(a.image)
 if a.centers:
  centers=json.loads(Path(a.centers).read_text());proposals=[{'center':c,'score':1.} for c in centers]
 else:
  from .proposals import NativeProposer
  proposals=NativeProposer(a.proposal_weights,a.proposal_config,a.device)(image)
 K=json.loads(Path(a.camera).read_text())['K'];result=NativeLocalizer(a.weights,a.device)(image,proposals,K)
 with Path(a.output).open('x') as out:json.dump({'image':a.image,'input_rgb_resized':False,'predictions':result},out,indent=2);out.write('\n')
 print(json.dumps({'objects':len(result),'output':a.output}))
if __name__=='__main__':main()
