"""Export the ROI corner decoder; coarse detector and PnP stay separate."""
import argparse
from pathlib import Path
import torch
from torch import nn
from .model import RoiPrecision
class Decoded(nn.Module):
 def __init__(self,model):super().__init__();self.model=model
 def forward(self,image):return self.model.points(self.model(image))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--weights',default='weights/native_roi_spatial.pt');ap.add_argument('--output',default='weights/native_roi_spatial.onnx');a=ap.parse_args();torch.set_num_threads(4);model=RoiPrecision('spatial');model.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=True)['model']);model.eval();dest=Path(a.output)
 if dest.exists():raise FileExistsError(dest)
 torch.onnx.export(Decoded(model),torch.zeros(1,3,128,128),dest,input_names=['rgb'],output_names=['corners_px'],dynamic_axes={'rgb':{0:'batch'},'corners_px':{0:'batch'}},opset_version=17)
 print(str(dest))
if __name__=='__main__':main()
