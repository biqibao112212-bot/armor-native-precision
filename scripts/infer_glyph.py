"""Run the portable native-pixel #3 pipeline on one calibrated RGB image."""
import argparse,json
from pathlib import Path
import numpy as np
from armor_native.data import rgb
from armor_native.glyph import GlyphLocalizer
from armor_native.proposals import NativeProposer

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bundle',required=True);ap.add_argument('--image',required=True)
    ap.add_argument('--intrinsics',type=float,nargs=4,required=True,metavar=('FX','FY','CX','CY'))
    ap.add_argument('--camera-quaternion',type=float,nargs=4,required=True,metavar=('W','X','Y','Z'),
                    help='Same-exposure camera-link-to-world attitude; never target attitude')
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda');ap.add_argument('--output')
    a=ap.parse_args();root=Path(a.bundle);contract=json.loads((root/'model-contract.json').read_text())
    assert contract['schema']=='armor-native.glyph-bundle/1'
    image=rgb(a.image);assert list(image.shape)==contract['input_rgb_shape']
    q=np.asarray(a.camera_quaternion,float)
    if not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>.01:raise ValueError('Expected a finite unit camera quaternion')
    q=q/np.linalg.norm(q);fx,fy,cx,cy=a.intrinsics
    if min(fx,fy)<=0:raise ValueError('Focal lengths must be positive')
    K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
    fine=GlyphLocalizer(root/'glyph_native.pt',a.device)
    coarse=NativeProposer(root/'proposal_native.pt',root/'proposal_yolo11n_pose.json',a.device,'full_native')
    proposals=coarse(image,threshold=fine.proposal_threshold,include_geometry=True)
    result={'input_rgb_shape':list(image.shape),'rgb_resized':False,'translation_units':'m',
        'camera_attitude_input':q.tolist(),'predictions':fine(image,proposals,K,q)}
    text=json.dumps(result,indent=2)
    if a.output:
        with Path(a.output).open('x') as f:f.write(text+'\n')
    else:print(text)
if __name__=='__main__':main()
