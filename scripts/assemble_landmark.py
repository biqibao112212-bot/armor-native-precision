"""Assemble immutable branches after fixed training; no inference on test data."""
import argparse,hashlib,json
from pathlib import Path
import torch
from armor_native.observability import RULE

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--landmark',required=True);ap.add_argument('--report',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();out=Path(a.output);assert not out.exists();new=torch.load(a.landmark,map_location='cpu',weights_only=False);old=torch.load(a.report,map_location='cpu',weights_only=False);assert new['head']=='native_landmark128' and new['updates']==new['config']['steps'];cfg={'proposal_threshold':.25,'minimum_corner_probability':.1,'report_agreement_max_px':1.5,'RGB_resized':False,'roll':'camera and target worldroll0','visibility_rule':RULE,'route':'four native128 patches at image-based coarse corners; report128 used only when both image-supported estimates agree within1.5px on every corner'};ck={'head':'native_landmark_fusion','model':new['model'],'report_model':old['model'],'geometry':new['geometry'],'updates':new['updates'],'config':new['config'],'inference_config':cfg,'source_sha256':{k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in [('landmark',a.landmark),('report',a.report)]}};torch.save(ck,out);print(json.dumps({'output':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'inference_config':cfg}),flush=True)
if __name__=='__main__':main()
