"""Package inference tensors without optimizer state or machine-specific paths."""
import argparse,hashlib,json,shutil
from pathlib import Path
import torch

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save_checked(original,portable,path):
    torch.save(portable,path)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert set(original['model'])==set(saved['model'])
    assert all(torch.equal(v,saved['model'][k]) for k,v in original['model'].items())
    return {'file':path.name,'sha256':sha(path),'bytes':path.stat().st_size,
            'source_sha256':original['_source_sha256'],'tensors_identical':True}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--weights',required=True);ap.add_argument('--proposal',required=True)
    ap.add_argument('--proposal-config',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--source-commit',required=True);a=ap.parse_args()
    out=Path(a.output);out.mkdir(exist_ok=False)
    load=lambda p:dict(torch.load(p,map_location='cpu',weights_only=False),_source_sha256=sha(p))
    ck=load(a.weights);assert ck['head']=='native_glyph_geometry'
    fallback=Path(ck['inference_config']['fallback_weights'])
    if not fallback.is_absolute():fallback=Path(a.weights).resolve().parent/fallback
    fb=load(fallback);assert fb['head']=='native_quad_geometry'
    prop=load(a.proposal)
    def subset(d,keys):return {k:d[k] for k in keys if k in d}
    fallback_portable={**subset(fb,['head','model','geometry','inference_config']),
        'config':subset(fb['config'],['joint','coordinate_mode','width_multiplier','refinement_mode']),
        'source_training_sha256':fb['_source_sha256']}
    fine={**subset(ck,['head','model','geometry','glyph_geometry']),
        'config':subset(ck['config'],['context_dilations']),
        'inference_config':dict(ck['inference_config'],fallback_weights='quad_large_fallback.pt'),
        'source_training_sha256':ck['_source_sha256']}
    proposal={**subset(prop,['model','inference_config']),'source_training_sha256':prop['_source_sha256']}
    assets={}
    for name,original,portable in [('glyph_native.pt',ck,fine),('quad_large_fallback.pt',fb,fallback_portable),('proposal_native.pt',prop,proposal)]:
        assets[name]=save_checked(original,portable,out/name)
    shutil.copy2(a.proposal_config,out/'proposal_yolo11n_pose.json')
    assets['proposal_yolo11n_pose.json']={'file':'proposal_yolo11n_pose.json','sha256':sha(out/'proposal_yolo11n_pose.json')}
    for name in ['LICENSE','LICENSE.md','THIRD_PARTY_NOTICES.md']:
        path=Path(__file__).resolve().parents[1]/name
        if path.exists():shutil.copy2(path,out/name)
    contract={'schema':'armor-native.glyph-bundle/1','source_commit':a.source_commit,'assets':assets,
        'input_rgb_shape':[1080,1440,3],'rgb_resized':False,'coarse_mode':'full_native',
        'fine_native_buckets':[128,192,256,384,512],'larger_targets':'included native four-crop fallback',
        'target_number':3,'decoder':fine['inference_config']['decoder'],
        'required_camera_inputs':['fx','fy','cx','cy','same-exposure quaternion wxyz, camera-link to world'],
        'camera_link_axes':'x forward, y left, z up','optical_axes':'x right, y down, z forward',
        'target_assumptions':{'world_roll_deg':0,'world_pitch_deg':0,'fixed_armor_mount_pitch_deg':15},
        'corner_order':['physical_left_bottom','physical_left_top','physical_right_top','physical_right_bottom'],
        'corners':'joint-pose reprojections of physical lamp endpoints; raw16 landmarks also returned',
        'translation':'armor origin in optical camera coordinates, meters',
        'frame_evidence_rule':fine['inference_config']['observability_rule'],
        'inference_precision':'PyTorch CUDA FP16 autocast; checkpoint tensors FP32; CPU FP32 supported',
        'tensorrt_or_nx_10ms_verified':False,'real_camera_or_other_numbers_verified':False,
        'accuracy':'see separately sealed recognition confirmation; packaging does not establish accuracy'}
    (out/'model-contract.json').write_text(json.dumps(contract,indent=2))
    print(json.dumps({'output':str(out),'assets':assets},indent=2))
if __name__=='__main__':main()
