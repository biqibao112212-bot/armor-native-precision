"""Freeze inference-only ablations of identical trained glyph parameters."""
import argparse,json,hashlib
from pathlib import Path
import torch

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--weights',required=True);ap.add_argument('--ema',required=True);ap.add_argument('--output',required=True);ap.add_argument('--half-pixel-evidence',action='store_true');a=ap.parse_args()
    out=Path(a.output);out.mkdir(exist_ok=False);records={}
    for name,path,mode in [('final_level',a.weights,'glyph_level'),('ema_level',a.ema,'glyph_level'),('ema_pnp',a.ema,'glyph_pnp'),('ema_direct',a.ema,'direct'),('ema_lamp_level',a.ema,'lamp_level')]:
        old=torch.load(path,map_location='cpu',weights_only=False);assert old['head']=='native_glyph_geometry'
        ck={k:v for k,v in old.items() if k not in ['optimizer','scaler']};ck['inference_config']=dict(old['inference_config'],decoder=mode)
        if a.half_pixel_evidence:
            from armor_native.observability import CYAN_PIXEL_CENTER_RULE
            ck['inference_config']['observability_rule']=dict(CYAN_PIXEL_CENTER_RULE)
        ck['variant_provenance']={'source_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'parameters_changed':False,'mode':mode,'half_pixel_evidence':a.half_pixel_evidence,'selection_data':'none; declared inference ablations only'}
        file=out/(name+'.pt');torch.save(ck,file);saved=torch.load(file,map_location='cpu',weights_only=False)
        assert all(torch.equal(value,saved['model'][key]) for key,value in old['model'].items())
        records[name]={'path':str(file),'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'mode':mode}
    (out/'manifest.json').write_text(json.dumps(records,indent=2));print(json.dumps(records))
if __name__=='__main__':main()
