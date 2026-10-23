"""Export fixed native-resolution runtime graphs from the portable weight bundle."""
import argparse,json,hashlib
from pathlib import Path
import torch,onnx
from armor_native.glyph import NativeGlyphPrecision
from armor_native.efficient_glyph import FastGlyphPrecision
from armor_native.efficient_proposal import FastProposalNetwork
from armor_native.proposals import use_frame_statistics
from armor_native.runtime.exporters import SmallTuple,FineTuple,CoarseTuple,repair_norm
def main():
 p=argparse.ArgumentParser();p.add_argument("--bundle",required=True);p.add_argument("--output",required=True)
 a=p.parse_args();root=Path(a.bundle);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(1)
 coarse=FastProposalNetwork().cuda().eval();coarse.load_state_dict(torch.load(root/"proposal_fast.pt",map_location="cpu",weights_only=False)["model"]);use_frame_statistics(coarse);coarse.eval()
 ck=torch.load(root/"glyph_native.pt",map_location="cpu",weights_only=False)
 small=NativeGlyphPrecision(len(ck["glyph_geometry"]["object_points"]),tuple(ck["config"]["context_dilations"])).cuda().eval();small.load_state_dict(ck["model"])
 large=FastGlyphPrecision().cuda().eval();large.load_state_dict(torch.load(root/"glyph_large.pt",map_location="cpu",weights_only=False)["model"])
 models=[("coarse",CoarseTuple(coarse),None,"auto_fp16_coarse",1,1),("fine128",SmallTuple(small),128,"auto_fp16_fine",2,16)]
 models.extend((f"fine{s}",FineTuple(large,s),s,"explicit_amp",2 if s==192 or s==256 else 1,max(1,262144//(s*s))) for s in [192,256,384,512])
 manifest={"schema":"armor-native.runtime-onnx/1","native_rgb":True,"graphs":[]}
 for name,m,size,precision,opt,maximum in models:
  shape=(1,3,size,size) if size else (1,3,1088,1440)
  names=["points","peak_logits"] if size else ["scores","centers","corners"]
  path=out/(name+".onnx")
  with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16,enabled=precision=="explicit_amp",cache_enabled=False):
   torch.onnx.export(m,torch.zeros(shape,device="cuda"),str(path),input_names=["rgb"],output_names=names,opset_version=17,dynamic_axes={k:{0:"batch"} for k in ["rgb",*names]} if size else None)
  graph=onnx.load(path)
  if precision=="explicit_amp":graph,_=repair_norm(graph);onnx.save(graph,path)
  onnx.checker.check_model(graph,full_check=True)
  manifest["graphs"].append({"name":name,"file":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"size":size,"precision":precision,"opt_batch":opt,"max_batch":maximum})
 (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
