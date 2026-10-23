"""Build device-specific TRT 10.x engines, preserving FP32 coordinate decoders."""
import argparse,json,time,hashlib
from pathlib import Path
import torch,tensorrt as trt
def main():
 p=argparse.ArgumentParser();p.add_argument("--onnx",required=True);p.add_argument("--output",required=True)
 p.add_argument("--workspace-gib",type=float,default=2);a=p.parse_args()
 root=Path(a.onnx);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 manifest=json.loads((root/"manifest.json").read_text())
 if trt.__version__.split(".")[0]!="10":raise RuntimeError("This builder targets TensorRT 10.x; other versions need validation")
 torch.cuda.init();torch.set_num_threads(1);logger=trt.Logger(trt.Logger.WARNING);results=[]
 for spec in manifest["graphs"]:
  path=root/spec["file"]
  if hashlib.sha256(path.read_bytes()).hexdigest()!=spec["sha256"]:raise ValueError("ONNX hash mismatch")
  builder=trt.Builder(logger);explicit=spec["precision"]=="explicit_amp"
  net=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if explicit else 0)
  parser=trt.OnnxParser(net,logger)
  if not parser.parse(path.read_bytes()):raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
  cfg=builder.create_builder_config();cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,int(a.workspace_gib*(1<<30)))
  cfg.builder_optimization_level=3;cfg.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
  if not explicit:
   cfg.set_flag(trt.BuilderFlag.FP16);cfg.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
   last_conv=max(i for i in range(net.num_layers) if net.get_layer(i).type==trt.LayerType.CONVOLUTION)
   for i in range(net.num_layers):
    layer=net.get_layer(i)
    decoder=("/model/" not in layer.name) if spec["size"] is None else i>last_conv
    floats=[j for j in range(layer.num_outputs) if layer.get_output(j).dtype in [trt.float32,trt.float16]]
    if decoder and floats and layer.type not in [trt.LayerType.CONSTANT,trt.LayerType.SHAPE]:
     layer.precision=trt.float32
     for j in floats:layer.set_output_type(j,trt.float32)
  size=spec["size"]
  if size:
   profile=builder.create_optimization_profile();profile.set_shape("rgb",(1,3,size,size),(spec["opt_batch"],3,size,size),(spec["max_batch"],3,size,size));cfg.add_optimization_profile(profile)
  start=time.monotonic();engine=builder.build_serialized_network(net,cfg)
  if engine is None:raise RuntimeError("Build failed "+spec["name"])
  ep=out/(spec["name"]+".engine");ep.write_bytes(bytes(engine))
  results.append({**spec,"engine":ep.name,"engine_sha256":hashlib.sha256(ep.read_bytes()).hexdigest(),"build_seconds":time.monotonic()-start})
  (out/"build-progress.json").write_text(json.dumps(results,indent=2))
 result={"schema":"armor-native.runtime-engines/1","device":torch.cuda.get_device_name(),"compute_capability":torch.cuda.get_device_capability(),"tensorrt":trt.__version__,"torch":torch.__version__,"graphs":results,"INT8":False}
 (out/"manifest.json").write_text(json.dumps(result,indent=2))
if __name__=="__main__":main()
