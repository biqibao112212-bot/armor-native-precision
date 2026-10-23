"""Export and build fixed-weight TensorRT FP16 graphs; isolated engineering trial."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np,torch
from armor_native.proposals import NativeProposer
from armor_native.quad import QuadLocalizer
from export_quad import DecodedQuad
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();n=Path(a.root);out=Path(a.output);out.mkdir(exist_ok=False);repo=Path(__file__).resolve().parents[1]
 torch.set_num_threads(3);start=time.monotonic()
 pk=n/'proposal-frame-statistics.pt';fk=n/'capacity-extra-control/checkpoint-004000.pt'
 plan={'scope':'Fixed current weights, native RGB. TensorRT FP16 engineering comparison, not NX or accuracy acceptance. INT8 is not enabled.','seconds_per_build_limit':480,'weights_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [pk,fk]},'source_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()}
 (out/'plan.json').write_text(json.dumps(plan,indent=2));(out/'job.json').write_text(json.dumps({'pid':os.getpid(),'started_unix':time.time(),'argv':sys.argv},indent=2))
 p=NativeProposer(str(pk),repo/'configs/proposal_yolo11n_pose.json',device='cpu',mode='full_native')
 torch.onnx.export(p.model,torch.zeros(1,3,1088,1440),out/'coarse.onnx',input_names=['rgb_native'],output_names=['center_logits','geometry'],opset_version=17)
 f=QuadLocalizer(str(fk),device='cpu')
 torch.onnx.export(DecodedQuad(f.model),(torch.zeros(1,4,3,128,128),torch.zeros(1,4,2)),out/'fine.onnx',input_names=['rgb_native','proposal_anchors_px'],output_names=['corners_local_px','corner_peak_probabilities'],dynamic_axes={name:{0:'targets'} for name in ['rgb_native','proposal_anchors_px','corners_local_px','corner_peak_probabilities']},opset_version=17)
 import onnx
 for name in ['coarse','fine']:onnx.checker.check_model(onnx.load(out/(name+'.onnx')))
 for name in ['coarse','fine']:
  cmd=['/usr/src/tensorrt/bin/trtexec','--onnx='+str(out/(name+'.onnx')),'--saveEngine='+str(out/(name+'.engine')),'--fp16','--memPoolSize=workspace:1024','--builderOptimizationLevel=3','--profilingVerbosity=detailed','--skipInference']
  if name=='fine':cmd+=['--minShapes=rgb_native:1x4x3x128x128,proposal_anchors_px:1x4x2','--optShapes=rgb_native:4x4x3x128x128,proposal_anchors_px:4x4x2','--maxShapes=rgb_native:12x4x3x128x128,proposal_anchors_px:12x4x2']
  (out/(name+'-command.json')).write_text(json.dumps(cmd));print('building '+name,flush=True)
  t=time.monotonic()
  with (out/(name+'-build.log')).open('w') as log:
   process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
   (out/'active.json').write_text(json.dumps({'stage':name,'pid':process.pid,'started_unix':time.time(),'command':cmd}))
   try:rc=process.wait(timeout=480)
   except subprocess.TimeoutExpired:process.terminate();process.wait();raise TimeoutError(name+' engine build >480s')
  if rc:raise RuntimeError(name+' build failed; inspect retained log')
  (out/(name+'-receipt.json')).write_text(json.dumps({'seconds':time.monotonic()-t,'sha256':hashlib.sha256((out/(name+'.engine')).read_bytes()).hexdigest()}))
  print('built '+name,flush=True)
 (out/'complete.json').write_text(json.dumps({'engine_build_complete':True,'numerical_validation_complete':False,'nx_acceptance':False,'seconds':time.monotonic()-start}))
if __name__=='__main__':main()
