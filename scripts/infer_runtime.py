"""Run the calibrated #3 FP16 native-pixel runtime; engine files are target-specific."""
import argparse,json
from pathlib import Path
import numpy as np,torch,cv2
from armor_native.data import rgb
from armor_native.runtime.pipeline import RuntimePipeline
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--bundle",required=True);p.add_argument("--engines",required=True)
    p.add_argument("--native-library",required=True);p.add_argument("--image",required=True)
    p.add_argument("--intrinsics",nargs=4,type=float,required=True)
    p.add_argument("--camera-quaternion",nargs=4,type=float,required=True,metavar=("W","X","Y","Z"))
    p.add_argument("--output");p.add_argument("--repeats",type=int,default=1)
    a=p.parse_args()
    if a.repeats<1:p.error("--repeats must be positive")
    torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    runtime=RuntimePipeline(a.bundle,a.engines,a.native_library)
    memory=runtime.warmup();im=rgb(a.image);fx,fy,cx,cy=a.intrinsics
    K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
    rows=[runtime(im,K,a.camera_quaternion) for _ in range(a.repeats)];runtime.close()
    times=[r["runtime_ms"][0] for r in rows]
    result={"input_rgb_shape":list(im.shape),"rgb_resized":False,"device":torch.cuda.get_device_name(),
            "memory":memory,"timing_scope":"warm RGB host copy through geometry/gates; excludes file I/O/acquisition/queue",
            "repeats":len(rows),"mean_ms":float(np.mean(times)),"median_ms":float(np.median(times)),
            "p95_ms":float(np.percentile(times,95)),"max_ms":float(max(times)),"last_result":rows[-1]}
    text=json.dumps(result,indent=2)
    if a.output:
        with Path(a.output).open("x") as f:f.write(text+"\n")
    else:print(text)
if __name__=="__main__":main()
