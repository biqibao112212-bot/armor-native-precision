"""Build native geometry for the current machine; no model or simulator mutation."""
import argparse,subprocess,hashlib,json,shlex
from pathlib import Path
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--output")
 ap.add_argument("--eigen-include",default="/usr/include/eigen3")
 ap.add_argument("--compiler",default="c++")
 ap.add_argument("--full-pose",action="store_true",help="Compile OpenCV initialization and projection as well as TRF")
 ap.add_argument("--with-crops",action="store_true",help="Include native crop-size queries; requires --full-pose")
 a=ap.parse_args()
 if a.with_crops and not a.full_pose:ap.error("--with-crops requires --full-pose")
 import armor_native
 src=Path(armor_native.__file__).parent/"native"/("glyph_pose.cpp" if a.full_pose else "glyph_trf.cpp")
 out=Path(a.output) if a.output else src.with_suffix(".so")
 if out.exists():raise FileExistsError(f"Protected existing binary: {out}; choose a new output")
 if not (Path(a.eigen_include)/"Eigen/Dense").is_file():raise FileNotFoundError("Eigen3 headers required")
 out.parent.mkdir(parents=True,exist_ok=True)
 extra=[]
 if a.full_pose:
  for flag in ["--cflags","--libs-only-L"]:
   extra.extend(shlex.split(subprocess.check_output(["pkg-config",flag,"opencv4"],text=True)))
  extra.extend(["-lopencv_calib3d","-lopencv_core"])
 command=[a.compiler,"-O3","-std=c++17","-shared","-fPIC","-I"+a.eigen_include,str(src),*([str(src.with_name("native_crops.cpp"))] if a.with_crops else []),"-o",str(out),*extra]
 subprocess.run(command,check=True)
 print(json.dumps({"library":str(out.resolve()),"source_sha256":hashlib.sha256(src.read_bytes()).hexdigest(),"library_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),"command":command}))
if __name__=="__main__":main()
