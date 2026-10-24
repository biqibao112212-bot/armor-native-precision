"""Stream current native detections through the covariance-corrected 11-state predictor."""
import argparse,json,math,subprocess,sys
from pathlib import Path

def vector(v,n,name):
    if len(v)!=n or not all(math.isfinite(float(x)) for x in v):
        raise ValueError(f"{name}: expected {n} finite values")
    return [float(x) for x in v]

def observation_line(frame):
    session=str(frame.get("session","default"))
    if any(x in session for x in '\t\n\r"/\\'):raise ValueError("session contains unsupported characters")
    timestamp=int(frame["timestamp_ns"]);seq=int(frame["frame_seq"])
    if timestamp<=0 or seq<0:raise ValueError("invalid exposure identity")
    camera=vector(frame["camera_position_world"],3,"camera position")
    cq=vector(frame["camera_quaternion_wxyz"],4,"camera quaternion")
    gimbal=vector(frame["gimbal_position_world"],3,"gimbal position")
    gq=vector(frame["gimbal_quaternion_wxyz"],4,"gimbal quaternion")
    for q in (cq,gq):
        if math.sqrt(sum(x*x for x in q))<=0.5:raise ValueError("degenerate pose quaternion")
    detections=frame["detections"];payload=[]
    for det in detections:
        if det.get("xyz") is None:continue
        corners=det["corners"]
        if len(corners)!=4:raise ValueError("expected LB, LT, RT, RB corners")
        for point in corners:payload.extend(vector(point,2,"corner"))
        payload.extend(vector(det["xyz"],3,"optical XYZ"))
    fields=[session,"unspecified",seq,timestamp,"broad_native",len(payload)//11,*camera,*cq,*gimbal,*gq,*payload]
    return "\t".join(str(v) for v in fields)+"\n"

def forecast(state,gimbal,horizons_ms):
    if state is None:return {}
    x=vector(state,11,"state");g=vector(gimbal,3,"gimbal position");out={}
    for h in horizons_ms:
        t=h*.001;points=[]
        for index in range(4):
            angle=x[6]+x[7]*t+index*math.pi/2
            alternating=index%2
            radius=x[8]+alternating*x[9]
            points.append([x[0]+x[1]*t-radius*math.cos(angle)+g[0],
                           x[2]+x[3]*t-radius*math.sin(angle)+g[1],
                           x[4]+x[5]*t+alternating*x[10]+g[2]])
        out[str(h)]=points
    return out

class Predictor:
    """Single sequential stream worker; use separate instances for concurrent inputs."""
    def __init__(self,binary):
        self.process=subprocess.Popen([str(binary),"-","-"],stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE,stderr=sys.stderr,text=True,bufsize=1)
        self.last={}
    def update(self,frame,horizons_ms=(0,100,200,300,500)):
        session=str(frame.get("session","default"));timestamp=int(frame["timestamp_ns"])
        if timestamp<=self.last.get(session,0):raise ValueError("exposure timestamps must increase per session")
        row=observation_line(frame)
        self.process.stdin.write(row);self.process.stdin.flush()
        response=self.process.stdout.readline()
        if not response:raise RuntimeError(f"predictor closed stdout (exit={self.process.poll()})")
        result=json.loads(response)
        assert result["frame_seq"]==int(frame["frame_seq"]) and result["timestamp_ns"]==timestamp
        self.last[session]=timestamp
        result["forecasts_world_m"]=forecast(result["state_vector"] if result["state"] in ("tracking","temp_lost") else None,frame["gimbal_position_world"],horizons_ms)
        return result
    def close(self):
        self.process.stdin.close()
        try:code=self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:self.process.terminate();raise
        if code:raise RuntimeError(f"predictor exited {code}")

def main():
    p=argparse.ArgumentParser();p.add_argument("--binary",default=str(Path(__file__).parent/"build/armor-predict"))
    p.add_argument("--input",default="-");p.add_argument("--output",default="-")
    p.add_argument("--horizons-ms",type=int,nargs="+",default=[0,100,200,300,500]);args=p.parse_args()
    if any(h<0 or h>500 for h in args.horizons_ms):p.error("this release supports horizons 0..500 ms")
    source=sys.stdin if args.input=="-" else open(args.input)
    dest=sys.stdout if args.output=="-" else open(args.output,"x")
    predictor=Predictor(args.binary)
    try:
        for line in source:
            if line.strip():
                dest.write(json.dumps(predictor.update(json.loads(line),args.horizons_ms))+"\n");dest.flush()
    finally:
        predictor.close()
        if source is not sys.stdin:source.close()
        if dest is not sys.stdout:dest.close()
if __name__=="__main__":main()
