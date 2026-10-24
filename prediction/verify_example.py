"""Check the packaged streaming predictor against the preserved accepted outputs."""
import argparse,json,math
from pathlib import Path
from predict import Predictor
def main():
 p=argparse.ArgumentParser();p.add_argument("--binary",default=str(Path(__file__).parent/"build/armor-predict"));a=p.parse_args()
 folder=Path(__file__).parent/"examples";refs={(r["session"],r["frame_seq"]):r for r in map(json.loads,(folder/"expected_states.jsonl").open())}
 worker=Predictor(a.binary);count=0;maximum=0.
 try:
  for f in map(json.loads,(folder/"observations.jsonl").open()):
   got=worker.update(f);ref=refs[(f["session"],f["frame_seq"])]
   assert got["state"]==ref["state"] and got["association_ids"]==ref["association_ids"]
   assert (got["state_vector"] is None)==(ref["state_vector"] is None)
   if ref["state_vector"] is not None:
    assert all(math.isfinite(x) for x in got["state_vector"])
    delta=max(abs(x-y) for x,y in zip(got["state_vector"],ref["state_vector"]));maximum=max(maximum,delta);assert delta<1e-10
   count+=1
 finally:worker.close()
 assert count==len(refs)
 print(json.dumps({"passed":True,"frames":count,"maximum_state_difference":maximum}))
if __name__=="__main__":main()
