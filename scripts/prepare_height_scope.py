"""Build TRAIN/DEV manifests for a user-specified height scope, without moving RGB."""
import argparse, collections, hashlib, json, time
from pathlib import Path
from armor_native.height_scope import HEIGHT_LIMIT_M, height_region, HEIGHT_NAMES

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True);ap.add_argument("--output",required=True)
    a=ap.parse_args();n=Path(a.root);out=Path(a.output);out.mkdir(exist_ok=False)
    start=time.monotonic();summary={};sessions={}
    sources={"train":n/"retrain-anchors-cyan/train.jsonl","dev":n/"observable-cyan/dev.jsonl"}
    for split,path in sources.items():
        counts=collections.Counter();frames=0;kept=0;removed=0;identities=set();primary=0
        with (out/(split+".jsonl")).open("x") as f:
            for line in path.open():
                if time.monotonic()-start>90:raise TimeoutError("height scope preparation")
                row=json.loads(line);assert row["split"]==split
                assert row["requested_level_target_verified"] and abs(row["actual_target_pitch_deg"])<.01 and abs(row["actual_target_roll_deg"])<.01 and abs(row["actual_camera_roll_deg"])<.01
                include=[];exclude=[]
                for obj in row["objects"]:
                    if abs(obj["height_difference_m"])>HEIGHT_LIMIT_M+1e-9:
                        exclude.append(obj);removed+=1;continue
                    include.append(obj);kept+=1;primary+=int(3<=obj["range_m"]<=5)
                    counts[(obj["range_bin"],height_region(obj["height_difference_m"]),row["light"],row["mode"])]+=1
                row["objects"]=include
                row["height_scope_excluded_objects"]=exclude
                row["height_scope_definition"]="camera optical center world Z minus armor physical center world Z; abs<=0.4m"
                f.write(json.dumps(row)+"\n");frames+=1;identities.add(row["session"])
        sessions[split]=identities
        summary[split]={"frames":frames,"sessions":len(identities),"objects":kept,"primary_3_5_objects":primary,"out_of_height_scope_objects":removed,"source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"manifest_sha256":hashlib.sha256((out/(split+".jsonl")).read_bytes()).hexdigest(),"cells":[{"range_bin":k[0],"height_region":HEIGHT_NAMES[k[1]],"light":k[2],"mode":k[3],"objects":v} for k,v in sorted(counts.items())]}
    assert not sessions["train"]&sessions["dev"]
    card={"complete":True,"seconds":time.monotonic()-start,"user_scope":"signed vertical camera-to-armor height difference -40cm to+40cm; typical near level/about10cm","typical_sampling":"70%abs<=15cm,15%camera15-40cmabove,15%camera15-40cmbelow, within each distance","data_changes":"metadata manifests only; RGB, labels, weights and original results preserved in place on server","heldout_status":"DEV remains development; same source session split. No new independent confirmation.","no_local_backup":True,"splits":summary}
    (out/"scope.json").write_text(json.dumps(card,indent=2))
    print(json.dumps({k:{m:v[m] for m in ["frames","sessions","objects","primary_3_5_objects","out_of_height_scope_objects"]} for k,v in summary.items()}))
if __name__=="__main__":main()
