"""Analyze precision in the declared height scope, preserving all denominators."""
import argparse, ast, collections, hashlib, json
from pathlib import Path
import numpy as np
from evaluate_broad import summarize

METRICS=("corner_rms_px","depth_mm","xy_mm")
LIMITS=dict(zip(METRICS,(.236,74.3,10.6)))
def identity(r):return r["session"],r["frame_seq"],r["slot"]
def region(r):
    h=r["height_difference_m"]
    assert abs(h)<=.400000001
    return "typical_abs_le15cm" if abs(h)<=.15 else ("camera_above15_40cm" if h>0 else "camera_below15_40cm")
def paired(control,candidate,seed=2026102301):
    aa={identity(r):r for r in control if r["valid_pnp"]}
    bb={identity(r):r for r in candidate if r["valid_pnp"]}
    ids=sorted(aa.keys()&bb.keys())
    sessions=sorted({k[0] for k in ids})
    result={"common_objects":len(ids),"sessions":len(sessions),"control":summarize([aa[k] for k in ids]),"candidate":summarize([bb[k] for k in ids])}
    if not ids:return result
    rng=np.random.default_rng(seed);indices={s:np.array([i for i,k in enumerate(ids) if k[0]==s]) for s in sessions}
    deltas={k:[] for k in METRICS}
    values={k:(np.array([aa[i][k] for i in ids]),np.array([bb[i][k] for i in ids])) for k in METRICS}
    for _ in range(500):
        ii=np.concatenate([indices[s] for s in rng.choice(sessions,len(sessions),replace=True)])
        for k,(a,b) in values.items():deltas[k].append(float(np.quantile(b[ii],.95)-np.quantile(a[ii],.95)))
    result["candidate_minus_control_P95_interval95"]={k:np.quantile(v,[.025,.975]).tolist() for k,v in deltas.items()}
    result["uncertainty_scope"]="DEV whole-session resampling only; no training seed replication or fresh confirmation"
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",required=True);ap.add_argument("--output",required=True)
    a=ap.parse_args();n=Path(a.root);trial=n/"height40-dev";output=Path(a.output)
    assert not output.exists()
    plan=json.loads((trial/"plan.json").read_text())
    complete=json.loads((trial/"result.json").read_text());assert complete["complete"] and complete["frames"]==1536
    rows=[json.loads(l) for l in (trial/"records.jsonl").open()]
    methods=["height40_initial","height40_control","height40_typical"]
    grouped={m:[r for r in rows if r["method"]==m] for m in methods}
    assert all(len(v)==1064 for v in grouped.values())
    assert all(set(map(identity,grouped[m]))==set(map(identity,grouped[methods[0]])) for m in methods)
    primary={m:[r for r in rr if 3<=r["range_m"]<=5] for m,rr in grouped.items()}
    assert all(len(v)==325 for v in primary.values())
    regions=["typical_abs_le15cm","camera_above15_40cm","camera_below15_40cm"]
    reports={}
    for m,rr in grouped.items():
        reports[m]={"all":summarize(rr),"primary_3_5":summarize(primary[m]),
                    "primary_by_height":{s:summarize([r for r in primary[m] if region(r)==s]) for s in regions},
                    "distance_by_height":{str(d):{s:summarize([r for r in rr if r["range_bin"]==d and region(r)==s]) for s in regions} for d in range(8)}}
        report=reports[m]["primary_3_5"]
        reports[m]["development_goal_flags"]={k:report[k].get("p95",float("inf"))<=limit for k,limit in LIMITS.items()}
        reports[m]["development_goal_flags"]["coverage_reference"]=report["coverage"]>=812/817
    sampling={}
    for arm in ["height40-distance","height40-typical"]:
        cfg=json.loads((n/arm/"config.json").read_text());result=json.loads((n/arm/"result.json").read_text())
        assert result["complete"] and result["optimizer_updates"]==8000
        counts={ast.literal_eval(k):v for k,v in result["actual_range_height_draw_counts"].items()}
        assert sum(counts.values())==32000 and all(h in (0,1,2) for d,h in counts)
        sampling[arm]={"sampling":cfg["sampling"],"effective_updates":result["optimizer_updates"],
            "initial_sha256":cfg["initial_sha256"],"train_manifest_sha256":cfg["train_manifest_sha256"],
            "observed_height_proportions":{str(d):{str(h):counts.get((d,h),0)/sum(v for (dd,hh),v in counts.items() if dd==d) for h in (0,1,2)} for d in range(8)}}
    assert len({v["initial_sha256"] for v in sampling.values()})==len({v["train_manifest_sha256"] for v in sampling.values()})==1
    pairings={}
    for control,candidate in [("height40_control","height40_typical"),("height40_initial","height40_control"),("height40_initial","height40_typical")]:
        pairings[candidate+"_minus_"+control]={"all_primary":paired(primary[control],primary[candidate]),
            "by_height":{s:paired([r for r in primary[control] if region(r)==s],[r for r in primary[candidate] if region(r)==s]) for s in regions}}
    result={"complete":True,"scope":"Same full DEV frames and user-declared+-40cm height population; not fresh confirmation, not prediction or NX evaluation",
            "height_definition":"camera optical center worldZ minus physical armor center worldZ",
            "sampling_order_note":"Arms intentionally have different sample sequences because the sampling distribution is the intervention",
            "frame_count":1536,"reports":reports,"sampling":sampling,"paired":pairings,
            "records_sha256":hashlib.sha256((trial/"records.jsonl").read_bytes()).hexdigest(),"evaluation_plan":plan}
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps({m:{"primary":{k:v["primary_3_5"][k].get("p95") for k in METRICS},"coverage":v["primary_3_5"]["coverage"],"goal_flags":v["development_goal_flags"]} for m,v in reports.items()}))
if __name__=="__main__":main()
