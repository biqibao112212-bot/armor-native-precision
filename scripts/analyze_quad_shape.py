"""Offline error decomposition; GT is used only to diagnose saved image outputs."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from armor_native.matching import match_centers

def stats(values):
    a=np.asarray(values,dtype=float)
    return {"n":len(a),"mean":float(a.mean()),"median":float(np.median(a)),
            "p95":float(np.quantile(a,.95)),"max":float(a.max())} if len(a) else {"n":0}

def decomposition(want,got):
    error=got-want
    translation=np.broadcast_to(error.mean(0),error.shape)
    relative=error-translation
    centered=want-want.mean(0)
    rotated=centered[:,[1,0]]*np.array([-1.,1.])
    denominator=np.sum(centered**2)
    scale=np.sum(relative*centered)/denominator*centered
    rotation=np.sum(relative*rotated)/denominator*rotated
    residual=relative-scale-rotation
    parts={"translation":translation,"scale":scale,"rotation":rotation,"residual":residual}
    energies={k:float(np.mean(v**2)) for k,v in parts.items()}
    total=float(np.mean(error**2))
    assert abs(sum(energies.values())-total)<1e-8*max(1,total)
    along=[];normal=[];length=[]
    for bottom,top in [(0,1),(3,2)]:
        unit=want[top]-want[bottom];unit=unit/np.linalg.norm(unit)
        perpendicular=unit[[1,0]]*np.array([-1.,1.])
        along.extend((error[[bottom,top]]@unit).tolist())
        normal.extend((error[[bottom,top]]@perpendicular).tolist())
        length.append(float(np.linalg.norm(got[top]-got[bottom])-np.linalg.norm(want[top]-want[bottom])))
    def width(x):return np.linalg.norm(x[[2,3]].mean(0)-x[[0,1]].mean(0))
    return {"corner_rms_px":total**.5,"energy_components_px2":energies,
            "relative_shape_rms_px":float(np.sqrt(np.mean(relative**2))),
            "along_lamp_rms_px":float(np.sqrt(np.mean(np.array(along)**2))),
            "normal_to_lamp_rms_px":float(np.sqrt(np.mean(np.array(normal)**2))),
            "lamp_length_bias_px":float(np.mean(length)),"lamp_spacing_bias_px":float(width(got)-width(want)),
            "mean_error_xy_px":error.mean(0).tolist(),"error_by_corner_xy_px":error.tolist()}

def summary(rows):
    fields=["corner_rms_px","relative_shape_rms_px","along_lamp_rms_px","normal_to_lamp_rms_px","lamp_length_bias_px","lamp_spacing_bias_px"]
    out={"valid_objects":len(rows),"metrics":{k:stats([r[k] for r in rows]) for k in fields}}
    if not rows:return out
    energies={k:sum(r["energy_components_px2"][k] for r in rows) for k in rows[0]["energy_components_px2"]}
    out["total_MSE_share"]={k:v/sum(energies.values()) for k,v in energies.items()}
    out["mean_error_by_corner_xy_px"]=np.mean([r["error_by_corner_xy_px"] for r in rows],axis=0).tolist()
    out["depth_error_correlation"]={k:float(np.corrcoef([r[k] for r in rows],[r["signed_xyz_mm"][2] for r in rows])[0,1])
          for k in ["lamp_length_bias_px","lamp_spacing_bias_px"]}
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--evaluation-dir",required=True);ap.add_argument("--manifest",required=True)
    ap.add_argument("--method",required=True);ap.add_argument("--output",required=True);ap.add_argument("--max-height",type=float,default=.4)
    a=ap.parse_args();out=Path(a.output);assert not out.exists();d=Path(a.evaluation_dir);mf=Path(a.manifest)
    plan=json.loads((d/"plan.json").read_text())
    assert hashlib.sha256(mf.read_bytes()).hexdigest()==plan["manifest_sha256"]
    frames={(r["session"],r["frame_seq"]):r for r in map(json.loads,mf.open())}
    scores={(r["session"],r["frame_seq"],r["slot"]):r for r in map(json.loads,(d/"records.jsonl").open()) if r["method"]==a.method}
    rows=[];checked=0
    for pred in map(json.loads,(d/"image-only.jsonl").open()):
        key=pred["session"],pred["frame_seq"];frame=frames[key];pp=pred["methods"][a.method]
        matches=match_centers(pp,frame["objects"])
        for j,g in enumerate(frame["objects"]):
            r=scores[(*key,g["slot"])]
            assert r["detected"]==(j in matches)
            if j not in matches:continue
            p=pp[matches[j]];assert r["valid_pnp"]==(p["xyz"] is not None)
            dec=decomposition(np.asarray(g["corners"]),np.asarray(p["corners"]))
            assert abs(dec["corner_rms_px"]-r["corner_rms_px"])<1e-8
            checked+=1
            if not r["valid_pnp"] or abs(r["height_difference_m"])>a.max_height:continue
            rows.append(dict(r,**dec))
    primary=[r for r in rows if 3<=r["range_m"]<=5]
    result={"complete":True,"scope":"Offline GT decomposition of frozen image-only outputs; no correction applied, no new model or confirmation",
            "source":plan,"method":a.method,"max_height_m":a.max_height,"reconstructed_detected_objects":checked,
            "primary":summary(primary),"primary_by_light":{k:summary([r for r in primary if r["light"]==k]) for k in ["dark","normal","bright"]},
            "by_distance":{str(k):summary([r for r in rows if r["range_bin"]==k]) for k in range(8)},
            "primary_typical":summary([r for r in primary if abs(r["height_difference_m"])<=.15]),
            "rows":rows}
    out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:result[k] for k in ["reconstructed_detected_objects","primary","primary_typical"]}))
if __name__=="__main__":main()
