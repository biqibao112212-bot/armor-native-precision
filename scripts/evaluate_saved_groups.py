"""Summarize model comparisons without treating frames as independent sessions."""
import json,argparse,collections
from pathlib import Path
import numpy as np
from armor_native.broad import statistics

def summary(rows):
    valid=[r for r in rows if r['valid_pnp']]
    return {'eligible':len(rows),'valid':len(valid),'coverage':len(valid)/len(rows) if rows else None,**{k:statistics([r[k] for r in valid]) for k in ['corner_rms_px','depth_mm','xy_mm']}}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evaluation',required=True);ap.add_argument('--output',required=True);ap.add_argument('--control',default='initial');a=ap.parse_args();root=Path(a.evaluation);rows=[json.loads(l) for l in (root/'records.jsonl').open()];methods=sorted(set(r['method'] for r in rows));result={'methods':{},'paired_bootstrap':{},'scope':'DEV exploratory comparison; whole sessions resampled, identical valid objects within each pair'}
    for m in methods:
        rr=[r for r in rows if r['method']==m];primary=[r for r in rr if 3<=r['range_m']<=5]
        result['methods'][m]={'all':summary(rr),'primary':summary(primary),'primary_typical':summary([r for r in primary if abs(r['height_difference_m'])<=.15]),'by_distance':{str(b):summary([r for r in rr if r['range_bin']==b]) for b in range(8)},'primary_by_light':{l:summary([r for r in primary if r['light']==l]) for l in ['dark','normal','bright']},'primary_by_view':{name:summary([r for r in primary if lo<=r['facing_cos']<hi]) for name,lo,hi in [('frontal',.866,1.001),('moderate',.5,.866),('oblique',.12,.5)]}}
    initial={ (r['session'],r['frame_seq'],r['slot']):r for r in rows if r['method']==a.control and r['valid_pnp'] and 3<=r['range_m']<=5}
    assert a.control in methods, (a.control,methods)
    rng=np.random.default_rng(31337)
    for m in methods:
        if m==a.control:continue
        alt={(r['session'],r['frame_seq'],r['slot']):r for r in rows if r['method']==m and r['valid_pnp'] and 3<=r['range_m']<=5};keys=sorted(initial.keys()&alt.keys());sessions=sorted(set(k[0] for k in keys));by={s:[i for i,k in enumerate(keys) if k[0]==s] for s in sessions};ci={}
        if not keys:
            result['paired_bootstrap'][m]={'objects':0,'sessions':0,'metrics':{},'reason':'no common valid objects'}
            continue
        for metric in ['corner_rms_px','depth_mm','xy_mm']:
            x=np.array([initial[k][metric] for k in keys]);y=np.array([alt[k][metric] for k in keys]);d=[]
            for _ in range(1000):
                idx=np.concatenate([by[sessions[i]] for i in rng.integers(len(sessions),size=len(sessions))]);d.append(np.quantile(y[idx],.95)-np.quantile(x[idx],.95))
            ci[metric]={'candidate_minus_initial_p95':float(np.quantile(y,.95)-np.quantile(x,.95)),'paired_session_bootstrap95':np.quantile(d,[.025,.975]).tolist()}
        result['paired_bootstrap'][m]={'objects':len(keys),'sessions':len(sessions),'metrics':ci}
    p=Path(a.output);assert not p.exists();p.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({m:r['primary'] for m,r in result['methods'].items()}))
if __name__=='__main__':main()
