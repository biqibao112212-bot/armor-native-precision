"""Summarize frozen recognition outputs without changing population or thresholds."""
import argparse,csv,json
from pathlib import Path
import numpy as np

METRICS=('corner_rms_px','depth_mm','xy_mm')
def stats(rr,key):
    v=np.asarray([r[key] for r in rr if r['valid_pnp']],float)
    return {} if not len(v) else dict(n=len(v),mean=float(v.mean()),median=float(np.median(v)),p95=float(np.quantile(v,.95)))

def paired(a,b,repeats=500):
    aa={(r['session'],r['frame_seq'],r['slot']):r for r in a if r['valid_pnp']}
    bb={(r['session'],r['frame_seq'],r['slot']):r for r in b if r['valid_pnp']}
    keys=sorted(aa.keys()&bb.keys());sessions=sorted({k[0] for k in keys})
    if not keys:return {'objects':0}
    groups=[np.asarray([i for i,k in enumerate(keys) if k[0]==s]) for s in sessions]
    x={m:np.asarray([aa[k][m] for k in keys]) for m in METRICS}
    y={m:np.asarray([bb[k][m] for k in keys]) for m in METRICS}
    rng=np.random.default_rng(2026102301);draws={m:[] for m in METRICS}
    for _ in range(repeats):
        ids=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        for m in METRICS:draws[m].append(float(np.quantile(x[m][ids],.95)-np.quantile(y[m][ids],.95)))
    return {'objects':len(keys),'sessions':len(sessions),'bootstrap_repeats':repeats,
        'difference':'candidate minus comparator; negative is lower error; whole sessions resampled',
        'metrics':{m:{'candidate_p95':float(np.quantile(x[m],.95)),'comparator_p95':float(np.quantile(y[m],.95)),
          'p95_difference':float(np.quantile(x[m],.95)-np.quantile(y[m],.95)),
          'bootstrap95':np.quantile(draws[m],[.025,.975]).tolist()} for m in METRICS}}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evaluation',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    root=Path(a.evaluation);result=json.loads((root/'result.json').read_text());assert result['complete']
    out=Path(a.output);out.mkdir(exist_ok=False)
    rows=list(map(json.loads,(root/'records.jsonl').open()));names=sorted({r['method'] for r in rows})
    groups={n:[r for r in rows if r['method']==n] for n in names}
    primary={n:[r for r in rr if 3<=r['range_m']<=5] for n,rr in groups.items()}
    comparisons={n:paired(primary['candidate'],rr) for n,rr in primary.items() if n!='candidate'}
    (out/'paired-primary.json').write_text(json.dumps(comparisons,indent=2))
    table=[]
    for name,rr in groups.items():
        for band in ['all','3-5m',*range(8)]:
            selected=rr if band=='all' else ([r for r in rr if 3<=r['range_m']<=5] if band=='3-5m' else [r for r in rr if r['range_bin']==band])
            row={'method':name,'range':band,'eligible':len(selected),'valid':sum(r['valid_pnp'] for r in selected)}
            for m in METRICS:
                for k,v in stats(selected,m).items():row[m+'_'+k]=v
            table.append(row)
    with (out/'by-distance.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axs=plt.subplots(2,3,figsize=(12,6.8),sharex=True,sharey='col')
    panels=[('candidate','Native 16-point + level pose','#197A8C'),('szu_native','SZU native baseline','#C97439')]
    for row,(method,label,color) in enumerate(panels):
        rr=[r for r in groups[method] if r['valid_pnp']]
        for col,(metric,unit,limit,scale) in enumerate(zip(METRICS,['Corner RMS (px)','Depth error (cm)','Optical XY error (cm)'],[.236,7.43,1.06],[1,.1,.1])):
            ax=axs[row,col];values=np.asarray([r[metric]*scale for r in rr])
            ax.scatter([r['range_m'] for r in rr],values,s=5,alpha=.2,color=color,rasterized=True,linewidths=0)
            for q,style,legend in [(.5,'-','Median'),(.95,'--','P95')]:
                xs=[];ys=[]
                for band in range(8):
                    vs=[r[metric]*scale for r in rr if r['range_bin']==band]
                    if vs:xs.append(band+.5);ys.append(np.quantile(vs,q))
                ax.plot(xs,ys,style,color=color,lw=1.8,label=legend)
            ax.plot([3,5],[limit,limit],':',color='#555555',lw=1.6,label='3-5 m target')
            ax.set_yscale('symlog',linthresh=[.05,1,.2][col]);ax.set_xlim(.25,8);ax.set_ylim(bottom=0);ax.set_ylabel(unit);ax.grid(alpha=.15)
            ax.set_title((label+'\n' if col==0 else '')+unit.split(' (')[0],fontsize=10)
            if row==1:ax.set_xlabel('True range (m), scoring only')
            if col==0:ax.legend(loc='upper right',fontsize=8)
    # Set shared limits after both rows exist; do not freeze the first method's upper limit.
    for col,(metric,scale) in enumerate(zip(METRICS,[1,.1,.1])):
        upper=max(r[metric]*scale for name,_,_ in panels for r in groups[name] if r['valid_pnp'])
        axs[0,col].set_ylim(0,upper*1.15)
    fig.suptitle('Independent recognition confirmation - accepted matches; misses retained in coverage',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.95])
    fig.savefig(out/'error-vs-distance.png',dpi=180)
    fig.savefig(out/'error-vs-distance.svg')
    plt.close(fig)
    (out/'result.json').write_text(json.dumps({'complete':True,'input_independent':result.get('independent_confirmation'), 'candidate_primary':result['methods']['candidate']['primary_3_5'],'plot_scale':'symlog to retain all tail values; thresholds apply only to 3-5m','new_selection_performed':False},indent=2))
    print(json.dumps({'complete':True,'output':str(out),'paired':comparisons}),flush=True)
if __name__=='__main__':main()
