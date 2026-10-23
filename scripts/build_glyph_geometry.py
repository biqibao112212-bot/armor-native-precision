"""Select spatially spread numeral contour vertices from the public #3 asset."""
import argparse,json,collections,hashlib
from pathlib import Path
import numpy as np
from probe_lamp_visibility import glb_mesh

def glyph_points(asset,count=12):
    v,tri,material,parts=glb_mesh(asset,include_digit=True)
    faces=v[tri[material==2]];unique,inv=np.unique(np.round(faces.reshape(-1,3),8),axis=0,return_inverse=True)
    triangles=np.unique(np.sort(inv.reshape(-1,3),axis=1),axis=0)
    edge=collections.Counter(tuple(sorted((int(t[i]),int(t[(i+1)%3])))) for t in triangles for i in range(3))
    adjacency=collections.defaultdict(list)
    for (a,b),n in edge.items():
        if n==1:adjacency[a].append(b);adjacency[b].append(a)
    assert adjacency and all(len(x)==2 for x in adjacency.values())
    start=min(adjacency);loop=[start];previous=None;current=start
    while True:
        nxt=next(x for x in adjacency[current] if x!=previous)
        if nxt==start:break
        assert nxt not in loop
        loop.append(nxt);previous,current=current,nxt
    assert len(loop)==len(adjacency)
    boundary=unique[loop];before=boundary-np.roll(boundary,1,axis=0);after=np.roll(boundary,-1,axis=0)-boundary
    cos=np.sum(before*after,1)/(np.linalg.norm(before,axis=1)*np.linalg.norm(after,axis=1));angles=np.arccos(np.clip(cos,-1,1))
    selected=sorted(set([int(np.argmin(boundary[:,0])),int(np.argmax(boundary[:,0])),int(np.argmin(boundary[:,2])),int(np.argmax(boundary[:,2]))]))
    for i in np.argsort(-angles):
        if len(selected)>=count:break
        if min(np.linalg.norm(boundary[i]-boundary[selected],axis=1))>=.008:selected.append(int(i))
    while len(selected)<count:
        remaining=[i for i in range(len(boundary)) if i not in selected]
        selected.append(max(remaining,key=lambda i:min(np.linalg.norm(boundary[i]-boundary[selected],axis=1))))
    selected=sorted(selected)
    return boundary[selected],{'boundary_vertices':boundary.tolist(),'selected_boundary_indices':selected,'turn_degrees':np.rad2deg(angles[selected]).tolist(),'selection':'include horizontal/vertical extrema, then high-turn vertices separated by >=8mm; complete by farthest spatial point','parts':parts}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--asset',required=True);ap.add_argument('--physical',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    p=Path(a.output);assert not p.exists();glyph,meta=glyph_points(a.asset);physical=json.loads(Path(a.physical).read_text())['object_points_armor_m'];all_points=np.r_[physical,glyph]
    result={'schema':'armor-native.glyph-physical/1','armor_number':3,'physical_points':physical,'glyph_points':glyph.tolist(),'object_points':all_points.tolist(),'names':['lamp_left_bottom','lamp_left_top','lamp_right_top','lamp_right_bottom']+[f'glyph_contour_{i:02d}' for i in range(len(glyph))],'asset_sha256':hashlib.sha256(Path(a.asset).read_bytes()).hexdigest(),'scope':'fixed known geometry; target pose labels are TRAIN only','selection':meta}
    assert np.ptp(glyph[:,2])>.10 and np.ptp(glyph[:,0])>.06
    p.write_text(json.dumps(result,indent=2));print(json.dumps({'points':len(all_points),'digit_width_m':float(np.ptp(glyph[:,0])),'digit_height_m':float(np.ptp(glyph[:,2])),'turn_degrees':meta['turn_degrees']}))
if __name__=='__main__':main()
