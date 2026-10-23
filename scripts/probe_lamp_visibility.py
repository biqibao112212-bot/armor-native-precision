"""Read-only asset/label/image visibility diagnostic on TRAIN, not inference.

A CPU z-buffer uses public GLB geometry to check whether physical point labels
coincide with observable LED edges. Oracle pose is explicitly a diagnostic input.
"""
import argparse,ctypes,json,struct,hashlib,time,subprocess
from pathlib import Path
import numpy as np,cv2
from scipy.spatial.transform import Rotation
from armor_native.data import rgb
from armor_native.broad import solve_physical,statistics

def glb_mesh(path,include_digit=False):
    b=Path(path).read_bytes();n,_=struct.unpack_from('<II',b,12);g=json.loads(b[20:20+n]);o=20+n;length,_=struct.unpack_from('<II',b,o);binary=b[o+8:o+8+length]
    def accessor(index):
        a=g['accessors'][index];v=g['bufferViews'][a['bufferView']]
        d={5121:np.uint8,5123:np.uint16,5125:np.uint32,5126:np.float32}[a['componentType']];k={'SCALAR':1,'VEC3':3}[a['type']];item=np.dtype(d).itemsize
        return np.ndarray((a['count'],k),dtype=d,buffer=binary,offset=v.get('byteOffset',0)+a.get('byteOffset',0),strides=(v.get('byteStride',item*k),item)).copy()
    def matrix(n):
        if 'matrix' in n:return np.asarray(n['matrix']).reshape(4,4).T
        m=np.eye(4);m[:3,:3]=Rotation.from_quat(n.get('rotation',[0,0,0,1])).as_matrix()@np.diag(n.get('scale',[1,1,1]));m[:3,3]=n.get('translation',[0,0,0]);return m
    wanted={'_ARMOR':0,'_ARMOR_BASE':0,'_ARMOR_CENTER':0,'_ARMOR_L_L':1,'_ARMOR_L_R':1}
    if include_digit:wanted['3']=2
    digit_parent=next(n for n in g['nodes'] if n.get('name')=='_ARMOR_C')
    vertices=[];triangles=[];colors=[];offset=0;parts={}
    for node in g['nodes']:
        name=node.get('name')
        if name not in wanted:continue
        m=(matrix(digit_parent)@matrix(node)) if name=='3' else matrix(node);sizes=[]
        for p in g['meshes'][node['mesh']]['primitives']:
            assert p.get('mode',4)==4
            v=accessor(p['attributes']['POSITION']);v=np.c_[v,np.ones(len(v))]@m.T;v=v[:,:3]
            ids=accessor(p['indices']).reshape(-1,3).astype(np.int32)
            vertices.append(v);triangles.append(ids+offset);colors.extend([wanted[name]]*len(ids));offset+=len(v);sizes.append(len(ids))
        parts[name]=sizes
    assert set(parts)==set(wanted)
    return np.concatenate(vertices).astype(np.float64),np.concatenate(triangles).astype(np.int32),np.array(colors,np.uint8),parts

class Renderer:
    def __init__(self,library,asset,include_digit=False):
        self.v,self.tri,self.material,self.parts=glb_mesh(asset,include_digit)
        self.lib=ctypes.CDLL(str(library))
        p=np.ctypeslib.ndpointer
        self.lib.raster.argtypes=[p(np.float64,flags='C_CONTIGUOUS'),p(np.int32,flags='C_CONTIGUOUS'),p(np.uint8,flags='C_CONTIGUOUS'),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_double,p(np.uint8,flags='C_CONTIGUOUS'),p(np.float64,flags='C_CONTIGUOUS')]
    def __call__(self,rv,tv,K,origin,shape,center=.5):
        rot=cv2.Rodrigues(np.asarray(rv,float))[0];v=self.v@rot.T+np.asarray(tv);proj=v@np.asarray(K).T;points=np.c_[proj[:,:2]/proj[:,2:3]-origin,v[:,2]].astype(np.float64)
        h,w=shape;image=np.empty((h,w),np.uint8);depth=np.empty((h,w),np.float64)
        self.lib.raster(np.ascontiguousarray(points),self.tri,self.material,len(self.tri),w,h,center,image,depth);return image

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--geometry',required=True);ap.add_argument('--asset',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    out=Path(a.output);out.mkdir(exist_ok=False);cv2.setNumThreads(1)
    source=Path(__file__).with_name('lamp_raster.cpp');lib=out/'lamp_raster.so'
    subprocess.run(['g++','-O3','-shared','-fPIC','-std=c++17',str(source),'-o',str(lib)],check=True)
    render=Renderer(lib,a.asset);template=json.loads(Path(a.geometry).read_text())['object_points_armor_m']
    rows=[json.loads(l) for l in Path(a.manifest).open()];assert all(r['split']=='train' for r in rows)
    groups={(light,side):[] for light in ['dark','normal','bright'] for side in ['frontal','moderate']}
    seen=set()
    for r in rows:
        for o in r['objects']:
            ident=(r['session'],o['slot'])
            if ident in seen or not 3<=o['range_m']<=5:continue
            side='frontal' if o['facing_cos']>.866 else 'moderate'
            groups[(r['light'],side)].append((r,o));seen.add(ident)
    rng=np.random.default_rng(141);pairs=[v[int(i)] for v in groups.values() for i in rng.permutation(len(v))[:8]];records=[];start=time.monotonic()
    plan={'scope':'TRAIN-only oracle-pose rendering diagnostic, not deployable performance','objects':len(pairs),'asset_sha256':hashlib.sha256(Path(a.asset).read_bytes()).hexdigest(),'parts':render.parts,'simulator_edits':False,'native_rgb_resized':False,'pixel_center_candidates':[0,.5]}
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    for r,o in pairs:
        assert time.monotonic()-start<240
        im=rgb(r['image_path']);c=np.asarray(o['corners']);lo=np.maximum(np.floor(c.min(0)-8).astype(int),0);hi=np.minimum(np.ceil(c.max(0)+9).astype(int),[1440,1080])
        target=im[lo[1]:hi[1],lo[0]:hi[0]].astype(float);cyan=(target[:,:,2]-target[:,:,0]>35)&(target[:,:,2]>=target[:,:,1]-10)&(target.max(-1)>=80)
        pose=solve_physical(c,o['K'],template);masks={}
        for center in [0,.5]:
            mask=render(pose['rotation_vector'],pose['xyz_m'],o['K'],lo,cyan.shape,center)>0
            roi=cv2.dilate(mask.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool);truth=cyan&roi
            intersection=int((mask&truth).sum());union=int((mask|truth).sum())
            masks[str(center)]={'iou':intersection/max(1,union),'predicted_pixels':int(mask.sum()),'cyan_pixels_local':int(truth.sum()),'cyan_recall':intersection/max(1,int(truth.sum())),'rendered_precision':intersection/max(1,int(mask.sum()))}
        records.append({'session':r['session'],'frame_seq':r['frame_seq'],'slot':o['slot'],'range_m':o['range_m'],'facing_cos':o['facing_cos'],'light':r['light'],'masks':masks})
    result={'complete':True,'seconds':time.monotonic()-start,'scope':plan['scope'],'objects':len(records),'summary':{center:{key:statistics([r['masks'][center][key] for r in records]) for key in ['iou','cyan_recall','rendered_precision']} for center in ['0','0.5']}}
    (out/'records.json').write_text(json.dumps(records));(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
