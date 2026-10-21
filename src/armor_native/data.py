"""Original pixels and session-balanced, deterministic supervised ROI draws."""
import argparse,json
from pathlib import Path
import cv2,numpy as np,torch
from torch.utils.data import Dataset
from .geometry import crop_native,corner_targets,local_camera

def rgb(path):
 image=cv2.imread(str(path),cv2.IMREAD_COLOR)
 if image is None:raise FileNotFoundError(path)
 return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

def read_manifest(paths):
 rows=[]
 for path in paths:
  path=Path(path)
  for line in path.read_text().splitlines():
   if not line.strip():continue
   r=json.loads(line);image=Path(r.get('image_path',r.get('image','')))
   r['image_path']=str(image if image.is_absolute() else path.parent/image)
   for key in ['session','objects']:assert key in r,key
   rows.append(r)
 return rows

def bin_key(o):return (0 if o['xyz_m'][2]<4 else 1,min(2,max(0,int((o['facing_cos']-.25)/.25))))

class BalancedRois(Dataset):
 def __init__(self,rows,draws,seed=20261024,jitter=8,start=0):
  self.rows=rows;self.draws=draws;self.seed=seed;self.jitter=jitter;self.start=start;self.pool={}
  for ri,r in enumerate(rows):
   for oi,o in enumerate(r['objects']):
    if 3<=o['xyz_m'][2]<=5:self.pool.setdefault(bin_key(o),{}).setdefault(r['session'],[]).append((ri,oi))
  self.bins=sorted(self.pool)
  if not self.bins:raise ValueError('No labelled3-5m training objects')
 def __len__(self):return self.draws
 def describe(self,index):
  rng=np.random.default_rng(self.seed+self.start+index);key=self.bins[int(rng.integers(len(self.bins)))];ss=sorted(self.pool[key]);session=ss[int(rng.integers(len(ss)))];objects=self.pool[key][session];ri,oi=objects[int(rng.integers(len(objects)))];r=self.rows[ri];o=r['objects'][oi];anchor=np.array(o['center'])+rng.uniform(-self.jitter,self.jitter,2);return r,o,anchor
 def __getitem__(self,index):
  r,o,anchor=self.describe(index);patch,origin,_=crop_native(rgb(r['image_path']),anchor);corners=np.array(o['corners'])-origin;hm,offset,mask=corner_targets(corners)
  return {'image':torch.from_numpy(patch.transpose(2,0,1).copy()).float()/255,'heatmap':torch.from_numpy(hm),'offset':torch.from_numpy(offset),'point_mask':torch.from_numpy(mask),'corners':torch.tensor(corners,dtype=torch.float32),'K':torch.tensor(local_camera(o['K'],origin),dtype=torch.float64),'xyz':torch.tensor(o['xyz_m'],dtype=torch.float64),'origin':torch.tensor(origin,dtype=torch.int64)}

def main():
 ap=argparse.ArgumentParser(description='Build a manifest from existing session annotations; no frame split or relabeling.');ap.add_argument('--annotations',nargs='+',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True);count=0
 with dest.open('x') as out:
  for path in a.annotations:
   path=Path(path).resolve()
   for r in read_manifest([path]):
    out.write(json.dumps(r)+'\n');count+=1
 print(json.dumps({'frames':count,'output':str(dest)}))
if __name__=='__main__':main()
