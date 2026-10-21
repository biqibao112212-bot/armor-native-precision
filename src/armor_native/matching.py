"""Offline image-center assignment: accepted outputs cannot be stolen by refusals."""
import numpy as np
from scipy.optimize import linear_sum_assignment

def match_centers(predictions,objects):
 pairs={};remaining=set(range(len(objects)))
 for valid in [True,False]:
  ids=[i for i,p in enumerate(predictions) if (p.get('xyz') is not None)==valid];gt=sorted(remaining)
  if not ids or not gt:continue
  d=np.linalg.norm(np.array([predictions[i]['center'] for i in ids])[:,None]-np.array([objects[j]['center'] for j in gt])[None],axis=-1)
  limits=np.array([max(8,.15*objects[j].get('extent_px',0)) for j in gt]);ii,jj=linear_sum_assignment(np.where(d<=limits,d,1e9))
  for a,b in zip(ii,jj):
   if d[a,b]<=limits[b]:pairs[gt[b]]=ids[a];remaining.remove(gt[b])
 return pairs
