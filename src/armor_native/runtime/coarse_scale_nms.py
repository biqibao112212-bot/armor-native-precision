import numpy as np
def scaled_nms(proposals):
 kept=[]
 for p in sorted(proposals,key=lambda p:p["score"],reverse=True):
  if all(np.linalg.norm(np.asarray(p["center"])-q["center"])>max(10.,min(64.,.2*min(p["extent_px"],q["extent_px"]))) for q in kept):kept.append(p)
 return kept
