"""Image-only cyan/number masks inside predicted shape neighborhoods."""
import cv2,numpy as np

def evidence_masks(image,predicted,margin=4):
    x=np.asarray(image,float)
    cyan=(x[:,:,2]-x[:,:,0]>35)&(x[:,:,2]>=x[:,:,1]-10)&(x.max(-1)>=80)
    kernel=np.ones((2*margin+1,2*margin+1),np.uint8)
    blue_roi=cv2.dilate((predicted==1).astype(np.uint8),kernel).astype(bool)
    digit_roi=cv2.dilate((predicted==2).astype(np.uint8),kernel).astype(bool)
    gray=np.rint(x.mean(-1)).astype(np.uint8);sample=gray[digit_roi&~cyan]
    if len(sample)<8:return None
    threshold,_=cv2.threshold(sample.reshape(-1,1),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    white=(gray>threshold)&~cyan&digit_roi
    return {'blue':cyan&blue_roi,'white':white,'blue_roi':blue_roi,'white_roi':digit_roi,'threshold':float(threshold)}

def projected_bounds(renderer,rv,tv,K,image_shape,margin=5):
    rot=cv2.Rodrigues(np.asarray(rv))[0];p=renderer.v@rot.T+np.asarray(tv);p=p@np.asarray(K).T;p=p[:,:2]/p[:,2:3]
    lo=np.maximum(np.floor(p.min(0)-margin).astype(int),0);hi=np.minimum(np.ceil(p.max(0)+margin+1).astype(int),[image_shape[1],image_shape[0]])
    return lo,hi
