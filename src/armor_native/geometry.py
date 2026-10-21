"""Architecture-independent native RGB ROI and calibrated corner geometry.

Training labels are separate from image-only model inputs. Online crop anchors
must come from a detector or causal tracker, never from these training labels.
"""
import hashlib,json
from pathlib import Path
import cv2,numpy as np,torch
SIZE=128
WIDTH=.1337707
HEIGHT=.0538873
OBJECT_POINTS=np.array([[-WIDTH/2,HEIGHT/2,0],[-WIDTH/2,-HEIGHT/2,0],[WIDTH/2,-HEIGHT/2,0],[WIDTH/2,HEIGHT/2,0]],np.float64)

def crop_native(image,anchor,size=SIZE):
    """Copy pixels using integer translation and zero padding; never interpolate."""
    assert image.dtype==np.uint8 and image.ndim==3 and image.shape[2]==3
    origin=np.floor(np.asarray(anchor,np.float64)).astype(np.int64)-size//2
    x,y=origin;h,w=image.shape[:2];x0=max(x,0);y0=max(y,0);x1=min(x+size,w);y1=min(y+size,h)
    patch=np.zeros((size,size,3),np.uint8);valid=np.zeros((size,size),bool)
    if x1>x0 and y1>y0:
        patch[y0-y:y1-y,x0-x:x1-x]=image[y0:y1,x0:x1];valid[y0-y:y1-y,x0-x:x1-x]=True
    return patch,origin,valid

def local_camera(K,origin):
    k=np.asarray(K,np.float64).copy();k[:2,2]-=origin;return k

def corner_targets(corners,size=SIZE,sigma=1.5):
    """Four semantic point heatmaps and subpixel offsets, in native pixel units."""
    c=np.asarray(corners,np.float64);assert c.shape==(4,2)
    if not ((c>=0).all() and (c<size).all()):raise ValueError('supervised corners fall outside native ROI')
    heatmap=np.zeros((4,size,size),np.float32);offset=np.zeros((8,size,size),np.float32);mask=np.zeros((4,size,size),np.float32);yy,xx=np.ogrid[:size,:size]
    for j,(u,v) in enumerate(c):
        ix,iy=np.floor([u,v]).astype(int);heatmap[j]=np.exp(-((xx-ix)**2+(yy-iy)**2)/(2*sigma**2));offset[j*2:j*2+2,iy,ix]=[u-ix,v-iy];mask[j,iy,ix]=1
    return heatmap,offset,mask

def decode_points(heatmaps,offsets):
    """Decode four pixel peaks plus bounded subpixel offsets, batched Torch."""
    n,k,h,w=heatmaps.shape;assert k==4 and offsets.shape==(n,8,h,w)
    index=heatmaps.flatten(2).argmax(-1);base=torch.stack([index%w,index//w],dim=-1).to(offsets.dtype);off=offsets.reshape(n,4,2,h*w).gather(-1,index[:,:,None,None].expand(-1,-1,2,1)).squeeze(-1).sigmoid();return base+off

def solve_geometry(corners,K):
    """Same calibrated IPPE convention as the baseline; invalid poses explicit."""
    try:
        ok,rv,tv=cv2.solvePnP(OBJECT_POINTS,np.asarray(corners,np.float64),np.asarray(K,np.float64),np.zeros(5),flags=cv2.SOLVEPNP_IPPE)
        if not ok or not np.isfinite(tv).all() or tv[2,0]<=0:return None
        reproj,_=cv2.projectPoints(OBJECT_POINTS,rv,tv,np.asarray(K,np.float64),np.zeros(5))
        return {'xyz_m':tv.reshape(3).tolist(),'rotation_vector':rv.reshape(3).tolist(),'reprojection_rms_px':float(np.sqrt(np.mean((reproj.reshape(4,2)-corners)**2)))}
    except cv2.error:return None
