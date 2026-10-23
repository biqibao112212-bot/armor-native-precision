from ..glyph import *
from concurrent.futures import ThreadPoolExecutor
from ..native_geometry import NativeGeometry
from .native_crops import NativeCropSizes
from .device_crops import native_device_crops
class QueuedPackedGpuCropGlyphLocalizer:
    supports_camera_attitude=True
    def __init__(self,weights,native_library,device='cuda',workers=0):
        self.geometry=NativeGeometry(native_library)
        if not self.geometry.whole_pose:raise ValueError('Full native pose library required')
        self.crop_sizes=NativeCropSizes(native_library)
        self.gpu_image=None;self.crop_buffers={}
        self.executor=ThreadPoolExecutor(max_workers=workers) if workers else None
        ck=torch.load(weights,map_location='cpu',weights_only=False);assert ck['head']=='native_glyph_geometry'
        self.device=torch.device(device);self.template=ck['geometry'];self.glyph_template=ck['glyph_geometry']['object_points'];self.config=ck['inference_config'];self.proposal_threshold=self.config['proposal_threshold'];self.mode=self.config.get('decoder','glyph_level')
        self.model=None
        from armor_native.quad import QuadLocalizer
        fallback=Path(self.config['fallback_weights'])
        if not fallback.is_absolute():fallback=Path(weights).resolve().parent/fallback
        self.fallback=QuadLocalizer(str(fallback),device)
    @torch.inference_mode()
    def __call__(self,image,proposals,K,camera_quaternion_wxyz=None):
        results=[];groups={};large=[]
        for p,size in zip(proposals,self.crop_sizes(proposals,K,self.glyph_template)):
            if size is None or size>512:large.append(p);continue
            origin=np.floor(np.asarray(p['center'],np.float64)).astype(np.int64)-size//2;groups.setdefault(size,[]).append((p,size,origin))
        pending_meta=[];pending_outputs=[]
        for size,items in groups.items():
            batch=max(1,262144//(size*size))
            for start in range(0,len(items),batch):
                block=items[start:start+batch]
                if self.gpu_image is None or tuple(self.gpu_image.shape)!=(3,*image.shape[:2]):raise ValueError("Supply current coarse normalized native GPU image")
                key=(len(block),size)
                x=native_device_crops(self.gpu_image,[z[2] for z in block],size,self.crop_buffers.get(key));self.crop_buffers[key]=x
                with torch.autocast('cuda',dtype=torch.float16,enabled=self.device.type=='cuda'):pred=self.model(x)
                pending_meta.extend(block);pending_outputs.append(pred['packed'].clone())
        if pending_outputs:
            packed=torch.cat(pending_outputs,0).cpu().numpy();local=packed[...,:2];peaks=packed[...,2]
            def refine(item):
                (p,size,origin),points,confidence=item
                points=points+origin;pose=self.geometry.fit_glyph_pose(points,confidence,K,self.glyph_template,camera_quaternion_wxyz,self.mode)
                source=self.mode
                if pose is None and min(confidence[:4])>=.1:
                    pose=self.geometry.fit_glyph_pose(points,confidence,K,self.glyph_template,None,'direct');source='direct_fallback'
                corners=np.asarray(pose['corners'] if pose is not None else points[:4]);evidence=self.geometry.lamp_evidence(image,corners,rule=self.config['observability_rule'])
                boundary=bool((corners<0).any() or (corners[:,0]>=image.shape[1]).any() or (corners[:,1]>=image.shape[0]).any())
                accepted=pose is not None and evidence['accepted'] and not boundary
                if self.mode in {'direct','lamp_level'}:accepted=accepted and min(confidence[:4])>=.1
                reason='accepted' if accepted else ('unsupported_landmarks' if pose is None else ('image_boundary' if boundary else ('low_corner_probability' if self.mode in {'direct','lamp_level'} and min(confidence[:4])<.1 else evidence['reason'])))
                consensus=None
                if self.config.get('landmark_consensus') is not None:
                    from armor_native.glyph_quality import landmark_consensus
                    consensus=landmark_consensus(points,confidence,None if pose is None or source!='glyph_level' else pose.get('projected_landmarks'),self.config['landmark_consensus'])
                    if accepted and not consensus['accepted']:
                        accepted=False;reason=consensus['reason']
                raw=None if pose is None else pose['xyz_m']
                return {'score':p['score'],'proposal_center':p['center'],'center':((corners.min(0)+corners.max(0))/2).tolist(),'corners':corners.tolist(),'landmarks':points.tolist(),'point_probabilities':confidence.tolist(),'corner_peak_probabilities':confidence[:4].tolist(),'native_size':size,'xyz_m':raw if accepted else None,'raw_xyz_m':raw,'image_evidence':dict(evidence,accepted=bool(accepted),reason=reason),'reprojection_rms_px':None if pose is None else pose['reprojection_rms_px'],'used_points':None if pose is None else pose['used_points'],'decoder_source':source,'camera_attitude_used':False if pose is None else pose.get('camera_attitude_used',False),'landmark_consensus':consensus}
            pending=zip(pending_meta,local,peaks)
            results.extend(map(refine,pending) if self.executor is None else self.executor.map(refine,pending))
        from armor_native.glyph_quality import restore_preduplicate,suppress_duplicates
        if large:
            fallback=self.fallback(image,large,K)
            # One global suppression pass, including native large-crop results.
            if self.config.get('duplicate_relative_extent',0.)>0:
                fallback=[restore_preduplicate(q) for q in fallback]
            results.extend(fallback)
        return suppress_duplicates(results,self.config.get('duplicate_relative_extent',0.))
