"""Image-landmark consistency and size-aware duplicate suppression."""
import numpy as np

DEFAULT_CONSENSUS={'tolerance_px':1.,'relative_tolerance':.005,
                   'minimum_digit_inliers':6,'minimum_total_inliers':8,
                   'minimum_point_probability':.1}

def landmark_consensus(points,probabilities,projected,config=DEFAULT_CONSENSUS):
    if projected is None:
        return {'accepted':False,'reason':'no_joint_glyph_pose','digit_inliers':0,'total_inliers':0}
    points=np.asarray(points,float);projected=np.asarray(projected,float);p=np.asarray(probabilities,float)
    if points.shape!=projected.shape or len(points)<8:
        return {'accepted':False,'reason':'invalid_landmark_geometry','digit_inliers':0,'total_inliers':0}
    threshold=max(config['tolerance_px'],config['relative_tolerance']*float(np.max(np.ptp(projected,axis=0))))
    error=np.linalg.norm(points-projected,axis=1)
    inliers=(p>=config['minimum_point_probability'])&np.isfinite(error)&(error<=threshold)
    digit=int(inliers[4:].sum());total=int(inliers.sum())
    accepted=digit>=config['minimum_digit_inliers'] and total>=config['minimum_total_inliers']
    return {'accepted':bool(accepted),'reason':'landmark_consensus' if accepted else 'inconsistent_glyph_landmarks',
            'digit_inliers':digit,'total_inliers':total,'tolerance_px':threshold,'errors_px':error.tolist()}

def duplicate_quads(a,b,relative_extent=0.):
    a=np.asarray(a,float);b=np.asarray(b,float)
    extent=min(float(np.max(np.ptp(a,axis=0))),float(np.max(np.ptp(b,axis=0))))
    return bool(np.linalg.norm(a-b,axis=1).max()<max(3.,relative_extent*extent))

def restore_preduplicate(q):
    q=dict(q);q['image_evidence']=dict(q.get('image_evidence',{}))
    if q['image_evidence'].get('reason')=='duplicate_after_refinement':
        q['xyz_m']=q['raw_xyz_m']
        if 'xyz' in q:q['xyz']=q['raw_xyz_m']
        q['image_evidence'].update(accepted=True,reason='accepted')
    return q

def suppress_duplicates(results,relative_extent=0.):
    accepted=[]
    for q in sorted(results,key=lambda q:q['score'],reverse=True):
        if q['xyz_m'] is None:continue
        if any(duplicate_quads(q['corners'],v,relative_extent) for v in accepted):
            q['xyz_m']=None
            if 'xyz' in q:q['xyz']=None
            q['image_evidence']=dict(q['image_evidence'],accepted=False,reason='duplicate_after_refinement')
        else:accepted.append(np.asarray(q['corners']))
    return results
