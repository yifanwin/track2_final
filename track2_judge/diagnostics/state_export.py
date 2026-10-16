"""Judge-only physical actor -> canonical root -> frozen layout conversion."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from track2_judge.util.common import plain

def vector3(value,label):
    v=np.asarray(value,dtype=float)
    if v.shape!=(3,) or not np.isfinite(v).all():raise ValueError('invalid '+label)
    return v

def canonical_root(actor_p_m, q_wxyz, pivot_m, scale=(1,1,1)):
    q=np.asarray(q_wxyz,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)<1e-12:raise ValueError('invalid quaternion')
    rot=Rotation.from_quat(q[[1,2,3,0]]/np.linalg.norm(q))
    p=vector3(actor_p_m,'actor position'); pivot=vector3(pivot_m,'pivot');s=vector3(scale,'scale')
    if np.any(s<=0):raise ValueError('scale must be positive')
    return p-rot.apply(pivot*s)
def actor_from_root(root_m,q_wxyz,pivot_m,scale=(1,1,1)):
    return np.asarray(root_m)-canonical_root([0,0,0],q_wxyz,pivot_m,scale)
def export_layout(states, metadata, scene_settings):
    settings=copy.deepcopy(scene_settings)
    if settings.get('units')!='centimeters' or settings.get('up_axis')!='Z' or settings.get('rotation_convention')!='absolute_quaternion_xyzw':raise ValueError('unsupported frozen scene settings')
    offset=vector3(settings['scene_offset'],'scene offset')
    result={'coordinate_frame':'scene_local_cm','scene_settings':settings,'objects':[]}
    for name,s in states.items():
        m=metadata[name]; q=np.asarray(s['q_wxyz'],dtype=float)
        if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)<1e-12:raise ValueError('invalid quaternion')
        q=q/np.linalg.norm(q)
        # Legacy TTG manifest pivot is already scaled and converted to Z-up.
        # metadata pivot_scaled=True prevents a second scale application.
        scale=(1,1,1) if m.get('pivot_scaled',False) else m.get('scale',(1,1,1))
        root=canonical_root(s['actor_p_m'],q,m['pivot_m'],scale)
        result['objects'].append({'instance':name,'position':((root-offset)*100).tolist(),'rotation':q[[1,2,3,0]].tolist()})
    return plain(result)
