import numpy as np
from scipy.spatial.transform import Rotation
from track2_judge.util.common import array

def pose_matrix(pose):
    q=array(pose.q);p=array(pose.p);t=np.eye(4)
    t[:3,:3]=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();t[:3,3]=p
    return t

def body_points(body):
    clouds=[]
    for shape in body.collision_shapes:
        if hasattr(shape,'vertices'):
            points=np.asarray(shape.vertices,dtype=float)*np.asarray(getattr(shape,'scale',[1,1,1]))
        elif hasattr(shape,'half_size'):
            half=np.asarray(shape.half_size,dtype=float)
            points=np.array([[x,y,z] for x in (-half[0],half[0]) for y in (-half[1],half[1]) for z in (-half[2],half[2])])
        else:raise ValueError('unsupported collision shape')
        t=pose_matrix(shape.local_pose)
        clouds.append(points@t[:3,:3].T+t[:3,3])
    return np.concatenate(clouds) if clouds else np.zeros((0,3))

def robot_reach_bound(env,model):
    import sapien
    prefix={'world':0.};joints={child:(name,kind,parent,origin,axis) for name,kind,parent,child,origin,axis in model.joints}
    for name,kind,parent,child,origin,axis in model.joints:
        limit=.05 if kind=='prismatic' else 0.
        prefix[child]=prefix[parent]+float(np.linalg.norm(origin[:3,3]))+limit
    result=[]
    for link in env.agent.robot.get_links():
        name=link.name.removeprefix('scene-0_')
        if name not in prefix:raise ValueError('robot link identity mismatch')
        body=link._objs[0]
        if not hasattr(body,'collision_shapes'):
            body=body.find_component_by_type(sapien.physx.PhysxArticulationLinkComponent)
        points=body_points(body)
        if not len(points):continue
        if name in joints and joints[name][1]=='prismatic':
            jname,_,parent,origin,axis=joints[name]
            extremes=(0,.05) if jname=='gripper_joint1' else (-.05,0)
            radius=max(float(np.linalg.norm((points+axis*q)@origin[:3,:3].T+origin[:3,3],axis=1).max()) for q in extremes)
            bound=prefix[parent]+radius
        else:bound=prefix[name]+float(np.linalg.norm(points,axis=1).max())
        result.append({'link':name,'radius_upper_bound_m':bound})
    return max(r['radius_upper_bound_m'] for r in result),result
