"""Privileged geometry evidence derived from current engine collision shapes/poses."""
import hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation
from track2_judge.util.common import array,plain
from track2_judge.diagnostics.state_export import canonical_root

def transform(points,pose):
    q=array(pose.q);p=array(pose.p)
    return Rotation.from_quat(q[[1,2,3,0]]).apply(points)+p

def collision_aabb(actor):
    clouds=[]
    for shape in actor._bodies[0].collision_shapes:
        if hasattr(shape,'half_size'):
            half=np.asarray(shape.half_size,dtype=float)
            points=np.array([[x,y,z] for x in (-half[0],half[0]) for y in (-half[1],half[1]) for z in (-half[2],half[2])])
        elif hasattr(shape,'vertices'):
            points=np.asarray(shape.vertices,dtype=float)*np.asarray(getattr(shape,'scale',[1,1,1]))
        else:raise ValueError('unsupported engine shape bounds')
        if hasattr(shape,'local_pose'):points=transform(points,shape.local_pose)
        clouds.append(transform(points,actor.pose))
    if not clouds:raise ValueError('collision geometry absent')
    points=np.concatenate(clouds)
    return [points.min(axis=0).tolist(),points.max(axis=0).tolist()]

def spatial_snapshot(env,physical):
    actors={**env.objects,**getattr(env,'spatial_actors',{})};spatial={};points={}
    for name,actor in actors.items():
        meta=env.metadata.get(name,{'pivot_m':[0,0,0]});scale=[1,1,1] if meta.get('pivot_scaled') else meta.get('scale',[1,1,1])
        root=canonical_root(array(actor.pose.p),array(actor.pose.q),meta['pivot_m'],scale)
        spatial[name]={'canonical_root_m':root.tolist(),'aabb_world_m':collision_aabb(actor),'geometry_source':'live PhysX collision shapes'}
        points[name]=root.tolist()
    points['robot_tcp']=array(env.agent.tcp.pose.p).tolist()
    registry=getattr(env,'trusted_asset_registry',None)
    if registry is not None:
        records=registry['objects']
        digest=hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if registry['sha256']!=digest:raise ValueError('asset registry integrity failure')
        for name in records:
            if name not in spatial:raise ValueError('registry/live object identity mismatch')
            spatial[name]['asset_identity_verified']=True
    return plain({**physical,'spatial_objects':spatial,'judge_points_m':points,
                  'known_contact_bodies':sorted(set(actors)|{link.name for link in env.agent.robot.get_links()}),
                  **({'trusted_asset_registry':registry} if registry is not None else {})})
