"""Read-only GT/live geometry -> robot joint trajectories. No state setters."""
import math,time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation,Slerp
from track2_judge.oracle.oracle_geometry import body_points,pose_matrix,robot_reach_bound
from track2_judge.util.common import array
from scipy.spatial import ConvexHull

def section_center(body,actor_pose,height):
    """Area-weighted engine-hull section center; no actor writes or pose setters."""
    centers=[];weights=[];cloud=[]
    for shape in body.collision_shapes:
        if not hasattr(shape,'vertices'):continue
        vertices=np.asarray(shape.vertices,dtype=float)*np.asarray(getattr(shape,'scale',[1,1,1]))
        transform=actor_pose@pose_matrix(shape.local_pose);vertices=vertices@transform[:3,:3].T+transform[:3,3]
        hull=ConvexHull(vertices);edges=set(tuple(sorted((int(face[i]),int(face[j])))) for face in hull.simplices for i,j in ((0,1),(1,2),(0,2)))
        cut=[]
        for left,right in edges:
            p,q=vertices[left],vertices[right];d=q[2]-p[2]
            if abs(d)<1e-10:
                if abs(p[2]-height)<1e-7:cut.extend((p[:2],q[:2]))
            elif 0<=(height-p[2])/d<=1:cut.append((p+(q-p)*(height-p[2])/d)[:2])
        if len(cut)<3:continue
        cut=np.unique(np.round(cut,9),axis=0)
        try:
            section=ConvexHull(cut);poly=cut[section.vertices];cross=poly[:,0]*np.roll(poly[:,1],-1)-np.roll(poly[:,0],-1)*poly[:,1]
            twice=float(cross.sum());area=abs(twice)/2
            if area<1e-10:continue
            centroid=((poly+np.roll(poly,-1,axis=0))*cross[:,None]).sum(0)/(3*twice)
            centers.append(centroid);weights.append(area);cloud.extend(poly)
        except Exception:continue
    if not weights:return None
    return np.average(centers,axis=0,weights=weights),np.asarray(cloud)

def fit(model,base,position,rotation,start,wide=False):
    inverse=np.linalg.inv(base);p=(inverse@np.r_[position,1])[:3];r=inverse[:3,:3]@rotation
    def residual(q):
        t=model.fk(q)
        return np.r_[t[:3,3]-p,.10*Rotation.from_matrix(r@t[:3,:3].T).as_rotvec()]
    seeds=[start]
    if wide:
        seeds.extend(([0,1.3,-1.3,0,.5,0],[0,.6,-1.7,0,1.,0],[0,2.,-.8,0,-.5,0]))
        seeds.extend(np.random.default_rng(1873).uniform(model.limits[:,0],model.limits[:,1],size=(3,6)))
    best=None
    for seed in seeds:
        sol=least_squares(residual,np.clip(seed,model.limits[:,0]+1e-6,model.limits[:,1]-1e-6),
                          bounds=(model.limits[:,0]+1e-7,model.limits[:,1]-1e-7),max_nfev=85,
                          ftol=1e-8,xtol=1e-8,gtol=1e-8)
        loss=float(np.linalg.norm(residual(sol.x)))
        if best is None or loss<best[0]:best=(loss,sol.x)
        if loss<.0004:break
    t=model.fk(best[1]);pe=float(np.linalg.norm(t[:3,3]-p))
    re=float(Rotation.from_matrix(r@t[:3,:3].T).magnitude())
    return best[1],{'weighted_residual':best[0],'position_error_m':pe,'rotation_error_rad':re}

def make_plan(env,task,target,model):
    started=time.monotonic();q0=array(env.agent.robot.get_qpos());base=pose_matrix(env.agent.robot.pose)
    actor=env.objects[target];body=actor._bodies[0];local=body_points(body);initial=pose_matrix(actor.pose)
    points=local@initial[:3,:3].T+initial[:3,3];bounds=np.stack([points.min(0),points.max(0)])
    radius,perlink=robot_reach_bound(env,model)
    delta=np.maximum(bounds[0]-base[:3,3],0)+np.minimum(bounds[1]-base[:3,3],0)
    lower=float(np.linalg.norm(delta))
    goal=next(r for r in task['target_layout']['objects'] if r['instance']==target)
    goalpose=np.eye(4);goalpose[:3,3]=np.asarray(goal['position'])/100
    goalpose[:3,:3]=Rotation.from_quat(goal['rotation']).as_matrix()
    move=goalpose@np.linalg.inv(initial)
    plan={'schema':'judge-private-oracle-plan-v1','oracle_only':True,'actions':[],
          'diagnostic':{'robot_collision_radius_bound_m':radius,'initial_object_aabb_distance_lower_bound_m':lower,
                        'geometry_contact_unreachable':lower>radius,'attempted_candidates':0},'stages':[]}
    def safe_probe(reason):
        plan['failure_reason']=reason
        opening=float(q0[6]-q0[7])
        for offset in (.015,0.):
            q=[float(v) for v in q0[:6]];q[0]+=offset
            for _ in range(12):plan['actions'].append([*q,opening])
        plan['stages']=[{'phase':'SAFE_INTERFACE_PROBE_NOT_GRASP','end':24}]
        plan['diagnostic']['planning_wall_s']=time.monotonic()-started
        return plan
    if lower>radius:return safe_probe('ENVIRONMENT_INITIAL_DIRECT_CONTACT_UNREACHABLE')
    center=bounds.mean(0);height=bounds[1,2]-bounds[0,2]
    forward=center[:2]-base[:2,3];forward/=max(np.linalg.norm(forward),1e-9)
    covariance=np.cov(points[:,:2].T);_,axes=np.linalg.eigh(covariance)
    narrow=axes[:,0]
    candidates=[]
    if height>.05:
        candidates.append((np.r_[center[:2],bounds[1,2]-.025],narrow,'upper_contact_band_palm_clearance'))
    # TCP marks distal fingertip, not center of the whole 10 cm finger pad.
    for frac in (.28,.45,.65):
        p=center.copy();p[2]=bounds[0,2]+height*frac
        candidates.append((p,narrow,'body_center_height_'+str(frac)))
    for distance in (.025,.055,.075):
        p=center.copy();p[:2]-=forward*distance;p[2]=bounds[0,2]+height*.30
        candidates.append((p,narrow,'near_arm_body_offset_'+str(distance)))
    if any(w in target for w in ('cup','mug','bowl')):
        near=points[np.argmin(points[:,:2]@forward)].copy()
        p=center.copy();p[:2]=near[:2]+forward*.006;p[2]=max(bounds[0,2]+.012,bounds[1,2]-.020)
        candidates.append((p,forward,'near_arm_rim'))
    if 'bottle' in target:
        candidates.insert(0,(np.r_[center[:2],bounds[1,2]-.025],narrow,'upper_body_palm_clearance'))
    if getattr(env,'judge_oracle_side_body',False):
        candidates=[(np.r_[center[:2],bounds[0,2]+height*frac],narrow,'side_body_'+str(frac)) for frac in (.45,.6,.35)]
        env.judge_oracle_tilts=(math.pi/3,-math.pi/3,math.pi/4,-math.pi/4,5*math.pi/12,-5*math.pi/12,math.pi/2,-math.pi/2)
    # AABB XY center can be far from a tapered/tilted object's contact band.
    corrected=[]
    for contact,closing,label in candidates:
        section=section_center(body,initial,float(contact[2]))
        if section is not None and not ('offset' in label or 'rim' in label):
            contact=contact.copy();contact[:2]=section[0]
            _,section_axes=np.linalg.eigh(np.cov(section[1].T));closing=section_axes[:,0]
        corrected.append((contact,closing,label))
    candidates=corrected
    selected=None;attempts=[]
    candidates=[(grip,closing*sign,label+('_axis_flipped' if sign<0 else '')) for grip,closing,label in candidates for sign in (1,-1)]
    for contact_point,closing,label in candidates:
        closing=np.r_[closing,0.];closing/=np.linalg.norm(closing)
        downward=np.array([0.,0.,-1.]);r0=np.column_stack([np.cross(closing,downward),closing,downward])
        for tilt in getattr(env,'judge_oracle_tilts',(0.,math.pi/6,-math.pi/6,math.pi/3,-math.pi/3,math.pi/2,-math.pi/2)):
            r=Rotation.from_rotvec(closing*tilt).as_matrix()@r0
            if getattr(env,'judge_oracle_side_body',False) and np.dot(r[:2,2],forward)<0:continue
            # Engine jaw pads extend 74.7 mm behind distal TCP. Aim their
            # center at the intended contact, rather than treating TCP as pad.
            side=getattr(env,'judge_oracle_side_body',False)
            offset=(.020 if abs(tilt)>1.45 else -.008) if side else .035
            grip=contact_point+r[:,2]*offset
            grip[2]=max(grip[2],bounds[0,2]+.004)
            target_grip=(move@np.r_[grip,1])[:3];target_r=move[:3,:3]@r
            pre=grip+np.array([0,0,.08]);place_pre=target_grip+np.array([0,0,.08])
            solutions=[];valid=True;start=q0[:6]
            for position,rotation in ((pre,r),(grip,r),(place_pre,target_r),(target_grip,target_r)):
                q,info=fit(model,base,position,rotation,start,wide=True);solutions.append((q,info));start=q
                if info['position_error_m']>.003 or info['rotation_error_rad']>.08:
                    valid=False;break
            attempts.append({'candidate':label,'tilt_rad':tilt,'valid':valid,'fits':[v[1] for v in solutions]})
            if valid:
                selected=(grip,r,target_grip,target_r,label,tilt,solutions);break
        if selected:break
    plan['diagnostic']['attempted_candidates']=len(attempts)
    plan['private_candidate_fits']=attempts
    if not selected:return safe_probe('ORACLE_PLANNER_GRASP_AND_PLACE_IK_UNRESOLVED')
    grip,r,target_grip,target_r,label,tilt,solutions=selected
    actions=[];stages=[];previous=np.r_[q0[:6],q0[6]-q0[7]].astype(float)
    max_fit=0.
    def joints(phase,wanted,opening,minimum):
        nonlocal previous
        start=previous.copy();want=np.r_[wanted,opening]
        duration=max(minimum,int(np.max(np.abs(want[:6]-start[:6]))/.015)+30,
                     int(abs(opening-start[6])/.003)+10)
        for n in range(1,duration+1):
            f=.5-.5*math.cos(math.pi*n/duration);command=start+(want-start)*f
            actions.append(command.tolist())
        previous=want.copy();stages.append({'phase':phase,'end':len(actions)})
    def cartesian(phase,first,last,rotation,opening,minimum,last_rotation=None):
        nonlocal previous,max_fit
        n=max(minimum,int(np.linalg.norm(last-first)/.0015)+1)
        interpolate=Slerp([0,1],Rotation.from_matrix(np.stack([rotation,last_rotation]))) if last_rotation is not None else None
        for k in range(1,n+1):
            f=.5-.5*math.cos(math.pi*k/n);position=first+(last-first)*f
            current_rotation=interpolate(f).as_matrix() if interpolate is not None else rotation
            q,info=fit(model,base,position,current_rotation,previous[:6])
            if info['position_error_m']>.003 or info['rotation_error_rad']>.08:
                q,info=fit(model,base,position,current_rotation,previous[:6],wide=True)
            max_fit=max(max_fit,info['position_error_m'])
            if info['position_error_m']>.003 or info['rotation_error_rad']>.08:
                plan['private_path_failure']={'phase':phase,'frame':k,'fit':info}
                raise ValueError('PATH_IK_UNRESOLVED')
            want=np.r_[q,opening]
            subdiv=max(1,int(np.max(np.abs(want[:6]-previous[:6]))/.025)+1,int(abs(want[6]-previous[6])/.004)+1)
            start=previous.copy()
            for j in range(1,subdiv+1):actions.append((start+(want-start)*j/subdiv).tolist())
            previous=want
        stages.append({'phase':phase,'end':len(actions)})
    try:
        pre=grip+[0,0,.08];lift=grip+[0,0,.08];above=target_grip+[0,0,.08]
        joints('PREGRASP_OPEN',solutions[0][0],.1,65)
        cartesian('APPROACH',pre,grip,r,.1,50)
        joints('CLOSE',previous[:6],0.,45)
        cartesian('LIFT',grip,lift,r,0.,85)
        cartesian('TRANSPORT',lift,above,r,0.,max(100,int(np.linalg.norm(above-lift)/.0015)),last_rotation=target_r)
        lower=target_grip+[0,0,.002]
        cartesian('LOWER',above,lower,target_r,0.,85)
        joints('RELEASE',previous[:6],.1,40)
        cartesian('RETREAT',lower,above+[0,0,.04],target_r,.1,65)
        joints('SETTLE',previous[:6],.1,30)
    except ValueError as error:
        plan['private_failed_path']=str(error);return safe_probe('ORACLE_PLANNER_'+str(error))
    if len(actions)>1800:return safe_probe('ORACLE_PLANNER_ACTION_BUDGET')
    plan['actions']=actions;plan['stages']=stages
    plan['diagnostic'].update(candidate=label,tilt_rad=tilt,planning_wall_s=time.monotonic()-started,
                              max_planned_position_error_m=max_fit,planned_actions=len(actions))
    plan['private_grasp_world_m']=grip.tolist();plan['private_target_grasp_world_m']=target_grip.tolist()
    plan['diagnostic']['grasp_reference']='GT contact band -> distal TCP using engine-measured pad offset'
    return plan
