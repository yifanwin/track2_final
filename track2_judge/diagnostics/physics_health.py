"""Judge-only observational physics health. No controller or score mutation."""
import hashlib,json,math
from collections import deque

def vector(value):
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    if hasattr(value,'tolist'):value=value.tolist()
    while isinstance(value,list) and len(value)==1 and isinstance(value[0],list):value=value[0]
    return [float(x) for x in value]

def finite_tree(value,path=''):
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    if hasattr(value,'tolist'):value=value.tolist()
    if isinstance(value,dict):return {k:finite_tree(v,path+'/'+str(k)) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [finite_tree(v,path+'/'+str(i)) for i,v in enumerate(value)]
    if isinstance(value,float) and not math.isfinite(value):return {'NONFINITE':str(value),'field':path}
    return value

def has_nonfinite(value):
    if isinstance(value,dict):
        return set(value)=={'NONFINITE','field'} or any(has_nonfinite(v) for v in value.values())
    if isinstance(value,list):return any(has_nonfinite(v) for v in value)
    return False

class TrustedPhysicsHealthRecorder:
    def __init__(self,*,dt,gravity=(0,0,-9.81),speed_alert_m_s=10.,gravity_residual_alert_m_s2=1.,penetration_alert_m=.01,point_impulse_alert_Ns=1.,support_xy_bounds=None,evidence_frames=32):
        if not math.isfinite(dt) or dt<=0 or not 1<=evidence_frames<=64:raise ValueError('invalid health configuration')
        self.dt=float(dt);self.gravity=vector(gravity);self.speed_limit=float(speed_alert_m_s);self.residual_limit=float(gravity_residual_alert_m_s2)
        self.penetration_limit=float(penetration_alert_m);self.impulse_limit=float(point_impulse_alert_Ns)
        if len(self.gravity)!=3 or not all(math.isfinite(v) for v in self.gravity):raise ValueError('invalid gravity')
        if not all(math.isfinite(v) and v>0 for v in (self.speed_limit,self.residual_limit,self.penetration_limit,self.impulse_limit)):raise ValueError('invalid diagnostic threshold')
        self.bounds=None if support_xy_bounds is None else [vector(v) for v in support_xy_bounds]
        if self.bounds is not None and (len(self.bounds)!=2 or any(len(v)<2 for v in self.bounds) or not all(math.isfinite(v) for row in self.bounds for v in row) or any(self.bounds[0][i]>=self.bounds[1][i] for i in (0,1))):raise ValueError('invalid private support bounds')
        self.frames=0;self.previous={};self.first_flags={};self.counts={};self.max_speed=0;self.nonfinite=0;self.velocity_limits_checked=False
        self.ring=deque(maxlen=evidence_frames);self.windows={};self.chain='0'*64;self.max_ballistic_residual=0;self.ballistic_frames=0
    def _flag(self,kind,detail):
        self.counts[kind]=self.counts.get(kind,0)+1
        if kind not in self.first_flags:
            self.first_flags[kind]={'substep':self.frames,**detail};self.windows[kind]=list(self.ring)
    def sample(self,state,*,command=None):
        if state.get('trusted_physics_substep') is not True:raise ValueError('untrusted health sample')
        self.frames+=1
        safe=finite_tree(state);record={'substep':self.frames,'state':safe,'command':finite_tree(command)}
        encoded=json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False)
        self.chain=hashlib.sha256((self.chain+encoded).encode()).hexdigest()
        self.ring.append(record)
        if has_nonfinite(safe):
            self.nonfinite+=1;self._flag('NONFINITE_PHYSICS_STATE',{'detail':'private field evidence retained'});return
        control=state.get('judge_control',{})
        if 'declared_velocity_limits' in control and 'qvel' in state:
            self.velocity_limits_checked=True
            velocities=vector(state['qvel']);limits=vector(control['declared_velocity_limits'])
            indices=[i for i,(v,limit) in enumerate(zip(velocities,limits)) if abs(v)>limit+1e-4]
            if indices:self._flag('DECLARED_JOINT_VELOCITY_EXCEEDED',{'joint_indices':indices,'units':'rad/s for arm; m/s for fingers','scope':'observed physical velocity, not commanded slew or score'})
        contacts=state.get('contacts',[])
        for contact in contacts:
            for point in contact.get('points',[]):
                depth=max(0,-float(point['separation_m']));impulse=math.sqrt(sum(v*v for v in vector(point['impulse_Ns'])))
                if depth>self.penetration_limit:self._flag('DEEP_CONTACT_ALERT',{'bodies':contact.get('bodies',[]),'penetration_m':depth,'not_divergence_proof':True})
                if impulse>self.impulse_limit:self._flag('CONTACT_IMPULSE_ALERT',{'bodies':contact.get('bodies',[]),'point_impulse_Ns':impulse,'not_divergence_proof':True})
        for name,obj in state.get('objects',{}).items():
            if 'linear_velocity_m_s' not in obj:continue
            v=vector(obj['linear_velocity_m_s']);p=vector(obj['actor_p_m'])
            if len(v)!=3 or len(p)!=3:raise ValueError('invalid physical vector')
            speed=math.sqrt(sum(x*x for x in v));self.max_speed=max(self.max_speed,speed)
            touched=any(name in c.get('bodies',[]) for c in contacts)
            if speed>self.speed_limit:self._flag('SPEED_ALERT',{'object':name,'speed_m_s':speed,'not_divergence_proof':True})
            if self.bounds is not None:
                if any(p[i]<self.bounds[0][i] or p[i]>self.bounds[1][i] for i in (0,1)):
                    self._flag('OUT_OF_SUPPORT_XY',{'object':name,'membership':'actor center; not forbidden-zone policy'})
            previous=self.previous.get(name)
            if previous and not touched and not previous['contact'] and max(speed,math.sqrt(sum(x*x for x in previous['v'])))>.1:
                residual=math.sqrt(sum(((v[i]-previous['v'][i])/self.dt-self.gravity[i])**2 for i in range(3)))
                if residual>self.residual_limit:
                    self._flag('CONTACT_FREE_ACCELERATION_ALERT',{'object':name,'gravity_residual_m_s2':residual,'not_root_cause_proof':True})
                elif speed>.1:
                    self.ballistic_frames+=1;self.max_ballistic_residual=max(self.max_ballistic_residual,residual)
                    if speed>self.speed_limit:self._flag('HIGH_SPEED_GRAVITY_COMPATIBLE',{'object':name,'gravity_residual_m_s2':residual})
            self.previous[name]={'v':v,'contact':touched}
    def report(self,expected_frames):
        complete=self.frames==expected_frames and self.frames>0
        return {'schema':'physics-health-diagnostic-v1','coverage':'COMPLETE' if complete else 'MISSING_OR_INCOMPLETE','frames':self.frames,'expected_frames':expected_frames,
            'status':'NO_SAMPLES' if self.frames==0 else 'NONFINITE_PHYSICS' if self.nonfinite else 'REVIEW_REQUIRED' if self.counts else 'NO_ALERTS_IN_SAMPLES',
            'nonfinite_frames':self.nonfinite,'first_flags':self.first_flags,'flag_counts':self.counts,'max_speed_m_s':self.max_speed,
            'gravity_compatible_frames':self.ballistic_frames,'max_accepted_gravity_residual_m_s2':self.max_ballistic_residual,'chain_sha256':self.chain,
            'support_envelope_configured':self.bounds is not None,'declared_velocity_limits_checked':self.velocity_limits_checked,
            'threshold_scope':'DIAGNOSTIC ONLY; no automatic score, clamp, abort, or policy change','public_skill_visibility':False}
    def private_evidence(self):return self.windows
