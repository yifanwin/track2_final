"""Judge-only PhysX substep event stream, explicit candidate event semantics."""
import copy, hashlib, json
import numpy as np
from track2_judge.contract.predicates import bounds,vec,OrderedCheckpoints

class EvidenceUnavailable(RuntimeError):pass

class TrustedEventRecorder:
    def __init__(self,policy,*,checkpoints=(),recovery=False):
        self.policy=copy.deepcopy(policy);self.frames=0;self.failed=False;self.entries=[];self.active=set();self.chain='0'*64
        self.counts={};self.monitors=OrderedCheckpoints(checkpoints if recovery else ())
        if policy is not None:
            if policy.get('schema')!='candidate-event-policy-v2':raise ValueError('explicit event policy required')
            if 'collision' in policy:
                rule=policy['collision']
                if rule.get('count')!='contact_onset' or not isinstance(rule.get('forbidden_pairs'),list) or not rule['forbidden_pairs']:raise ValueError('collision semantics required')
                if any(not isinstance(p,list) or len(p)!=2 or p[0]==p[1] or any(not isinstance(n,str) or not n for n in p) for p in rule['forbidden_pairs']):raise ValueError('collision pair identity required')
                self.counts['collision']=0
            if 'forbidden_zone' in policy:
                rule=policy['forbidden_zone']
                if rule.get('count')!='entry' or not isinstance(rule.get('zones'),list) or not rule['zones']:raise ValueError('zone semantics required')
                for zone in rule['zones']:
                    if zone.get('membership') not in ('point','aabb_overlap') or not isinstance(zone.get('subject'),str) or not isinstance(zone.get('id'),str):raise ValueError('explicit zone subject/membership required')
                    bounds(zone['aabb_m'])
                self.counts['forbidden_zone']=0
    def sample(self,state):
        try:
            if not state.get('trusted_physics_substep'):raise EvidenceUnavailable('nonphysical sample')
            self.frames+=1;now=set();evidence={}
            if 'collision' in self.counts:
                if not isinstance(state.get('contacts'),list):raise EvidenceUnavailable('contacts absent')
                rule=self.policy['collision'];threshold=float(rule.get('min_impulse_Ns',1e-7))
                if not np.isfinite(threshold) or threshold<0:raise EvidenceUnavailable('invalid impulse threshold')
                allowed={tuple(sorted(p)) for p in rule['forbidden_pairs']}
                known=set(state['known_contact_bodies'])
                if any(set(pair)-known for pair in allowed):raise EvidenceUnavailable('collision body identity not observed')
                for c in state['contacts']:
                    pair=tuple(sorted(c['bodies']))
                    if pair not in allowed:continue
                    points=[p for p in c['points'] if float(p['separation_m'])<=0 and np.linalg.norm(vec(p['impulse_Ns']))>=threshold]
                    if points:
                        key=('collision',)+pair;now.add(key);evidence[key]={'pair':pair,'point':copy.deepcopy(points[0])}
            if 'forbidden_zone' in self.counts:
                for zone in self.policy['forbidden_zone']['zones']:
                    box=bounds(zone['aabb_m'])
                    if zone['membership']=='point':
                        point=vec(state['judge_points_m'][zone['subject']]);inside=np.all(point>=box[0]) and np.all(point<=box[1]);witness={'point_m':point.tolist()}
                    else:
                        body=bounds(state['spatial_objects'][zone['subject']]['aabb_world_m']);inside=np.all(body[1]>=box[0]) and np.all(body[0]<=box[1]);witness={'body_aabb_world_m':body.tolist()}
                    if inside:
                        key=('forbidden_zone',zone['id'],zone['subject']);now.add(key)
                        evidence[key]={**witness,'zone_id':zone['id'],'subject':zone['subject']}
            for key in sorted(now-self.active):
                self.counts[key[0]]+=1;self.entries.append({'event':key[0],'physics_frame':self.frames,'evidence':evidence[key]})
            self.active=now
            self.monitors.sample(state,self.frames)
            encoded=json.dumps(state,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            self.chain=hashlib.sha256(self.chain.encode()+encoded).hexdigest()
        except Exception as exc:
            self.failed=True;raise EvidenceUnavailable('trusted physical sampling incomplete') from None
    def report(self,enabled,expected_frames):
        if enabled and (self.failed or self.frames==0 or self.frames!=expected_frames or set(enabled)-set(self.counts)):
            raise EvidenceUnavailable('enabled event constraints lack complete substep evidence')
        return {'event_counts':{key:self.counts[key] for key in enabled},'frames':self.frames,'expected_frames':expected_frames,
                'coverage':'physics_substep' if self.frames==expected_frames and self.frames>0 and not self.failed else 'INCOMPLETE',
                'hash_chain':self.chain,'entries':copy.deepcopy(self.entries),'P5':self.monitors.result(),
                'policy_status':'explicit candidate semantics; official definitions pending confirmation'}
