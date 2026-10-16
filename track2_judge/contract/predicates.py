"""Supplemental trusted P1/P3/P4/P5 diagnostics, separate from immutable V3 scores.
All geometry semantics are explicit versioned candidate definitions, not silently
asserted to be an official contract. Missing evidence -> inconclusive, never PASS.
"""
import copy, hashlib, json, math
import numpy as np

class MissingEvidence(ValueError):pass

def vec(value,n=3):
    a=np.asarray(value,dtype=float)
    if a.shape!=(n,) or not np.isfinite(a).all():raise MissingEvidence('finite geometry missing')
    return a

def bounds(value):
    b=np.asarray(value,dtype=float)
    if b.shape!=(2,3) or not np.isfinite(b).all() or np.any(b[0]>b[1]):raise MissingEvidence('AABB missing/invalid')
    return b

def evaluate_predicate(predicate,state):
    kind=predicate.get('type');objects=state.get('spatial_objects',{})
    try:
        if kind=='P1':
            obj=objects[predicate['object']];region=bounds(predicate['region_aabb_m'])
            mode=predicate['membership']
            if mode=='point':
                p=vec(obj['canonical_root_m']);passed=bool(np.all(p>=region[0]) and np.all(p<=region[1]))
            elif mode=='whole_aabb':
                b=bounds(obj['aabb_world_m']);passed=bool(np.all(b[0]>=region[0]) and np.all(b[1]<=region[1]))
            else:raise MissingEvidence('unsupported membership definition')
        elif kind=='P2':
            obj=objects[predicate['object']];target=predicate['target_pose_m']
            p=vec(obj['canonical_root_m']);goal=vec(target['position_m'])
            q=vec(state['objects'][predicate['object']]['q_wxyz'],4);desired=vec(target['q_wxyz'],4)
            if np.linalg.norm(q)<=0 or np.linalg.norm(desired)<=0:raise MissingEvidence('invalid rotation')
            dot=float(np.clip(abs(np.dot(q/np.linalg.norm(q),desired/np.linalg.norm(desired))),0,1))
            angle=math.degrees(2*math.acos(dot));eps=float(predicate['eps_pos_m']);rot=float(predicate['eps_rot_deg'])
            if not math.isfinite(eps) or not math.isfinite(rot) or eps<=0 or rot<=0:raise MissingEvidence('invalid pose tolerance')
            passed=bool(np.linalg.norm(p-goal)<=eps and angle<=rot)
        elif kind=='P3':
            a=objects[predicate['object']];b=objects[predicate['reference']];relation=predicate['relation'];tol=float(predicate.get('tolerance_m',0))
            if not math.isfinite(tol) or tol<0:raise MissingEvidence('invalid tolerance')
            pa=vec(a['canonical_root_m']);pb=vec(b['canonical_root_m'])
            if relation in ('left-of','right-of','behind'):
                axis=predicate['axis'];axis={'X':0,'Y':1,'Z':2}[axis]
                passed=bool(pa[axis]<pb[axis]-tol) if relation in ('left-of','behind') else bool(pa[axis]>pb[axis]+tol)
            elif relation=='between':
                pc=vec(objects[predicate['reference2']]['canonical_root_m']);d=pc-pb
                if np.linalg.norm(d)<1e-9:raise MissingEvidence('degenerate between references')
                t=float(np.dot(pa-pb,d)/np.dot(d,d));passed=0<=t<=1 and np.linalg.norm(pa-(pb+t*d))<=tol
            elif relation=='in':
                if b.get('container_geometry_type')!='approved_axis_aligned_container_region':raise MissingEvidence('container semantics not verified')
                ba=bounds(a['aabb_world_m']);bb=bounds(b['container_aabb_world_m'])
                passed=bool(np.all(ba[0]>=bb[0]-tol) and np.all(ba[1]<=bb[1]+tol))
            elif relation=='on':
                ba=bounds(a['aabb_world_m']);bb=bounds(b['aabb_world_m'])
                if 'contacts' not in state:raise MissingEvidence('support contact evidence absent')
                pair={predicate['object'],predicate['reference']}
                candidates=[point for c in state['contacts'] if set(c['bodies'])==pair for point in c.get('points',[])]
                if candidates and any('impulse_Ns' not in p or 'separation_m' not in p for p in candidates):raise MissingEvidence('support impulse evidence missing')
                touching=any(float(p['separation_m'])<=tol and np.linalg.norm(vec(p['impulse_Ns']))>1e-7 for p in candidates)
                overlap=np.all(ba[1,:2]>=bb[0,:2]) and np.all(ba[0,:2]<=bb[1,:2])
                passed=bool(touching and overlap and abs(ba[0,2]-bb[1,2])<=tol)
            else:raise MissingEvidence('unknown spatial relation')
        elif kind=='P4':
            obj=objects[predicate['object']];registry=state['trusted_asset_registry'];record=registry['objects'][predicate['object']]
            expected_hash=hashlib.sha256(json.dumps(registry['objects'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
            if registry['sha256']!=expected_hash or not obj.get('asset_identity_verified'):raise MissingEvidence('unverified asset labels')
            passed=record[predicate['attribute']]==predicate['value']
        else:raise MissingEvidence('predicate unavailable')
        return {'type':kind,'status':'success' if passed else 'failure','passed':bool(passed),'compatibility':'SUPPLEMENTAL_ONLY; not frozen V3 score'}
    except (KeyError,TypeError,ValueError,OverflowError) as exc:
        return {'type':kind,'status':'inconclusive','passed':False,'reason':'trusted evidence or explicit semantics missing','compatibility':'SUPPLEMENTAL_ONLY'}

class OrderedCheckpoints:
    def __init__(self,checkpoints=()):self.checkpoints=copy.deepcopy(list(checkpoints));self.reached=[]
    def sample(self,state,frame):
        # At most one checkpoint per frame; no Skill assertion is accepted.
        if len(self.reached)>=len(self.checkpoints):return
        item=self.checkpoints[len(self.reached)]
        results=[evaluate_predicate(p,state) for p in item['predicates']]
        if results and all(r['passed'] for r in results):self.reached.append({'name':item['name'],'physics_frame':frame,'evidence':results})
    def result(self):
        if not self.checkpoints:return {'type':'P5','status':'NOT CONFIGURED','passed':None,'reached':[],'role':'recovery diagnostic only'}
        return {'type':'P5','status':'success' if len(self.reached)==len(self.checkpoints) else 'failure',
                'passed':len(self.reached)==len(self.checkpoints),'reached':copy.deepcopy(self.reached),'role':'recovery diagnostic only; no standalone score'}
