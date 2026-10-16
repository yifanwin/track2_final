"""Trusted candidate predicate extension; immutable V3 arithmetic is reused.

The confirmed 73 profile uses author-PDF P1/P2 rules and unchanged V3 arithmetic.
Other task types remain isolated diagnostics. No platform scores are published.
"""
import copy,hashlib,json,math
from fractions import Fraction
from pathlib import Path
from track2_judge.scoring.scorer_bridge import v3
from track2_judge.contract.predicates import evaluate_predicate

SCHEMA='track2-contract-extension-candidate-1'
REQUIRED={'pick_place':{'P1','P2'},'attribute_sort':{'P1','P4'},'spatial_relation':{'P1','P3'},'multi_object_organize':{'P1','P2','P3','P5','C1'}}

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

class ApprovalRequired(ValueError):pass

class TrustedContractExtension:
    def __init__(self):self.packets={}
    def validate_task(self,task):
        if 'scoring_profile' in task:
            from track2_judge.contract.profile import validate
            validate(task)
        if task.get('contract_schema')!=SCHEMA:raise ValueError('explicit extension schema required')
        kind=task.get('task_type')
        if kind not in REQUIRED:raise ValueError('unsupported task type; no dual-arm Piper tasks')
        predicates=task.get('predicates');constraints=task.get('constraints',[])
        if not isinstance(predicates,list) or not predicates:raise ValueError('predicates required')
        kinds={p.get('type') for p in predicates}|({'C1'} if constraints else set())
        if not REQUIRED[kind]<=kinds:raise ValueError('PDF-required task predicates missing')
        if kinds-{'P1','P2','P3','P4','P5','C1'}:raise ValueError('unknown predicate')
        terminal=[p for p in predicates if p['type']!='P5']
        for p in terminal:
            w=p.get('weight',1)
            if isinstance(w,bool) or not isinstance(w,(int,float)) or not math.isfinite(w) or w<=0:raise ValueError('invalid predicate weight')
            if not isinstance(p.get('object'),str) or not p['object']:raise ValueError('object required')
        objects={p['object'] for p in terminal}
        for obj in objects:
            group={p['type'] for p in terminal if p['object']==obj}
            needed=REQUIRED[kind]-{'P5','C1'} if kind!='multi_object_organize' else {'P1'}
            if not needed<=group:raise ValueError('required predicates missing for object')
            if kind=='multi_object_organize' and not (group&{'P2','P3'}):raise ValueError('organize object needs pose/relation')
        if kind=='multi_object_organize' and len(objects)<2:raise ValueError('multi-object task needs >=2 targets')
        if task.get('tier') not in v3.TIERS:raise ValueError('frozen tolerance tier required')
        for c in constraints:
            if c.get('type')!='C1' or c.get('kind') not in ('non_target_displacement','collision','forbidden_zone'):raise ValueError('explicit optional C1 required')
        checkpoints=[p for p in predicates if p['type']=='P5']
        if checkpoints and (len(checkpoints)!=1 or not checkpoints[0].get('checkpoints')):raise ValueError('P5 sequence required')
        if any('weight' in p for p in checkpoints):raise ValueError('P5 cannot have standalone score weight')
        if any(p['type']=='P4' for p in terminal) and not task.get('asset_registry_sha256'):raise ValueError('pinned trusted attribute registry required')
    def config_for_task(self,task,config):
        v3.validate_config(config)
        cfg=copy.deepcopy(config)
        tier=v3.default_config(config['environment'],task['tier'])
        for key in ('tier','eps_pos_cm','eps_rot_deg','delta_max_cm'):cfg[key]=tier[key]
        cfg['event_constraints']=self.event_constraints(task,cfg)
        if 'scoring_profile' in task and cfg!=v3.default_config('sim',task['tier']):raise ValueError('frozen 73 scoring parameters cannot be overridden')
        return cfg
    def event_constraints(self,task,config):return sorted({c['kind'] for c in task.get('constraints',[]) if c['kind'] in v3.EVENTS})
    def verify(self,task,initial,final,config,event_evidence=None,*,initial_state=None,final_state=None,mode='first'):
        self.validate_task(task);config=self.config_for_task(task,config)
        if initial_state is None or final_state is None:raise ValueError('judge physical states required')
        # Caller is the judge Runner; these never originate from a Skill packet.
        results=[]
        p2=[p for p in task['predicates'] if p['type']=='P2']
        if p2:
            raw_task={k:copy.deepcopy(v) for k,v in task.items() if k in ('task_id','target_layout','initial_layout','moved_objects')}
            raw_task['predicates']=[{'type':'P2','object':p['object'],'weight':p.get('weight',1)} for p in p2]
            p2cfg=copy.deepcopy(config);p2cfg['event_constraints']=[];p2cfg['non_target_displacement']=False
            raw=v3.score_state(raw_task,final,initial,p2cfg,{})
            pose_results={p['object']:p for p in raw['predicates']}
        for p in task['predicates']:
            if p['type']=='P5':continue
            if p['type']=='P2':r=copy.deepcopy(pose_results[p['object']])
            else:
                if p['type']=='P4' and final_state.get('trusted_asset_registry',{}).get('sha256')!=task['asset_registry_sha256']:
                    r={'type':'P4','status':'inconclusive','passed':False,'reason':'approved registry mismatch'}
                else:r=evaluate_predicate(p,final_state)
            if 'scoring_profile' in task:r['compatibility']='FROZEN_AUTHOR_PDF_FINAL_STATE'
            r['weight']=p.get('weight',1);results.append(r)
        completion=float(sum((Fraction(r['weight']) for r in results if r['passed']),Fraction(0))/sum(Fraction(r['weight']) for r in results))
        score=completion if completion>=config['completion_threshold'] else 0.
        selected={p['object'] for p in task['predicates'] if 'object' in p};violations=0;constraint_details=[]
        enabled=self.event_constraints(task,config)
        if enabled:
            if event_evidence is None or event_evidence.get('coverage')!='physics_substep' or set(event_evidence.get('event_counts',{}))!=set(enabled):raise ValueError('trusted constraint evidence missing')
        for c in task.get('constraints',[]):
            if c['kind']=='non_target_displacement':
                goal=task.get('target_layout',initial)
                start=v3.object_map(initial,goal);end=v3.object_map(final,goal)
                count=sum(k not in end or math.dist(start[k]['position'],end[k]['position'])>config['delta_max_cm'] for k in start.keys()-selected)
            else:
                count=event_evidence['event_counts'][c['kind']]
                if type(count)!=int or count<0:raise ValueError('invalid trusted event count')
            violations+=count;constraint_details.append({'kind':c['kind'],'violation_count':count})
        checkpoints=[p for p in task['predicates'] if p['type']=='P5'];p5={'status':'not_required','passed':True}
        if checkpoints and mode=='final':
            p5=(event_evidence or {}).get('P5',{'status':'inconclusive','passed':False})
            # The PDF excludes standalone P5 points but does not specify how a
            # failed recovery checkpoint changes partial completion. Fail closed
            # on final scoring until this edge policy receives explicit approval.
            if p5.get('passed') is not True:raise ApprovalRequired('P5 recovery invalid: final partial-score mapping needs approval')
        strict=all(r['passed'] for r in results) and violations==0 and p5['passed']
        return {'completion':completion,'s':score,'strict_success':strict,'success':strict,'status':'success' if strict else 'inconclusive' if any(r['status']=='inconclusive' for r in results) else 'failure',
                'predicates':results,'violation_count':violations,'constraints':constraint_details,'P5':p5,'scoring_engine':'trusted candidate extension + unchanged V3 summarize','official_score':None}
    def capture(self,task,mode,seed,initial_state,final_state,result,config,event_evidence,initial_layout,final_layout):
        self.packets[(task['task_id'],mode,seed)]={'task_sha256':digest(task),'config':self.config_for_task(task,config),'result':copy.deepcopy(result),
            'event_evidence':copy.deepcopy(event_evidence),'event_evidence_sha256':digest(event_evidence),'initial_layout_sha256':digest(initial_layout),'final_layout_sha256':digest(final_layout),'initial_state_sha256':digest(initial_state),'final_state_sha256':digest(final_state),'initial_state':copy.deepcopy(initial_state),'final_state':copy.deepcopy(final_state)}
    def bind_recorded_run(self,task,mode,seed,run):
        self.packets[(task['task_id'],mode,seed)]['recorded_run_sha256']=digest(run)
    def evaluate(self,suite,submission,base_dir):
        runs=submission['runs'];tasks=suite['tasks'];R=suite['config']['repeats']
        if len(runs)!=len(tasks)*R*2:raise ValueError('full paired repeats required')
        seen=set();per_task=[]
        for task in tasks:
            groups={'first':{},'final':{}};initial={}
            for run in [r for r in runs if r['task_id']==task['task_id']]:
                key=(task['task_id'],run['mode'],run['seed'])
                if key in seen:raise ValueError('duplicate trial')
                seen.add(key);packet=self.packets[key]
                if digest(run)!=packet.get('recorded_run_sha256'):raise ValueError('run timing/replan/evidence record tamper detected')
                if packet['task_sha256']!=digest(task):raise ValueError('contract changed after execution')
                initial_layout=v3.read_json(Path(base_dir)/run['initial_layout']);final_layout=v3.read_json(Path(base_dir)/run['final_layout'])
                if digest(initial_layout)!=packet['initial_layout_sha256'] or digest(final_layout)!=packet['final_layout_sha256']:raise ValueError('recorded state tamper detected')
                if digest(packet['initial_state'])!=packet['initial_state_sha256'] or digest(packet['final_state'])!=packet['final_state_sha256'] or digest(packet['event_evidence'])!=packet['event_evidence_sha256']:raise ValueError('private evidence integrity rejected')
                verified=self.verify(task,initial_layout,final_layout,packet['config'],event_evidence=packet['event_evidence'],initial_state=packet['initial_state'],final_state=packet['final_state'],mode=run['mode'])
                if digest(verified)!=digest(packet['result']):raise ValueError('independent predicate replay mismatch')
                r=copy.deepcopy(packet['result']);r.update(elapsed_s=run['elapsed_s'],plan_revisions=run['plan_revisions'])
                groups[run['mode']][run['seed']]=r;initial[(run['mode'],run['seed'])]=packet['initial_state_sha256']
            if any(len(g)!=R for g in groups.values()) or groups['first'].keys()!=groups['final'].keys():raise ValueError('paired seeds/repeats missing')
            # Raw contact ordering can differ; compare layouts canonically as V3
            # does rather than raw physical-state hashes.
            for seed in groups['first']:
                pair=[r for r in runs if r['task_id']==task['task_id'] and r['seed']==seed]
                poses=[v3.object_map(v3.read_json(Path(base_dir)/r['initial_layout']),task.get('target_layout',v3.read_json(Path(base_dir)/r['initial_layout']))) for r in pair]
                if len(pair)!=2 or poses[0].keys()!=poses[1].keys() or not all(math.dist(poses[0][k]['position'],poses[1][k]['position'])<=1e-6 and v3.rotation_error_deg(poses[0][k]['rotation'],poses[1][k]['rotation'])<=1e-5 for k in poses[0]):raise ValueError('initial seed pairing mismatch')
            cfg=self.config_for_task(task,suite['config'])
            summary=v3.summarize([groups['first'][s] for s in sorted(groups['first'])],[groups['final'][s] for s in sorted(groups['final'])],cfg)
            per_task.append({'task_id':task['task_id'],'tier':task['tier'],**summary})
        path=Path(base_dir)/'extension_private_evidence.json';path.write_text(json.dumps({str(k):v for k,v in self.packets.items()},allow_nan=False));path.chmod(0o600)
        from track2_judge.contract.profile import PROFILE
        confirmed=all(t.get('scoring_profile')==PROFILE for t in tasks)
        score=100*sum(t['S'] for t in per_task)/len(per_task)
        return {'schema':SCHEMA,'tasks':per_task,'candidate_score_100':score,'frozen_rule_score_100':score if confirmed else None,'official_score':None,'formula':'actual immutable V3.summarize','contract_status':'FROZEN_AUTHOR_PDF_RULES' if confirmed else 'DIAGNOSTIC_EXTENSION','private_evidence_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
