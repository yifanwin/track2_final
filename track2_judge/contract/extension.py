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
    def verify(self,task,initial,final,config,event_evidence=None,*,initial_state=None,final_state=None,static_objects=()):
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
        # ★ 2026-10-18 连续评分：非目标物体的**逐项**位移（算 30 分的保持项）。
        #
        #   口径（`plan/Track2_终态连续评分_改造方案.md` §1.4）：
        #   * 目标物体（P1/P2 的对象）**不算**
        #   * `case.json` 的 `static_objects`（桌面等）**一律不算**
        #   * 其余可移动物体**逐项**给 `(d_pos_cm, d_rot_deg)`，`v_j = 1/M` 均分
        #   * `M = 0` ⇒ 满分（无物可碰 = 完美保持）
        #
        #   几何量**复用冻结件**（`v3.object_map` / `v3.rotation_error_deg`），
        #   不复制几何代码、也不改冻结件（§3.3 选 C）。
        goal=task.get('target_layout',initial)
        start_map=v3.object_map(initial,goal);end_map=v3.object_map(final,goal)
        static=set(static_objects or ())
        preserve=[]
        for key in sorted(set(start_map)-selected-static):
            if key not in end_map:continue          # 终态缺该物体：不算位移（缺失另有 C1 记）
            dpos=math.dist(start_map[key]['position'],end_map[key]['position'])
            drot=v3.rotation_error_deg(start_map[key]['rotation'],end_map[key]['rotation'])
            preserve.append((float(dpos),float(drot)))
        p5={'status':'not_required','passed':True}
        strict=all(r['passed'] for r in results) and violations==0
        # ★ 2026-10-18 终态连续评分（新分；旧分 `s` 保留为 legacy diagnostic）。
        #   `P1` 是硬门（§3.5 红线：它仍在 `predicates[]` 里，Gate 照常检查）。
        from track2_judge.scoring.judge_scorer import score_case, constants_for_tier
        p1=[r for r in results if r['type']=='P1']
        p2r=[r for r in results if r['type']=='P2']
        p1_passed=bool(p1 and p1[0]['passed'])
        d_pos=float(p2r[0]['position_error_cm']) if p2r and 'position_error_cm' in p2r[0] else 0.0
        d_rot=float(p2r[0]['rotation_error_deg']) if p2r and 'rotation_error_deg' in p2r[0] else 0.0
        continuous=score_case(
            targets=[{'p1_passed':p1_passed,'weight':1.0,'d_pos_cm':d_pos,'d_rot_deg':d_rot}],
            displacements=preserve,
            constants=constants_for_tier(task['tier']))
        return {'completion':completion,'s':score,'strict_success':strict,'success':strict,'status':'success' if strict else 'inconclusive' if any(r['status']=='inconclusive' for r in results) else 'failure',
                'predicates':results,'violation_count':violations,'constraints':constraint_details,'P5':p5,
                'preserve_displacements':preserve,
                'score_target':continuous['score_target'],'score_preserve':continuous['score_preserve'],
                'score_100':continuous['score_100'],
                'scoring_engine':'trusted candidate extension + continuous terminal-state scorer','official_score':None}
    def capture(self,task,seed,initial_state,final_state,result,config,event_evidence,initial_layout,final_layout,static_objects=()):
        self.packets[(task['task_id'],seed)]={'task_sha256':digest(task),'config':self.config_for_task(task,config),'result':copy.deepcopy(result),'static_objects':tuple(static_objects or ()),
            'event_evidence':copy.deepcopy(event_evidence),'event_evidence_sha256':digest(event_evidence),'initial_layout_sha256':digest(initial_layout),'final_layout_sha256':digest(final_layout),'initial_state_sha256':digest(initial_state),'final_state_sha256':digest(final_state),'initial_state':copy.deepcopy(initial_state),'final_state':copy.deepcopy(final_state)}
    def bind_recorded_run(self,task,seed,run):
        self.packets[(task['task_id'],seed)]['recorded_run_sha256']=digest(run)
    def evaluate(self,suite,submission,base_dir):
        runs=submission['runs'];tasks=suite['tasks'];R=suite['config']['repeats']
        # ★ 2026-10-18 取消 first/final：每 seed **1** 次（不再是 `*2`）。
        if len(runs)!=len(tasks)*R:raise ValueError('full repeats required')
        seen=set();per_task=[]
        for task in tasks:
            by_seed={}
            for run in [r for r in runs if r['task_id']==task['task_id']]:
                key=(task['task_id'],run['seed'])
                if key in seen:raise ValueError('duplicate trial')
                seen.add(key);packet=self.packets[key]
                if digest(run)!=packet.get('recorded_run_sha256'):raise ValueError('run timing/evidence record tamper detected')
                if packet['task_sha256']!=digest(task):raise ValueError('contract changed after execution')
                initial_layout=v3.read_json(Path(base_dir)/run['initial_layout']);final_layout=v3.read_json(Path(base_dir)/run['final_layout'])
                if digest(initial_layout)!=packet['initial_layout_sha256'] or digest(final_layout)!=packet['final_layout_sha256']:raise ValueError('recorded state tamper detected')
                if digest(packet['initial_state'])!=packet['initial_state_sha256'] or digest(packet['final_state'])!=packet['final_state_sha256'] or digest(packet['event_evidence'])!=packet['event_evidence_sha256']:raise ValueError('private evidence integrity rejected')
                # ★ 必须传**同一套** static_objects —— 否则保持项的位移列表
                #   与 capture 时不一致 → digest 失配（'independent predicate replay mismatch'）。
                verified=self.verify(task,initial_layout,final_layout,packet['config'],event_evidence=packet['event_evidence'],initial_state=packet['initial_state'],final_state=packet['final_state'],static_objects=packet.get('static_objects',()))
                if digest(verified)!=digest(packet['result']):raise ValueError('independent predicate replay mismatch')
                r=copy.deepcopy(packet['result']);r.update(elapsed_s=run['elapsed_s'])
                by_seed[run['seed']]=r
            if len(by_seed)!=R:raise ValueError('repeats missing')
            # ★ 新分：单 case 内部 **100 分**（目标 70 + 保持 30），
            #   5 个 seed 等权平均（`plan/Track2_终态连续评分_改造方案.md` §1）。
            scores=[by_seed[s]['score_100'] for s in sorted(by_seed)]
            score_100=sum(scores)/len(scores)
            per_task.append({'task_id':task['task_id'],'tier':task['tier'],
                             'score_100':score_100,
                             'score_target':sum(by_seed[s]['score_target'] for s in sorted(by_seed))/len(scores),
                             'score_preserve':sum(by_seed[s]['score_preserve'] for s in sorted(by_seed))/len(scores),
                             'episodes':len(scores)})
        path=Path(base_dir)/'extension_private_evidence.json';path.write_text(json.dumps({str(k):v for k,v in self.packets.items()},allow_nan=False));path.chmod(0o600)
        from track2_judge.contract.profile import PROFILE
        confirmed=all(t.get('scoring_profile')==PROFILE for t in tasks)
        # 总分：50 例（这里 `len(tasks)`）**等权平均**；每 case 满分 2 分保留在报告层。
        total=sum(t['score_100'] for t in per_task)/len(per_task)
        return {'schema':SCHEMA,'tasks':per_task,'candidate_score_100':total,
                # ⚠️ 旧分：**仅作 legacy diagnostic**，不是正式验收指标（§3.6）。
                'legacy_diagnostic':{'note':'旧口径（V3 summarize）已停用，语义不同，不做提升/下降解释'},
                'frozen_rule_score_100':None,'official_score':None,
                'formula':'continuous terminal-state scoring (target 70 + preserve 30)',
                'contract_status':'FROZEN_AUTHOR_PDF_RULES' if confirmed else 'DIAGNOSTIC_EXTENSION',
                'private_evidence_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
