"""Independent candidate Runner: actual state -> original Recorder -> V3.
All suite, seed and state data remain judge side. Only approved JSON crosses RPC.
"""
import copy, hashlib, json, time, os
from pathlib import Path
from track2_judge.util.common import plain
from track2_judge.scoring.scorer_bridge import v3, FrozenVerifier
from track2_judge.sandbox.rpc import (BudgetTimeout, SkillProcess, SkillTimeout,
                                     encode, SecurityNotReady)
from track2_judge.runner.recorder import TrustedRecorder
from track2_judge.runner.legacy_util import canonical_json_hash
from track2_judge.sandbox.package import unpack_skill,PackageExecutor
from track2_judge.contract.predicates import evaluate_predicate
import inspect
from track2_judge.scoring.result_gate import ResultGate, write_receipt, block
from track2_judge.toolapi.client import (
    ActionTimeout, ToolCallError, build_episode_tools,)

class ActionRejected(ValueError): pass
class SimulatorExecutionError(RuntimeError): pass

class CandidateRunner:
    def __init__(self, factory, *, isolated=True, trusted=False, timeout_s=20, max_steps=2400, skill_path=None,package_zip=None,supplemental_contracts=None,contract_extension=None,action_timeout_s=None,adapter_clock=None):
        self.factory=factory;self.isolated=isolated;self.trusted=trusted;self.timeout_s=timeout_s;self.max_steps=max_steps
        self.skill_path=Path(skill_path).resolve() if skill_path else Path(__file__).resolve().parents[1] / 'skills' / 'trusted.py'
        self.contract_extension=contract_extension
        self.package_zip=package_zip;self.supplemental_contracts=supplemental_contracts or {}
        # 在 adapter 外面算「仿真+等待」的时间；`runner.tool_service_s` 只记 Gateway 之前的搬运。
        self.action_timeout_s=action_timeout_s
        self.adapter_clock=adapter_clock
    def run(self,suite,output,public_inputs):
        v3.validate_config(suite['config'])
        if not self.isolated and not self.trusted: raise SecurityNotReady('SECURITY_NOT_READY')
        for task in suite['tasks']: (self.contract_extension or v3).validate_task(task)
        output=Path(output).resolve()
        if output.exists() and any(output.iterdir()): raise ValueError('output must be new/empty')
        output.mkdir(parents=True,mode=0o700,exist_ok=True)
        if os.name=='posix':
            if output.stat().st_uid!=os.getuid():raise PermissionError('judge output owner mismatch')
            output.chmod(0o700)
        skill=self.skill_path
        package=unpack_skill(self.package_zip,output/'private_skill_staging') if self.package_zip else None
        # 1.7：原 tool_profile 字符串门禁已删（Q16 单一新轨，不再按固定字符串判准入）。
        # 第 2 步改由 Gateway 的 required_tools 子集门禁接管 —— 见 toolapi/schema.py。
        skill_hash=package.package_hash if package else hashlib.sha256(skill.read_bytes()).hexdigest()
        recorder=TrustedRecorder(output/'artifacts',skill_hash=skill_hash,suite_hash=canonical_json_hash(suite),runs_dir=output)
        verifier=self.contract_extension or FrozenVerifier(); failures=[]; trace=[];supplemental=[];event_reports=[];lifecycles=[];health_reports=[]
        for index,task in enumerate(suite['tasks']):
            config=self.contract_extension.config_for_task(task,suite['config']) if self.contract_extension else suite['config']
            enabled=self.contract_extension.event_constraints(task,config) if self.contract_extension else config['event_constraints']
            for seed in range(suite['config']['repeats']):
                first_initial=None
                for mode in ('first','final'):
                    adapter=None; process=None; start=time.monotonic(); execution_elapsed=0.; compute_elapsed=0.; attempts=0; status='error'; initial=None; phase='factory'; error_category=None
                    try:
                        adapter=self.factory(task,seed,mode) if len(inspect.signature(self.factory).parameters)>=3 else self.factory(task,seed)
                        phase='reset'
                        adapter.reset(seed=seed,public_task_inputs=copy.deepcopy(public_inputs))
                        phase='initial_state'
                        initial=adapter.get_layout(); final=initial
                        initial_state=adapter.get_final_state() if self.contract_extension else None
                        def verify_current(final,event_evidence):
                            if not self.contract_extension:return verifier.verify(task,initial,final,config,event_evidence=event_evidence)
                            state=adapter.get_final_state()
                            result=verifier.verify(task,initial,final,config,event_evidence=event_evidence,initial_state=initial_state,final_state=state,mode=mode)
                            verifier.capture(task,mode,seed,initial_state,state,result,config,event_evidence,initial,final)
                            return result
                        if mode=='first': first_initial=canonical_json_hash(initial)
                        elif first_initial!=canonical_json_hash(initial): raise RuntimeError('paired initial state mismatch')
                        # 第 2 步：工具面经 Gateway。gateway → ToolClient → （沙箱内 JSON-lines，或直连）
                        tools=build_episode_tools(
                            adapter, sandboxed=bool(package),
                            action_timeout_s=self.action_timeout_s,
                            max_action_steps=self.max_steps,
                            adapter_clock=self.adapter_clock,
                        )
                        if package:
                            phase='skill_initialize'
                            process=PackageExecutor(package,output/'skill_work',isolated=self.isolated,trusted=self.trusted,timeout_s=self.timeout_s)
                            context={'skill_api_version':'ctrc-skill-v1','mode':mode,'environment':'sim','timeout_s':self.timeout_s,'tool_call_budget':self.max_steps}
                            # ⚠️ `mode` 保留是**只为诊断可观测**（A4 / 2026-10-17）：
                            #    它现在**不承诺任何行为差异** —— `first`/`final` 的
                            #    差别只在**裁判调几次 `run()`**（1 次 vs 最多 3 次），
                            #    与选手在一轮里怎么闭环无关。
                            #    选手**不应**根据它改变算法（见选手指南 §5.3）。
                            process.initialize(context)
                        else:process=SkillProcess(skill,output/'skill_work',isolated=self.isolated,trusted=self.trusted)
                        calls=[]
                        def dispatch(method,payload):
                            # 第 2 步：准入与 Action 生命周期交给 Gateway。
                            # 第 1.7 步删掉的白名单/字符串门禁由这里等价替代：
                            #   * 未声明的 tool  → ToolClient.call 拒绝或 Gateway 403
                            #   * schema 不过     → Gateway 400
                            #   * 并发已满        → Gateway 429
                            # 时间归属也在这里变准：一次 Action 的仿真+等待都算执行时间。
                            if len(calls)>=self.max_steps: raise ActionRejected('episode tool budget exhausted')
                            try: value=tools.dispatch(method,payload)
                            except ToolCallError as exc:
                                # 超时归 timeout（与第 1 步的行为一致）；其余算被拒。
                                # 2026-10-18：动作层超时是**链路故障**（动作超时且物理
                                # 状态不明），归 `infrastructure_timeout`；它不继承
                                # `BudgetTimeout`，所以不会被下面的 `except BudgetTimeout` 放行。
                                if isinstance(exc,ActionTimeout):
                                    raise SkillTimeout('action execution timeout') from None
                                raise ActionRejected('action rejected: '+str(exc)[:120]) from None
                            except (RuntimeError,OSError):
                                raise SimulatorExecutionError('simulator execution failed') from None
                            calls.append({'method':method,'payload':plain(payload)})
                            if method!='observe':
                                trace.append({'task_index':index,'seed':seed,'mode':mode,'step':len(calls),'physical':adapter.get_final_state()})
                            return value
                        phase='skill_run'
                        for attempt in range(1 if mode=='first' else 3):
                            attempts+=1
                            try:
                                remaining=self.timeout_s-compute_elapsed
                                # 这条是**比赛预算**（episode 墙钟）—— 可评分终态。
                                if remaining<=0: raise BudgetTimeout('episode deadline')
                                active_start=time.monotonic()
                                service_start=getattr(process,'tool_service_s',0.)
                                phase='skill_run'
                                try: process.run(adapter.observe(),dispatch,remaining,self.max_steps-len(calls))
                                finally:
                                    active=time.monotonic()-active_start;execution_elapsed+=active
                                    compute_elapsed+=max(0.,active-(getattr(process,'tool_service_s',0.)-service_start))
                                phase='verification'
                                final=adapter.get_layout();event_evidence=adapter.get_event_evidence(enabled)
                                verification=verify_current(final,event_evidence)
                                status='success' if verification['success'] else 'failure'
                            except BudgetTimeout:
                                # 正常预算耗尽（episode 墙钟 / 控制步预算）：
                                # 冻结当前物理状态、照常评分 —— 不算执行异常。
                                status='budget_timeout'
                                error_category='BUDGET_TIMEOUT'
                                if isinstance(process,PackageExecutor):process.abort()
                                break
                            except SkillTimeout:
                                # 动作层超时（物理状态不明）= 链路故障 —— 阻断。
                                status='infrastructure_timeout'
                                error_category='SKILL_TIMEOUT'
                                if isinstance(process,PackageExecutor):process.abort()
                                break
                            except (ValueError,RuntimeError,OSError) as execution_error:
                                status='error'
                                error_category='ACTION_REJECTED' if isinstance(execution_error,ActionRejected) else 'SIMULATOR_EXECUTION_ERROR' if isinstance(execution_error,SimulatorExecutionError) else 'VERIFIER_OR_STATE_ERROR' if phase=='verification' else 'SKILL_OR_PROTOCOL_ERROR'
                                if isinstance(process,PackageExecutor):process.abort()
                                break
                            if status=='success': break
                        phase='verification'
                        final=adapter.get_layout();event_evidence=adapter.get_event_evidence(enabled)
                        verification=verify_current(final,event_evidence)
                        event_reports.append({'task_index':index,'mode':mode,'seed':seed,**event_evidence})
                        state=adapter.get_final_state()
                        contract=self.supplemental_contracts.get(task['task_id'],{})
                        supplemental.append({'task_index':index,'mode':mode,'seed':seed,
                            'predicates':[evaluate_predicate(p,state) for p in contract.get('predicates',[])],
                            'P5':event_evidence['P5'],'score_policy':'SEPARATE DIAGNOSTICS; frozen score unchanged'})
                        phase='record'
                        recorded_run=recorder.record(task_index=index,task=task,mode=mode,seed=seed,environment='sim',initial_layout=initial,final_layout=final,
                            elapsed_s=max(execution_elapsed,1e-6),plan_revisions=max(0,attempts-1),event_violations=event_evidence['event_counts'],verification=verification,
                            tool_calls=calls,status=status,error=error_category)
                        if self.contract_extension:verifier.bind_recorded_run(task,mode,seed,recorded_run)
                        failures.append({'seed':seed,'mode':mode,'status':status,'attempts':attempts,'error_category':error_category,'skill_active_wall_s':execution_elapsed,'skill_compute_wall_s':compute_elapsed,'tool_service_s':getattr(process,'tool_service_s',0.),'total_wall_s':time.monotonic()-start,'success_scope':'FROZEN_AUTHOR_PDF_RULES' if 'scoring_profile' in task else 'TRUSTED_DIAGNOSTIC_EXTENSION' if self.contract_extension else 'FROZEN_V3_P2_C1_ONLY'})
                    except Exception as exc:
                        failures.append({'seed':seed,'mode':mode,'status':'initialization_or_state_failure','phase':phase,'type':type(exc).__name__})
                    finally:
                        if process:
                            try: process.close()
                            except Exception as cleanup_error: failures.append({'seed':seed,'mode':mode,'status':'cleanup_failure','type':type(cleanup_error).__name__})
                            if isinstance(process,PackageExecutor):lifecycles.append({'seed':seed,'mode':mode,'events':process.lifecycle})
                        if adapter:
                            try:
                                health=adapter.get_physics_health();evidence=adapter.get_physics_health_private_evidence()
                                name='health_%d_%s_%d.json'%(index,mode,seed)
                                private_dir=output/'physics_health_private';private_dir.mkdir(exist_ok=True,mode=0o700)
                                path=private_dir/name;path.write_text(encode(evidence),encoding='utf8');path.chmod(0o600)
                                health_reports.append({'task_index':index,'mode':mode,'seed':seed,'summary':health,
                                    'private_evidence':str(path.relative_to(output)),'private_evidence_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
                            except Exception as health_error:
                                health_reports.append({'task_index':index,'mode':mode,'seed':seed,'summary':{'status':'EVIDENCE_UNAVAILABLE','type':type(health_error).__name__}})
                            try: adapter.close()
                            except Exception as cleanup_error: failures.append({'seed':seed,'mode':mode,'status':'cleanup_failure','type':type(cleanup_error).__name__})
        (output/'suite.json').write_text(encode(suite),encoding='utf8')
        recorder.finalize(suite_path=output/'suite.json',output_path=output/'runs.json')
        for name in ('evidence.json','runner_manifest.json'):
            path=output/'artifacts'/name;data=json.loads(path.read_text(encoding='utf8'))
            data.update(source='candidate_runner_v2',runner_version='candidate-v2',simulator_version=getattr(self.factory,'source','SYNTHETIC'))
            for record in data.get('runs',[]):
                record.update(source='candidate_runner_v2',runner_version='candidate-v2',simulator_version=getattr(self.factory,'source','SYNTHETIC'))
            path.write_text(encode(data),encoding='utf8')
        (output/'physics_health_diagnostic.json').write_text(encode(health_reports),encoding='utf8')
        (output/'physical_trace.json').write_text(encode(trace),encoding='utf8')
        (output/'supplemental_verification.json').write_text(encode(supplemental),encoding='utf8')
        (output/'trusted_event_evidence.json').write_text(encode(event_reports),encoding='utf8')
        (output/'skill_lifecycle.json').write_text(encode(lifecycles),encoding='utf8')
        diagnosis={'candidate_status':'NOT_READY','official_score':None,'episodes':failures,
                   'physics_health':health_reports,'physics_health_policy':'independent diagnostics; no automatic scoring or clamp',
                   'source':getattr(self.factory,'source','SYNTHETIC'), 'physical_events':'PHYSX SUBSTEP EVIDENCE WHEN CONFIGURED; semantics pending official approval',
                   'isolation':'namespace_chroot' if self.isolated else 'SECURITY_NOT_READY',
                   'exception_policy':'Frozen V3 has no execution-status field; raw diagnostic may retain auxiliary points or pose success on a timeout. Operational readiness always fails on error/timeout.'}
        valid=len(failures)==len(suite['tasks'])*suite['config']['repeats']*2 and not any(r['status'] in ('initialization_or_state_failure','cleanup_failure') for r in failures)
        diagnosis['gt_access_status']='PASS' if valid else 'FAILED'
        score_name='extension_candidate_score.json' if self.contract_extension else 'frozen_score_diagnostic.json'
        replay_ok=False
        if valid:
            try:
                report=(self.contract_extension or v3).evaluate(suite,{'schema_version':1,'runs':recorder.runs},output)
                (output/score_name).write_text(encode(report),encoding='utf8')
                replay_ok=True
                diagnosis['extension_candidate_score_100' if self.contract_extension else 'raw_frozen_score_100']=report['candidate_score_100' if self.contract_extension else 'score_100']
                if report.get('contract_status')=='FROZEN_AUTHOR_PDF_RULES':
                    diagnosis.update(scoring_status='FROZEN_AUTHOR_PDF_RULES',frozen_rule_score_100=report['frozen_rule_score_100'],candidate_status='EVALUATION_COMPLETED' if all(r['status'] in ('success','failure') for r in failures) else 'EXECUTION_FAILURE')
            except Exception as replay_error:
                diagnosis.update(scoring_status='REFUSED: verifier/scorer replay failed',candidate_status='EXECUTION_FAILURE',error_category='VERIFIER_OR_SCORER_'+type(replay_error).__name__)
        else: diagnosis['scoring_status']='REFUSED: missing trustworthy state'
        (output/'diagnosis.json').write_text(encode(diagnosis),encoding='utf8')
        if replay_ok:
            write_receipt(output,score_name,expected_runs=len(suite['tasks'])*suite['config']['repeats']*2,verifier='trusted_contract_extension' if self.contract_extension else 'frozen_v3')
            gate=ResultGate().check(output)
        else: gate=block(output,'VERIFIER_OR_STATE_INCOMPLETE')
        diagnosis['result_gate']=gate
        return diagnosis
