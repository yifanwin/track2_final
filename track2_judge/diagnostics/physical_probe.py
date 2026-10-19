"""Private single-pair diagnostic on original frozen assets.

Existing selected-case loader owns GT. Skill receives its usual public inputs.
Two runs are explicitly incomplete for official R=5; ResultGate blocks them.
No copied/repeated fabricated trials and no official score are generated.
"""
import argparse,json,os,sys,time,hashlib
from pathlib import Path
from track2_judge.scoring.result_gate import atomic_json,block
from track2_judge.sandbox.package import unpack_skill,PackageExecutor
from track2_judge.runner.recorder import TrustedRecorder
from track2_judge.runner.legacy_util import canonical_json_hash
from track2_judge.scoring.scorer_bridge import v3

class ProbeRunner:
    def __init__(self,factory,*,package_zip,contract_extension,**kwargs):
        self.factory=factory;self.package_zip=package_zip;self.verifier=contract_extension
    def run(self,suite,output,public_inputs):
        out=Path(output).resolve();out.mkdir(mode=0o700,parents=True,exist_ok=False)
        package=unpack_skill(self.package_zip,out/'private_skill_staging')
        assert package.manifest.get('tool_profile')=='piper-joint-v2-candidate'
        task=suite['tasks'][0];config=suite['config'];name=next(p['object'] for p in public_inputs['predicates'] if p['type']=='P2')
        recorder=TrustedRecorder(out/'artifacts',skill_hash=package.package_hash,suite_hash=canonical_json_hash(suite),runs_dir=out)
        results=[];summaries=[]
        for mode in ('single',):   # ★ 2026-10-18 取消 first/final
            adapter=None;process=None;phase='initialization';metrics={'control_calls':0,'peak_lift_m':0.,'bilateral_samples':0,'airborne_samples':0};status='error';category=None
            try:
                adapter=self.factory(task,0,mode);adapter.reset(seed=0,public_task_inputs=public_inputs)
                initial=adapter.get_layout();istate=adapter.get_final_state();z0=istate['objects'][name]['actor_p_m'][2];calls=[]
                process=PackageExecutor(package,out/'skill_work',isolated=True,timeout_s=90)
                process.initialize({'skill_api_version':'ctrc-skill-v1','tool_profile':'piper-joint-v2-candidate','environment':'sim','timeout_s':90,'tool_call_budget':200})
                def dispatch(method,payload):
                    if method not in ('observe','joint_trajectory','piper_gripper_control'):raise ValueError('forbidden tool')
                    result=adapter.dispatch_public(method,payload);calls.append({'method':method,'payload':payload})
                    if method!='observe':
                        metrics['control_calls']+=1
                        if metrics['control_calls']%5==0:
                            state=adapter.get_final_state();lift=state['objects'][name]['actor_p_m'][2]-z0;metrics['peak_lift_m']=max(metrics['peak_lift_m'],lift)
                            fingers=set()
                            for contact in state['contacts']:
                                if name in contact['bodies'] and any(sum(x*x for x in p['impulse_Ns'])>1e-10 for p in contact.get('points',[])):
                                    for body in contact['bodies']:
                                        if body.endswith(('gripper_link1','gripper_link2')):fingers.add(body)
                            if len(fingers)>=2:metrics['bilateral_samples']+=1
                            if len(fingers)>=2 and lift>.04:metrics['airborne_samples']+=1
                    return result
                phase='skill';start=time.monotonic()
                feedback={}
                try:feedback=process.run(adapter.observe(),dispatch,timeout_s=90,max_requests=2400);status='completed'
                except Exception as exc:status='error';category='SKILL_OR_INTERFACE_'+type(exc).__name__
                elapsed=max(time.monotonic()-start,1e-6);phase='verification';final=adapter.get_layout();fstate=adapter.get_final_state();events=adapter.get_event_evidence([])
                verified=self.verifier.verify(task,initial,final,config,event_evidence=events,initial_state=istate,final_state=fstate)
                recorder.record(task_index=0,task=task,seed=0,environment='sim',initial_layout=initial,final_layout=final,elapsed_s=elapsed,event_violations={},verification=verified,tool_calls=calls,status='success' if status=='completed' and verified['success'] else 'failure' if status=='completed' else 'error',error=category)
                health=adapter.get_physics_health()
                metrics['baseline_ik_residual']=feedback.get('max_ik_residual')
                metrics['baseline_note']=feedback.get('baseline')
                results.append({'mode':mode,'execution':status,'error_category':category,'active_wall_s':elapsed,'tool_service_s':process.tool_service_s,'P1':next(p['passed'] for p in verified['predicates'] if p['type']=='P1'),'P2':next(p['passed'] for p in verified['predicates'] if p['type']=='P2'),'C1_violations':verified['violation_count'],'health_flags':sorted(health.get('flag_counts',{})),'health_frames':health.get('frames',0),'nonfinite_frames':health.get('nonfinite_frames',0),**metrics})
                summaries.append({**verified,'elapsed_s':elapsed})
            except Exception as exc:results.append({'mode':mode,'execution':'error','phase':phase,'error_category':type(exc).__name__,**metrics})
            finally:
                if process:process.close()
                if adapter:adapter.close()
        atomic_json(out/'suite.json',suite);recorder.finalize(suite_path=out/'suite.json',output_path=out/'runs.json')
        gate=block(out,'DIAGNOSTIC_ONLY_TWO_RUNS_EXPECTED_TEN')
        result={'public_target_label':name,'results':results,'official_score':None,'gate':gate['status'],'complete_official_repeats':False,'scope':'original frozen Case, legal isolated public-goal Skill, one true first/final pair; not performance evaluation'}
        atomic_json(out/'sanitized_result.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--skill-zip',required=True);p.add_argument('--output',required=True);p.add_argument('--ordinals',default='0,1,2');p.add_argument('--diagnostic-rgb',action='store_true');a=p.parse_args();os.umask(0o077)
    import track2_judge.cli.select_case as selected
    selected.CandidateRunner=ProbeRunner
    phase=Path(a.phase)
    rows=json.loads(lock.read_text())['cases'];out=Path(a.output);out.mkdir(parents=True,exist_ok=False,mode=0o700);results=[]
    for ordinal in map(int,a.ordinals.split(',')):
        r=selected.run(rows[ordinal]['case_id'],a.skill_zip,out/('ordinal_%03d'%ordinal),diagnostic_rgb=a.diagnostic_rgb,casebank=phase/'casebank_frozen_final73')
        results.append({'ordinal':ordinal,**r});atomic_json(out/'sanitized_summary.json',results);print(json.dumps(results[-1]),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
