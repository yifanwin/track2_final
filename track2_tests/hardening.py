"""Meaningful negative release gates; synthetic data only, outputs stay private."""
import argparse
import copy
import json
import tempfile
import time
import zipfile
from pathlib import Path
from track2_judge.scoring.result_gate import ResultGate, write_receipt, atomic_json, block
from track2_judge.observation.input_adapter import InputAdapter, validate_envelope, SCHEMA
from track2_judge.runner.evidence import legacy_callback_payload

def fixture(root):
    root=Path(root);root.mkdir(mode=0o700)
    layout={'objects':[{'instance':'SYNTHETIC_BLOCK','position':[0,0,0],'rotation':[0,0,0,1]}]}
    task={'task_id':'SYNTHETIC_ONLY','target_layout':layout}
    runs=[];evidence=[];episodes=[]
    # ★ 2026-10-18 取消 first/final：每 seed **1** 条 run（不再 `first`/`final` 两条），
    #   落盘记录里没有 `mode`、也没有 `plan_revisions`（见 runner/recorder.py）。
    for seed in range(5):
        ident=str(seed);initial='artifacts/states/'+ident+'_initial.json';final='artifacts/states/'+ident+'_final.json'
        atomic_json(root/initial,layout);atomic_json(root/final,layout)
        runs.append({'run_id':ident,'task_id':task['task_id'],'seed':seed,'initial_layout':initial,'final_layout':final,'elapsed_s':1.})
        evidence.append({'run_id':ident,'status':'failure','error':None,'verification':{'status':'failure','predicates':[{'status':'failure'}]}})
        episodes.append({'status':'failure'})
    for n,d in {'suite.json':{'tasks':[task],'config':{'repeats':5}},'runs.json':{'runs':runs},
                'artifacts/evidence.json':{'runs':evidence},'artifacts/runner_manifest.json':{'run_count':5},
                'diagnosis.json':{'episodes':episodes,'gt_access_status':'PASS'},'frozen_score_diagnostic.json':{'score_100':30.}}.items():atomic_json(root/n,d)
    write_receipt(root,'frozen_score_diagnostic.json',expected_runs=5,verifier='frozen_v3')
    return root

def main(output):
    out=Path(output).resolve();out.mkdir(mode=0o700,parents=True,exist_ok=False);checks={}
    good=fixture(out/'valid_verified_task_failure');gate=ResultGate();assert gate.check(good)['publishable'];checks['verified_task_failure_preserves_frozen_score']='PASS'
    faults={
        'recorder_missing':lambda r:(r/'artifacts/evidence.json').unlink(),
        'receipt_missing':lambda r:(r/'verification_receipt.json').unlink(),
        'record_tamper':lambda r:atomic_json(r/'runs.json',{'runs':[]}),
        'state_missing':lambda r:(r/'artifacts/states/0_final.json').unlink(),
        'verifier_failed':lambda r:atomic_json(r/'verification_receipt.json',{'verified':False}),
        'runtime_exception':lambda r:atomic_json(r/'diagnosis.json',{'episodes':[{'status':'timeout'}]*5,'gt_access_status':'PASS'}),
        'gt_read_exception':lambda r:atomic_json(r/'diagnosis.json',{'episodes':[{'status':'failure'}]*5,'gt_access_status':'FAILED'}),
        'nonfinite_state':lambda r: (r/'artifacts/states/0_final.json').write_text('{"objects":[{"instance":"SYNTHETIC_BLOCK","position":[NaN,0,0],"rotation":[0,0,0,1]}]}'),
        'score_tamper':lambda r:atomic_json(r/'frozen_score_diagnostic.json',{'score_100':100}),
    }
    for name,mutate in faults.items():
        r=fixture(out/name);mutate(r);result=gate.check(r)
        assert not result['publishable'] and result['score_100'] is None and result['reasons'];checks[name]='PASS'
    # 1.5/D3：评分器哈希门禁已删 —— 该检查不再适用（发布期由流程保证）。
    for diagnostic in ({'official_score':30,'official_approval':True,'candidate_status':'EXECUTION_FAILURE','episodes':[{'status':'timeout'}]}, {'official_score':30,'official_approval':True}):
        try:legacy_callback_payload('SYNTHETIC',diagnostic)
        except ValueError:pass
        else:raise AssertionError('ungated publication')
    checks['old_approved_failure_counterexample_blocked']='PASS'
    result=gate.check(good);payload=legacy_callback_payload('SYNTHETIC',{'official_score':30.,'official_approval':True},verified_output=good);assert payload['score']==30
    checks['explicit_approved_verified_score_only']='PASS'
    import track2_judge.scoring.result_gate as gate_module
    import io,contextlib
    writer=gate_module.atomic_json
    try:
        def storage_failure(*_):raise OSError('SYNTHETIC_DISK_FAILURE')
        gate_module.atomic_json=storage_failure;logged=io.StringIO()
        with contextlib.redirect_stderr(logged):failed_storage=gate.check(good)
        assert not failed_storage['publishable'] and failed_storage['score_100'] is None and 'ARTIFACT_STORAGE_FAILURE' in failed_storage['reasons'] and 'ARTIFACT_STORAGE_FAILURE' in logged.getvalue()
        checks['storage_failure_blocks_score_and_logs_reason']='PASS'
    finally:gate_module.atomic_json=writer
    try:legacy_callback_payload('SYNTHETIC',{'official_score':30.,'official_approval':True,'candidate_status':'EXECUTION_FAILURE'},verified_output=good)
    except ValueError:checks['approved_runtime_failure_with_valid_artifact_still_blocked']='PASS'
    else:raise AssertionError('approved runtime failure accepted')
    # Slow trusted tool service must not consume contestant computation budget.
    from track2_judge.sandbox.package import unpack_skill,PackageExecutor
    code="def initialize(context):pass\ndef shutdown(context):pass\ndef run(observation,tools):\n    for _ in range(10):tools.observe()\n    return {'status':'completed'}\n"
    skill=out/'latency_skill.zip'
    with zipfile.ZipFile(skill,'x') as z:
        z.writestr('manifest.json',json.dumps({'skill_name':'SYNTHETIC_LATENCY','version':'1','api_version':'ctrc-skill-v1','entrypoint':'main.py'}));z.writestr('main.py',code)
    package=unpack_skill(skill,out/'latency_staging');process=PackageExecutor(package,out/'latency_work',isolated=False,trusted=True,timeout_s=5)
    try:
        process.initialize({})
        def slow_service(*_):time.sleep(.05);return {'source':'SYNTHETIC'}
        answer=process.run({},slow_service,timeout_s=.2,max_requests=10)
        assert answer['status']=='completed' and process.tool_service_s>=.5
        checks['trusted_tool_latency_excluded_from_compute_deadline']='PASS'
    finally:process.close()
    for mode,payload in [('A',{'interpreted_text':'public OS output','detections':[{'label':'cup','bbox_xyxy':[.1,.1,.2,.2],'confidence':.8}]}),('B',{'text_instruction':'public raw text'})]:
        a=InputAdapter(mode=mode,producer=lambda _,p=payload:p);assert a.observation({})['mode']==mode
        for key in ('hidden_gt','initial_pose','final_pose','manifest','scorer_state','host_path'):
            try:validate_envelope({'schema':SCHEMA,'mode':mode,'payload':{**payload,key:{'data':[0,0,0]}}})
            except ValueError:pass
            else:raise AssertionError('private input accepted')
    checks['both_input_modes_no_default_and_private_field_rejection']='PASS'
    result={'status':'PASS','checks':checks,'scope':'synthetic release gates, no network or GT','official_score':None}
    atomic_json(out/'result.json',result);print(json.dumps(result))
    return 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);raise SystemExit(main(p.parse_args().output))
