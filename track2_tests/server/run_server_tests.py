"""Authorized private CPU validation only. No installation, GPU, network or production."""
import datetime,hashlib,json,os,pathlib,signal,subprocess,sys,time
ROOT=pathlib.Path(os.environ['TRACK2_STAGING_ROOT']).resolve()
work=pathlib.Path(sys.argv[1]).resolve(strict=True)
assert work.is_relative_to(ROOT) and work!=ROOT and work.stat().st_uid==os.getuid()
os.umask(0o077)
source=work/'source/Track2_ManiSkill_Evaluator_Candidate_V2'
assert source.is_dir()
out=work/'evidence';out.mkdir(mode=0o700,exist_ok=False)
cache=work/'private_cache';cache.mkdir(mode=0o700)
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
           TMPDIR=str(cache),XDG_CACHE_HOME=str(cache),MESA_SHADER_CACHE_DIR=str(cache),PYTHONNOUSERSITE='1')
os.nice(10)
smoke="""import json,numpy as np
from track2_judge.scenario.prototype_env_v2 import make_env
from track2_judge.toolapi.dispatch import ManiSkillSimulatorAdapter
from track2_judge.scenario.prototype_backend import physical_snapshot
a=ManiSkillSimulatorAdapter(make_env,snapshot=physical_snapshot,max_steps=2)
try:
    a.reset(seed=0)
    q=np.asarray(a.get_final_state()['qpos'],dtype=float)
    assert q.shape[-1]==8 and np.isfinite(q).all()
    a.dispatch_public('piper_gripper_control',{'gripper_opening_m':float(q[6]-q[7])})
    s=a.get_final_state()
    assert np.isfinite(np.asarray(s['qpos'],dtype=float)).all()
    print(json.dumps({'status':'PASS','scope':'actual CPU Piper reset and one checked gripper action','production_ready':False}))
finally:a.close()
"""
commands=[
 ('install_check',['check_install.py'],60,None),
 ('piper_smoke',['-c',smoke],120,None),
 ('v1_regression',['tests.py','--isolated','--output',str(out/'v1_regression')],240,None),
 ('v2_functional',['v2_tests.py','--isolated','--output',str(out/'functional')],300,None),
 ('sandbox_security',['security_test.py','--output',str(out/'security')],120,None),
 ('package_security',['package_security_test.py','--output',str(out/'package_security')],120,None),
 ('real_physics',['v2_physics_test.py','--output',str(out/'physics')],360,None),
 ('non_target_drop',['v2_physics_test.py','--no-events','--non-target-drop','--output',str(out/'non_target_drop')],360,None),
 ('missing_evidence',['v2_physics_test.py','--instruction','noop','--output',str(out/'missing_evidence')],240,None),
 ]
results=[]
for name,args,limit,_ in commands:
    start=time.monotonic();argv=[sys.executable,'-B',*args]
    print(json.dumps({'event':'START','test':name,'limit_s':limit}),flush=True)
    with (out/(name+'.log')).open('w') as log:
        p=subprocess.Popen(argv,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=limit);status='PASS' if code==0 else 'FAIL'
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL);p.wait();code=p.returncode;status='FAIL: TIMEOUT'
    r={'test':name,'status':status,'exit_code':code,'wall_s':round(time.monotonic()-start,3),'command':argv,'log':name+'.log'}
    results.append(r)
    (out/'execution_results.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(r),flush=True)
    if name in ('install_check','manifest_before') and status!='PASS':break
summary={'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tests':results,
         'real_camera':{'status':'NOT TESTED','reason':'No device-specific GPU authorization; no verified CPU Vulkan device'},
         'overall':'NOT_READY','production_deployment':False,'scope':'internal synthetic Skill/canaries only; no production data or callbacks',
         'archive_sha256':hashlib.sha256((work/'candidate_v2.zip').read_bytes()).hexdigest()}
(out/'server_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({'event':'COMPLETE','overall':'NOT_READY','tests':len(results)}),flush=True)
