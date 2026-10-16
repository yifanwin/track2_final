"""One CPU acceptance command. Real rendering requires separate device approval.

Synthetic tests create private evidence under a new directory, never publish
those raw traces. No platform client, queue, production suite or GT is used.
"""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path

def main(output):
    here=Path(__file__).resolve().parent
    final=here.parent
    out=Path(output).resolve();out.mkdir(parents=True,mode=0o700,exist_ok=False);out.chmod(0o700)
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(final))
    tests=[('install',[str(final/'track2_judge/diagnostics/check_install.py')],60),
           ('health',[str(here/'health_contract.py'),str(final),str(out/'health')],120),
           ('v1',[str(here/'v1_regression.py'),'--isolated','--output',str(out/'v1')],240),
           ('v2',[str(here/'functional.py'),'--isolated','--output',str(out/'v2')],300),
           ('security',[str(here/'security.py'),'--output',str(out/'security')],120),
           ('package_security',[str(here/'package_security.py'),'--output',str(out/'package_security')],120),
           ('extended_security',[str(here/'acceptance_security.py'),'--output',str(out/'extended_security')],90),
           ('trusted_data',[str(here/'trusted_data.py'),'--output',str(out/'trusted_data')],60),
           ('physics',[str(final/'track2_judge/scenario/prototype_env_v2.py'),'--output',str(out/'physics')],240),
           ('non_target_drop',[str(final/'track2_judge/scenario/prototype_env_v2.py'),'--no-events','--non-target-drop','--output',str(out/'drop')],240),
           ('missing_evidence',[str(final/'track2_judge/scenario/prototype_env_v2.py'),'--instruction','noop','--output',str(out/'missing')],120),
           ('asset_release',[str(here/'asset_release.py'),str(out/'asset_release')],60)]
    results=[]
    for name,args,limit in tests:
        start=time.monotonic()
        with (out/(name+'.log')).open('w') as log:
            p=subprocess.Popen([sys.executable,'-B',*args],cwd=final,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=p.wait(timeout=limit);status='PASS' if code==0 else 'FAIL'
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();code=p.returncode;status='FAIL:TIMEOUT'
        r={'test':name,'status':status,'exit':code,'wall_s':round(time.monotonic()-start,3),'script':args[0]};results.append(r)
        (out/'execution.json').write_text(json.dumps(results,indent=2));print(json.dumps(r),flush=True)
    return int(any(r['status']!='PASS' for r in results))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);raise SystemExit(main(p.parse_args().output))
