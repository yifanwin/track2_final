"""Local real process/filesystem isolation tests using synthetic secrets only."""
import argparse, hashlib, json, os, sys
from pathlib import Path
from track2_judge.sandbox.rpc import SkillProcess, encode

def main(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    canary=output/'judge_private_canary.txt';canary.write_text('SYNTHETIC_CANARY_NOT_GT',encoding='utf8')
    before=hashlib.sha256(canary.read_bytes()).hexdigest();captured={};process=None
    canary_paths={};hashes={}
    for name in ('hidden_suite_read','hidden_gt_read','score_read','score_tamper','trace_tamper'):
        p=output/(name+'.json');p.write_text('{"SYNTHETIC_CANARY":true}',encoding='utf8')
        canary_paths[name]=str(p);hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    try:
        os.environ['CANARY_SECRET']='synthetic environment secret'
        process=SkillProcess(Path(__file__).resolve().parents[1] / 'track2_judge' / 'skills' / 'security_probe.py',output/'work',isolated=True)
        def dispatch(method,payload):
            if method!='security_report':raise ValueError('method')
            captured.update(payload);return {'accepted':True}
        process.run({'canary_path':str(canary),'canary_paths':canary_paths},dispatch,timeout_s=20)
        assert captured and all(v=='PASS' for v in captured.values()),captured
        assert hashlib.sha256(canary.read_bytes()).hexdigest()==before
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==hashes[n] for n,p in canary_paths.items())
        result={'status':'PASS','checks':captured,'canary_unchanged':True,
                'scope':'local WSL synthetic adversary; no real contestant; not a full independent security certification',
                'architecture':'user/mount/network/PID namespaces + chroot + dropped caps + no_new_privs + seccomp + bounded stdio RPC',
                'production_readiness':'NOT_READY: visual observation/public schema confirmation and independent security review still required'}
    except Exception as exc:result={'status':'FAIL','error':type(exc).__name__,'security':'SECURITY_NOT_READY'}
    finally:
        if process:process.close()
        os.environ.pop('CANARY_SECRET',None)
    (output/'security_result.json').write_text(encode(result),encoding='utf8')
    print(json.dumps(result));return 0 if result['status']=='PASS' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();sys.exit(main(args.output))
