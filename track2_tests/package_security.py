"""Actual isolated whole-package mounts and original lifecycle security probe."""
import argparse,hashlib,json,sys,time,zipfile
from pathlib import Path
from track2_judge.sandbox.rpc import encode
from track2_judge.sandbox.package import PackageExecutor,unpack_skill

CODE='''import os
def initialize(context):pass
def run(observation,tools):
    checks={}
    for label,path in observation['synthetic_canaries'].items():
        try:
            with open(path,'r') as f:f.read()
            checks[label]='FAIL'
        except OSError:checks[label]='PASS'
    for label,path in [('package_modify','/skill/main.py'),('host_modify','/host/main.py')]:
        try:
            with open(path,'w') as f:f.write('tamper')
            checks[label]='FAIL'
        except OSError:checks[label]='PASS'
    try:
        with open('/proc/1/root','r') as f:f.read()
        checks['proc_escape']='FAIL'
    except OSError:checks['proc_escape']='PASS'
    checks['private_environment']='PASS' if 'CANARY_SECRET' not in os.environ else 'FAIL'
    return {'status':'completed','checks':checks}
def shutdown(context):pass
'''

def main(output):
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True);paths={};digests={}
    for name in ('hidden_gt','hidden_suite','score','trajectory'):
        p=root/(name+'_synthetic_canary.json');p.write_text('{"SYNTHETIC_CANARY":true}',encoding='utf8');paths[name]=str(p);digests[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    source=root/'probe.zip'
    with zipfile.ZipFile(source,'w') as z:
        z.writestr('manifest.json',json.dumps({'skill_name':'internal_package_security_probe','version':'2','entrypoint':'main.py','api_version':'ctrc-skill-v1'}));z.writestr('main.py',CODE)
    package=unpack_skill(source,root/'unpacked');process=None
    try:
        import os
        os.environ['CANARY_SECRET']='synthetic_secret'
        process=PackageExecutor(package,root/'work',isolated=True,timeout_s=15);process.initialize({'skill_api_version':'ctrc-skill-v1'})
        result=process.run({'synthetic_canaries':paths},lambda *_:(_ for _ in ()).throw(ValueError('no tools permitted')),timeout_s=10)
        assert len(result['checks'])==8 and all(v=='PASS' for v in result['checks'].values())
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digests[n] for n,p in paths.items())
        process.close();assert process.lifecycle==['ready','initialize','run','shutdown']
        final={'status':'PASS','checks':result['checks'],'original_manifest_lifecycle':process.lifecycle,'canaries_unchanged':True,'scope':'synthetic adversary inside actual read-only package/host mounts; not security certification'}
    except Exception as exc:final={'status':'FAIL','error':type(exc).__name__}
    finally:
        if process:process.close()
        import os
        os.environ.pop('CANARY_SECRET',None)
    (root/'package_security_result.json').write_text(encode(final),encoding='utf8');print(json.dumps(final));return int(final['status']!='PASS')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();sys.exit(main(a.output))
