"""Additional actual isolated synthetic Skill probes. No real submissions."""
import argparse, hashlib, json, os, resource, time, zipfile
from pathlib import Path
from track2_judge.sandbox.package import unpack_skill, PackageExecutor

PROBE='''import importlib,os,sys,inspect,resource,errno
def initialize(context):pass
def shutdown(context):pass
def run(observation,tools):
    c={}
    for name in ('mani_skill','sapien','adapter','frozen_scorer','physics_health'):
        try:importlib.import_module(name);c['import_'+name]='FAIL'
        except ImportError:c['import_'+name]='PASS'
    for label,path in observation['canaries'].items():
        for mode in ('r','w'):
            try:open(path,mode);c[label+'_'+mode]='FAIL'
            except OSError:c[label+'_'+mode]='PASS'
    c['host_shell_absent']='PASS' if not os.path.exists('/bin/sh') else 'FAIL'
    c['proc_absent']='PASS' if not os.path.exists('/proc') else 'FAIL'
    c['host_config_absent']='PASS' if not os.path.exists('/etc/passwd') else 'FAIL'
    c['only_clean_environment']='PASS' if set(os.environ)<= {'PATH','HOME','TMPDIR','PYTHONIOENCODING','LC_CTYPE'} else 'FAIL'
    # Reflection inside the child does not recover judge modules or actors.
    frames=inspect.stack();seen=[str(list(f.frame.f_globals.keys())) for f in frames]
    c['judge_modules_absent']='PASS' if all(x not in sys.modules for x in ('adapter','backend','sapien','mani_skill')) else 'FAIL'
    c['runtime_readonly']='FAIL'
    try:open('/usr/lib/python%d.%d/json/__init__.py'%sys.version_info[:2],'w')
    except OSError as e:c['runtime_readonly']='PASS' if e.errno in (errno.EROFS,errno.EACCES) else 'FAIL'
    try:bytearray(300*1024*1024);c['memory_limit']='FAIL'
    except MemoryError:c['memory_limit']='PASS'
    try:
        with open('/tmp/large','wb') as f:f.write(b'x'*(2*1024*1024))
        c['file_limit']='FAIL'
    except OSError as e:c['file_limit']='PASS' if e.errno==errno.EFBIG else 'FAIL'
    handles=[]
    try:
        for _ in range(64):handles.append(open('/dev/null','rb'))
        c['fd_limit']='FAIL'
    except OSError as e:c['fd_limit']='PASS' if e.errno==errno.EMFILE else 'FAIL'
    finally:
        for f in handles:f.close()
    c['core_disabled']='PASS' if resource.getrlimit(resource.RLIMIT_CORE)==(0,0) else 'FAIL'
    return {'status':'completed','checks':c,'cpu_limits':resource.getrlimit(resource.RLIMIT_CPU)}
'''

def package(root,name,code):
    path=root/(name+'.zip')
    with zipfile.ZipFile(path,'x') as z:
        z.writestr('manifest.json',json.dumps({'skill_name':'internal_'+name,'version':'2','entrypoint':'main.py','api_version':'ctrc-skill-v1'}))
        z.writestr('main.py',code)
    return unpack_skill(path,root/(name+'_staging'))

def main(output):
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False);os.umask(0o077)
    canaries={};hashes={}
    for name in ('private_health','score_code','hidden_case','judge_environment'):
        p=root/(name+'.json');p.write_text('SYNTHETIC_ONLY');canaries[name]=str(p);hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    result={'scope':'synthetic malicious packages, real server namespaces; not exhaustive security certification','checks':{}}
    p=None
    try:
        p=PackageExecutor(package(root,'extended',PROBE),root/'probe_work',isolated=True,timeout_s=10)
        p.initialize({});r=p.run({'canaries':canaries},lambda *_:None,timeout_s=12);p.close()
        result['checks'].update(r['checks']);assert all(v=='PASS' for v in r['checks'].values()),r
        assert all(hashlib.sha256(Path(v).read_bytes()).hexdigest()==hashes[k] for k,v in canaries.items())
        result['checks']['canary_hashes_unchanged']='PASS'
        # Actual CPU quota: trusted synthetic infinite loop, not just getrlimit.
        busy='def initialize(context):pass\ndef shutdown(context):pass\ndef run(observation,tools):\n    while True:pass\n'
        p=PackageExecutor(package(root,'cpu_burn',busy),root/'cpu_work',isolated=True,timeout_s=10);p.initialize({})
        start=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
        try:p.run({},lambda *_:None,timeout_s=45);raise AssertionError('CPU burn returned')
        except RuntimeError:pass
        code=p.process.p.poll();p.abort();after=resource.getrusage(resource.RUSAGE_CHILDREN)
        elapsed=time.monotonic()-start;cpu=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime)
        assert code not in (None,0) and 27<=cpu<=34 and elapsed<45,(code,cpu,elapsed)
        result['checks']['actual_cpu_limit']='PASS';result['cpu_evidence']={'exit':code,'cpu_s':cpu,'wall_s':elapsed,'configured_s':30}
        result['status']='PASS'
    except Exception as e:result.update(status='FAIL',error=type(e).__name__)
    finally:
        if p:p.close()
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));return int(result['status']!='PASS')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);raise SystemExit(main(p.parse_args().output))
