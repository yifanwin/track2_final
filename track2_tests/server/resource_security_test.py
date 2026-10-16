"""Internal synthetic package probes: actual memory/file/fd denial."""
import json,os,sys,zipfile
from pathlib import Path
WORK=Path(sys.argv[1]).resolve(strict=True)
ROOT=Path(os.environ['TRACK2_STAGING_ROOT']).resolve(strict=True)
assert WORK.is_relative_to(ROOT) and WORK.stat().st_uid==os.getuid()
sys.path.insert(0,str(WORK/'source/Track2_ManiSkill_Evaluator_Candidate_V2'));os.umask(0o077)
from track2_judge.sandbox.package import unpack_skill,PackageExecutor
out=WORK/'resource_security';out.mkdir(mode=0o700,exist_ok=False)
source='''import errno,resource
def initialize(context):pass
def run(observation,tools):
    checks={}
    try:memory=bytearray(300*1024*1024);checks['memory_oversize']='FAIL'
    except MemoryError:checks['memory_oversize']='PASS'
    try:
        with open('/tmp/oversized','wb') as f:f.write(b'x'*(2*1024*1024))
        checks['file_oversize']='FAIL'
    except OSError as e:checks['file_oversize']='PASS' if e.errno==errno.EFBIG else 'FAIL'
    files=[]
    try:
        for _ in range(64):files.append(open('/dev/null','rb'))
        checks['fd_exhaustion']='FAIL'
    except OSError as e:checks['fd_exhaustion']='PASS' if e.errno==errno.EMFILE else 'FAIL'
    finally:
        for f in files:f.close()
    checks['core_disabled']='PASS' if resource.getrlimit(resource.RLIMIT_CORE)==(0,0) else 'FAIL'
    return {'status':'completed','checks':checks,'cpu_limit_config':resource.getrlimit(resource.RLIMIT_CPU)}
def shutdown(context):pass
'''
path=out/'resource_skill.zip'
with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as z:
    z.writestr('resource/manifest.json',json.dumps({'skill_name':'internal_resource_limits','version':'2.0-candidate','entrypoint':'main.py','api_version':'ctrc-skill-v1'}));z.writestr('resource/main.py',source)
package=unpack_skill(path,out/'staging');process=None
try:
    process=PackageExecutor(package,out/'work',isolated=True,timeout_s=10);process.initialize({'skill_api_version':'ctrc-skill-v1'})
    r=process.run({},lambda *_: (_ for _ in ()).throw(ValueError('no RPC')),timeout_s=10)
    assert len(r['checks'])==4 and all(v=='PASS' for v in r['checks'].values()),r
    process.close();result={'status':'PASS','checks':r['checks'],'cpu_limit_config':r['cpu_limit_config'],'cpu_quota_exhaustion':'NOT TESTED','wall_timeout':'covered by isolated v1/v2 suite','scope':'actual isolated synthetic package; not full security certification'}
except Exception as e:result={'status':'FAIL','error':type(e).__name__}
finally:
    if process is not None:process.close()
(WORK/'evidence/resource_security_sanitized.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
