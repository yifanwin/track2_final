"""New extension/casebank isolation tests with synthetic canaries only."""
import argparse,hashlib,json,os,zipfile
from pathlib import Path
from track2_judge.sandbox.package import unpack_skill,PackageExecutor
from track2_judge.toolapi.dispatch import ManiSkillSimulatorAdapter
from track2_judge.scenario.prototype_env_v2 import make_env
from track2_judge.scenario.prototype_backend import physical_snapshot

CODE='''import importlib,os
def initialize(context):pass
def shutdown(context):pass
def run(observation,tools):
    c={}
    for name in ('contract_extension','case_backend','run_selected_candidate','runner','sapien'):
        try:importlib.import_module(name);c['import_'+name]='FAIL'
        except ImportError:c['import_'+name]='PASS'
    for name,path in observation['synthetic_paths'].items():
        for mode in ('r','w'):
            try:open(path,mode);c[name+'_'+mode]='FAIL'
            except OSError:c[name+'_'+mode]='PASS'
    return {'status':'completed','checks':c,'candidate_score_100':9999,'final_state':'FAKE_SKILL_CLAIM'}
'''

def package(root,name,code):
    p=root/(name+'.zip')
    with zipfile.ZipFile(p,'w') as z:
        z.writestr('manifest.json',json.dumps({'skill_name':'internal_'+name,'version':'2','entrypoint':'main.py','api_version':'ctrc-skill-v1'}));z.writestr('main.py',code)
    return unpack_skill(p,root/(name+'_unpacked'))

def main(output):
    root=Path(output).resolve();root.mkdir(parents=True,mode=0o700,exist_ok=False);os.umask(0o077);paths={};hashes={};checks={}
    for name in ('other_case_GT','hidden_initial_pose','hidden_final_pose','private_casebank','private_freeze_manifest','scorer_internal_state','extension_evidence','extension_score'):
        p=root/(name+'.json');p.write_text('SYNTHETIC_CANARY_ONLY');paths[name]=str(p);hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    p=PackageExecutor(package(root,'canary',CODE),root/'canary_work',isolated=True);p.initialize({})
    try:r=p.run({'synthetic_paths':paths},lambda *_:None);assert all(x=='PASS' for x in r['checks'].values());checks.update(r['checks'])
    finally:p.close()
    assert all(hashlib.sha256(Path(v).read_bytes()).hexdigest()==hashes[k] for k,v in paths.items());checks['canaries_unchanged']='PASS'
    a=ManiSkillSimulatorAdapter(make_env,snapshot=physical_snapshot,max_steps=2);a.reset(seed=0);before=json.dumps(a.get_final_state(),sort_keys=True)
    try:
        for method in ('set_pose','translate','rotate','get_layout','get_final_state','get_physics_health','get_initialization_evidence','score_callback'):
            code="def initialize(context):pass\ndef shutdown(context):pass\ndef run(observation,tools):\n    tools.call("+repr(method)+",{})\n    return {'status':'completed'}\n"
            p=PackageExecutor(package(root,method,code),root/(method+'_work'),isolated=True);p.initialize({})
            try:
                try:p.run({},a.dispatch_public)
                except ValueError:checks['RPC_'+method]='PASS'
                else:raise AssertionError('private RPC accepted')
            finally:p.abort()
        assert before==json.dumps(a.get_final_state(),sort_keys=True);checks['physical_state_unchanged']='PASS'
    finally:a.close()
    result={'status':'PASS','checks':checks,'scope':'actual isolated synthetic packages, extension/casebank canaries and real Piper state; not exhaustive security certification','official_callback_sent':False}
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args().output)
