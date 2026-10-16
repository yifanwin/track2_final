"""Synthetic-only case repository and independent public approval tests."""
import argparse,hashlib,json,os
from pathlib import Path
from track2_judge.contract.trusted_data import TrustedCaseBank

def main(output):
    root=Path(output).resolve();root.mkdir(parents=True,mode=0o700,exist_ok=False);root.chmod(0o700)
    def put(name,value):
        p=root/name;p.write_text(json.dumps(value));p.chmod(0o600);return hashlib.sha256(p.read_bytes()).hexdigest()
    task=put('task.json',{'SYNTHETIC_PRIVATE_GT':'never public','initial_layout':'synthetic'})
    public=put('public.json',{'text_instruction':'Synthetic control interface test'})
    asset=put('asset.json',{'synthetic_geometry':True})
    policy={'cases':{'case0':{'task_file':'task.json','task_sha256':task,'public_file':'public.json','public_sha256':public,'public_fields':['text_instruction'],'assets':[{'file':'asset.json','sha256':asset}]},'untouched':{'task_file':'missing.json','task_sha256':'0'*64}}}
    bank=TrustedCaseBank(root,policy);case=bank.load('case0');checks={}
    assert bank.loaded==['case0'] and 'SYNTHETIC_PRIVATE_GT' not in json.dumps(case['public_inputs']);checks['single_case_no_private_projection']='PASS'
    assert len(case['asset_paths'])==1;checks['per_case_assets']='PASS'
    for name,action in [
        ('traversal',lambda:bank.file('../outside',task)),
        ('wrong_digest',lambda:bank.file('task.json','0'*64)),
        ('hidden_input_rejected',lambda:__import__('track2_judge.observation.provider', fromlist=['x']).ObservationProvider(approved_fields=['text_instruction']).prepare({'initial_layout':{}}))]:
        try:action();raise AssertionError(name)
        except ValueError:checks[name]='PASS'
    (root/'link.json').symlink_to(root/'task.json')
    try:bank.file('link.json',task);raise AssertionError('symlink')
    except ValueError:checks['symlink']='PASS'
    (root/'public.json').write_text('{"text_instruction":"changed"}')
    try:bank.load('case0');raise AssertionError('approval modified')
    except ValueError:checks['public_approval_hash']='PASS'
    root.chmod(0o755)
    try:TrustedCaseBank(root,policy);raise AssertionError('permissions')
    except PermissionError:checks['private_permissions']='PASS'
    finally:root.chmod(0o700)
    result={'status':'PASS','checks':checks,'official78_execution':'NOT TESTED: approved task definitions and physical asset conversion required'}
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args().output)
