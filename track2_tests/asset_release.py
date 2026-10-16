import hashlib,json,struct,sys
from pathlib import Path
from track2_judge.scoring.asset_release_gate import check_glb,publish_copy

def main(output):
    r=Path(output).resolve();r.mkdir(parents=True,exist_ok=False)
    def glb(name,meta):
        blob=json.dumps(meta).encode();blob+=b' '*((-len(blob))%4)
        raw=struct.pack('<4sII',b'glTF',2,20+len(blob))+struct.pack('<I4s',len(blob),b'JSON')+blob
        p=r/name;p.write_bytes(raw);return p
    p=glb('synthetic.glb',{'asset':{'version':'2.0'},'nodes':[{'name':'public_synthetic'}]})
    approval={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'local_geometry_reviewed':True,'texture_reviewed':True,'approved_public_name':'public_synthetic','geometry_provenance':'unbaked_actor_local'}
    checks={}
    publish_copy(p,approval,r/'approved_copy.glb');checks['synthetic_approved_copy']='PASS'
    for name,meta in [('extras',{'extras':{'target_pose':[0,0,0]}}),('transform',{'nodes':[{'translation':[1,2,3]}]}),('path',{'buffers':[{'uri':'/mnt/private/answer.bin'}]})]:
        q=glb(name+'.glb',{'asset':{'version':'2.0'},**meta});a={**approval,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()}
        try:check_glb(q,a);raise AssertionError(name)
        except ValueError:checks[name+'_rejected']='PASS'
    try:check_glb(p,{});raise AssertionError('missing independent review')
    except ValueError:checks['missing_approval_rejected']='PASS'
    result={'status':'PASS','checks':checks,'real_asset_publication':'NOT TESTED/NOT AUTHORIZED: no real scene assets released','note':'metadata scanning alone cannot detect baked geometry or image answers'}
    (r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main(sys.argv[1])
