"""Judge-only lazy case loading. Never mount this repository into a Skill.

No official TTG78 task conversion is implied: source final-layout material is
not a validated initial/goal task pair. Approved public files are independent
snapshots pinned in judge policy, never derived from hidden task dictionaries.
"""
import hashlib,json,os,re,stat
from pathlib import Path
from track2_judge.observation.provider import ObservationProvider

def read_json(path,limit=8*1024**2):
    if path.stat().st_size>limit:raise ValueError('trusted data budget rejected')
    return json.loads(path.read_text(encoding='utf8'),parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite trusted data')))

class TrustedCaseBank:
    def __init__(self,root,policy):
        self.root=Path(root).resolve(strict=True)
        if os.name!='posix':raise RuntimeError('Linux private case repository required')
        mode=self.root.stat()
        if mode.st_uid!=os.getuid() or stat.S_IMODE(mode.st_mode)&0o077:raise PermissionError('private repository must be judge-owned 0700')
        # Policy is judge configuration, never taken from a Skill or case file.
        self.policy=json.loads(json.dumps(policy,allow_nan=False));self.loaded=[]
    def file(self,relative,digest):
        rel=Path(relative)
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('trusted path rejected')
        p=self.root/rel
        if any(x.is_symlink() for x in [p,*list(p.parents)[:len(rel.parts)-1]]):raise ValueError('trusted symlink rejected')
        p=p.resolve(strict=True)
        if self.root not in p.parents or not p.is_file():raise ValueError('trusted containment rejected')
        if not re.fullmatch('[0-9a-f]{64}',digest):raise ValueError('trusted digest rejected')
        h=hashlib.sha256()
        with p.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk)
        if h.hexdigest()!=digest:raise ValueError('trusted integrity rejected')
        return p
    def load(self,case_id):
        if not re.fullmatch('[A-Za-z0-9_-]{1,96}',case_id):raise ValueError('opaque case id rejected')
        approved=self.policy['cases'][case_id]
        private=read_json(self.file(approved['task_file'],approved['task_sha256']))
        # Only this case's asset hashes are touched. No directory/suite traversal.
        assets=[self.file(a['file'],a['sha256']) for a in approved.get('assets',[])]
        if len(assets)>512:raise ValueError('asset budget rejected')
        fields=approved.get('public_fields',[])
        public=read_json(self.file(approved['public_file'],approved['public_sha256']),2*1024**2) if fields else {}
        public=ObservationProvider(approved_fields=fields).prepare(public)
        self.loaded.append(case_id)
        return {'task':private,'asset_paths':assets,'public_inputs':public}

def run_selected_case(bank,case_id,runner_factory,config,output):
    """One selected case, original scorer and repeat contract; no platform callback.

    factory receives private case only in the judge process. Score output stays
    private. Multi-case official scheduling requires approved task definitions.
    """
    case=bank.load(case_id);runner=runner_factory(case)
    return runner.run({'config':config,'tasks':[case['task']]},output,case['public_inputs'])
