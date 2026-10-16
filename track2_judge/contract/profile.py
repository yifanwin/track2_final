"""User-confirmed author-PDF scoring scope; no production authorization."""
PROFILE='track2-ttg73-author-pdf-final-state-v1'

def load_config():
    import json
    from pathlib import Path
    from track2_judge.scoring.scorer_bridge import v3
    data=json.loads(Path(__file__).with_name('scoring_config.json').read_text())
    if data['profile']!=PROFILE or data['tiers']!={t:v3.default_config('sim',t) for t in v3.TIERS}:raise ValueError('frozen scoring configuration mismatch')
    return data['tiers']['standard']

def validate(task):
    if task.get('scoring_profile')!=PROFILE:raise ValueError('frozen 73 scoring profile required')
    if task.get('task_type')!='pick_place':raise ValueError('73 scope is single-arm pick/place')
    predicates=task.get('predicates',[])
    if len(predicates)!=2 or {p['type'] for p in predicates}!={'P1','P2'}:raise ValueError('exact P1+P2 required')
    if len({p['object'] for p in predicates})!=1 or any(p.get('weight',1)!=1 for p in predicates):raise ValueError('same target and equal weights required')
    if any(c.get('type')!='C1' or c.get('kind')!='non_target_displacement' for c in task.get('constraints',[])):raise ValueError('final-state-only optional displacement C1 required')
    if len(task.get('constraints',[]))>1:raise ValueError('duplicate C1')

