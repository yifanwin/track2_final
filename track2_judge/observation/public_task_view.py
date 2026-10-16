"""Candidate public goal projection from one judge-selected task.

The PDF explicitly makes task target fields public and initial states private.
The shape of region/pose values still needs the official interface snapshot.
"""
import json
import math
from track2_judge.scoring.scorer_bridge import v3
from track2_judge.observation.provider import ObservationProvider

PROFILE='track2-public-goals-candidate-1'

def project(task):
    if task.get('task_type')!='pick_place':
        raise ValueError('public profile supports only the confirmed 73 pick/place tasks')
    tier=v3.default_config('sim',task['tier'])
    goal={row['instance']:row for row in task['target_layout']['objects']}
    public=[]
    for pred in task['predicates']:
        kind=pred['type'];entry={'type':kind,'object':pred['object']}
        if kind=='P1':
            entry['region']=pred['region_aabb_m']
        elif kind=='P2':
            rec=goal[pred['object']]
            entry['target_pose']=[*[float(v)/100 for v in rec['position']],*[float(v) for v in rec['rotation']]]
            entry['eps_pos']=tier['eps_pos_cm']/100
            entry['eps_rot_deg']=tier['eps_rot_deg']
        elif kind=='P3':
            entry['reference']=pred['reference'];entry['relation']=pred['relation']
        else:raise ValueError('unapproved predicate projection')
        public.append(entry)
    constraints=[]
    for c in task.get('constraints',[]):
        if c['kind']!='non_target_displacement':raise ValueError('unapproved constraint projection')
        constraints.append({'type':'C1','delta_max':tier['delta_max_cm']/100})
    obj=next(p['object'] for p in task['predicates'] if p['type']=='P2')
    raw={'text_instruction':f'Move {obj} into the declared target region and pose; preserve other objects.',
         'predicates':public,'constraints':constraints}
    encoded=json.dumps(raw,allow_nan=False)
    if len(encoded)>16384 or any(not math.isfinite(v) for v in [tier['eps_pos_cm'],tier['eps_rot_deg'],tier['delta_max_cm']]):raise ValueError('public task budget')
    return ObservationProvider(approved_fields=('text_instruction','predicates','constraints')).prepare(raw)
