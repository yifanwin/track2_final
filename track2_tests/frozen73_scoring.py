"""Confirmed 73-rule fixtures; synthetic states, never official trial results."""
import argparse,copy,json
from pathlib import Path
from track2_judge.contract.profile import PROFILE,load_config
from track2_judge.contract.extension import TrustedContractExtension,SCHEMA
from track2_tests.fixtures.synthetic import suite
from track2_judge.scoring.scorer_bridge import v3

p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
old=suite()['tasks'][0];name=old['predicates'][0]['object'];e=TrustedContractExtension();cfg=load_config()
t={**copy.deepcopy(old),'contract_schema':SCHEMA,'scoring_profile':PROFILE,'task_type':'pick_place','tier':'standard','constraints':[],
   'predicates':[{'type':'P1','object':name,'membership':'point','region_aabb_m':[[-1,-1,-1],[1,1,1]],'weight':1},{'type':'P2','object':name,'weight':1}]}
checks={}
for region_pass,pose_pass,completion,s in [(False,False,0,0),(True,False,.5,0),(False,True,.5,0),(True,True,1,1)]:
    state={'spatial_objects':{name:{'canonical_root_m':[0,0,0] if region_pass else [2,2,2]}}}
    result=e.verify(t,old['initial_layout'],old['target_layout'] if pose_pass else old['initial_layout'],cfg,initial_state=state,final_state=state,event_evidence={'event_counts':{'collision':99999}})
    assert result['completion']==completion and result['s']==s and result['violation_count']==0
checks['P1_P2_equal_weight_threshold']='PASS';checks['trajectory_events_not_scored']='PASS'
for change in ('P3','weight','trajectory_C1','threshold','time_ref'):
    bad=copy.deepcopy(t);c=copy.deepcopy(cfg)
    if change=='P3':bad['predicates'].append({'type':'P3','object':name})
    elif change=='weight':bad['predicates'][0]['weight']=2
    elif change=='trajectory_C1':bad['constraints']=[{'type':'C1','kind':'collision'}]
    elif change=='threshold':c['completion_threshold']=.5
    else:c['time_ref_s']=45
    try:e.validate_task(bad);e.config_for_task(bad,c)
    except ValueError:checks['reject_'+change]='PASS'
    else:raise AssertionError('unfrozen semantics accepted')
for tier in v3.TIERS:assert e.config_for_task({**t,'tier':tier},cfg)==v3.default_config('sim',tier)
checks['three_frozen_tiers']='PASS'
report={'status':'PASS','checks':checks,'source':'synthetic frozen scoring fixtures','WAIT_CONFIRM':['input_interface','production_switch_permission']}
(root/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
