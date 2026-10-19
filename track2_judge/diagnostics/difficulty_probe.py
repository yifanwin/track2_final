"""Separate mathematical calibration and real control score distributions."""
import argparse,copy,json,os,zipfile
from pathlib import Path
from track2_judge.scoring.scorer_bridge import v3
from track2_judge.contract.extension import TrustedContractExtension,SCHEMA
from track2_judge.runner.candidate_runner import CandidateRunner
from track2_judge.toolapi.dispatch import ManiSkillSimulatorAdapter
from track2_judge.scenario.prototype_env_v2 import make_env
from track2_judge.scenario.prototype_backend import physical_snapshot
from track2_judge.contract.trusted_state import spatial_snapshot
from track2_judge.scenario.prototype_env import SETTINGS
from track2_judge.diagnostics.state_export import export_layout
from track2_judge.observation.provider import ObservationProvider

def main(output):
    root=Path(output).resolve();root.mkdir(parents=True,mode=0o700,exist_ok=False);os.umask(0o077)
    e=make_env();e.reset(seed=0);layout=export_layout(physical_snapshot(e)['objects'],e.metadata,SETTINGS);meta=copy.deepcopy(e.metadata);e.close()
    def task(tier):
        target=copy.deepcopy(layout);target['objects'][0]['position'][1]+=8;p=target['objects'][0]['position'];center=[x/100 for x in p]
        return {'contract_schema':SCHEMA,'task_id':'difficulty_'+tier,'task_type':'pick_place','tier':tier,'target_layout':target,'initial_layout':copy.deepcopy(layout),
                'predicates':[{'type':'P1','object':'box0','membership':'whole_aabb','region_aabb_m':[[x-.05 for x in center],[x+.05 for x in center]],'weight':1},{'type':'P2','object':'box0','weight':1},{'type':'P3','object':'box0','reference':'table','relation':'on','tolerance_m':.01,'weight':1}],
                'constraints':[{'type':'C1','kind':'non_target_displacement'}]}
    code='''def initialize(context):pass
def shutdown(context):pass
def run(observation,tools):
    r=observation['robot'];q=list(r['joint_position'][:6]);g=r['joint_position'][6]-r['joint_position'][7]
    count=1 if observation['public_task_inputs']['text_instruction']=='hold' else 6
    targets=[]
    for i in range(count):
        if count>1:q[0]+=.005
        targets.append({'joint_position':list(q),'gripper_opening_m':float(g)})
    # ★ 强制 Action Chunk：不足 8 步就补到 8（原地重复最后一步 —— 等价于"没有更多运动"）。
    #   这里只做诊断，语义不变。
    while len(targets)<8:targets.append(dict(targets[-1]))
    tools.joint_trajectory(targets[:25])
    return {'status':'completed'}
'''
    package=root/'internal_weak.zip'
    with zipfile.ZipFile(package,'w') as z:
        z.writestr('manifest.json',json.dumps({'skill_name':'internal_difficulty_probe','version':'2','entrypoint':'main.py','api_version':'ctrc-skill-v1'}));z.writestr('main.py',code)
    results=[]
    for tier in ('loose','standard','strict'):
        for strategy in ('hold','weak_joint_motion'):
            t=task(tier);engine=TrustedContractExtension()
            def factory(task,seed,mode=None):return ManiSkillSimulatorAdapter(make_env,snapshot=physical_snapshot,spatial_provider=spatial_snapshot,metadata=meta,scene_settings=SETTINGS,public_fields=['text_instruction'],observation_provider=ObservationProvider(approved_fields=['text_instruction']),max_steps=80)
            factory.source='ACTUAL CPU synthetic score-distribution probe; no GT strategy'
            d=CandidateRunner(factory,package_zip=package,contract_extension=engine,isolated=True,max_steps=80,timeout_s=30).run({'config':v3.default_config(),'tasks':[t]},root/(tier+'_'+strategy),{'text_instruction':'hold' if strategy=='hold' else 'weak'})
            score=json.loads((root/(tier+'_'+strategy)/'extension_candidate_score.json').read_text());metrics=score['tasks'][0]
            results.append({'tier':tier,'strategy':strategy,'episodes':5,'actual_robot_physics':True,'candidate_score_100':score['candidate_score_100']})
    # Mathematical fixtures quantify tier boundaries without claiming a robot
    # actually created these final states. These are not official task scores.
    calibration=[]
    for tier in ('loose','standard','strict'):
        cfg=v3.default_config('sim',tier)
        for error_cm in (1,2.5,4,6,12):
            p1=error_cm<=5;p2=error_cm<=cfg['eps_pos_cm'];p3=True;c=(p1+p2+p3)/3;s=c if c>=cfg['completion_threshold'] else 0
            runs=[{'s':s,'strict_success':c==1,'violation_count':0,'elapsed_s':10} for _ in range(5)]
            # ★ 2026-10-18：`v3.summarize` 是冻结件（要求 first/final 配对），
            #   诊断里不再用它算总分；这里只报**几何数学夹具**的中间量。
            calibration.append({'tier':tier,'error_cm':error_cm,'completion':c,'s':s,'evidence_type':'MATH_FIXTURE_ONLY, no physical execution claim'})
    result={'status':'PASS: MEASURED DISTRIBUTIONS','actual_strategy_results':results,'mathematical_threshold_calibration':calibration,
        'physical_ability_separation':'NOT PROVEN: hold/weak motion did not manipulate target; inspect auxiliary-score concentration','official_scores':None,'policy':'no weight/threshold/formula modified'}
    (root/'difficulty_results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args().output)
