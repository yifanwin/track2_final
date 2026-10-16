"""New real Skill/Piper/PhysX/Recorder/extension closure, synthetic goals only."""
import argparse,copy,hashlib,json,os
from pathlib import Path
from track2_judge.scoring.scorer_bridge import v3
from track2_judge.contract.extension import TrustedContractExtension,SCHEMA,digest
from track2_judge.runner.candidate_runner import CandidateRunner
from track2_judge.toolapi.dispatch import ManiSkillSimulatorAdapter
from track2_judge.scenario.prototype_env_v2 import make_env
from track2_judge.scenario.prototype_env import SETTINGS
from track2_judge.scenario.prototype_backend import physical_snapshot
from track2_judge.diagnostics.state_export import export_layout
from track2_judge.contract.trusted_state import spatial_snapshot
from track2_judge.observation.provider import ObservationProvider
from track2_tests.fixtures.make_internal_skills import create

def main(output):
    root=Path(output).resolve();root.mkdir(parents=True,mode=0o700,exist_ok=False);os.umask(0o077)
    env=make_env();env.reset(seed=0);layout=export_layout(physical_snapshot(env)['objects'],env.metadata,SETTINGS);registry=env.trusted_asset_registry['sha256'];metadata=copy.deepcopy(env.metadata);env.close()
    def region(obj):return {'type':'P1','object':obj,'membership':'whole_aabb','region_aabb_m':[[-2,-2,-2],[2,2,2]],'weight':1}
    p3={'type':'P3','object':'box1','reference':'box2','relation':'left-of','axis':'X','tolerance_m':.01,'weight':1}
    checkpoints=[{'name':'in_work_region','predicates':[region('box0')]},{'name':'relation_checkpoint','predicates':[p3]}]
    def task(name,kind,predicates,constraints=()):return {'contract_schema':SCHEMA,'task_id':'synthetic_'+name,'task_type':kind,'tier':'standard','predicates':predicates,'constraints':list(constraints),'initial_layout':copy.deepcopy(layout),'target_layout':copy.deepcopy(layout)}
    pick=task('pick','pick_place',[region('box0'),{'type':'P2','object':'box0','weight':1},{'type':'P3','object':'box0','reference':'table','relation':'on','tolerance_m':.01,'weight':1}]);pick['target_layout']['objects'][0]['position'][1]+=8
    attr=task('attribute','attribute_sort',[region('box0'),{'type':'P4','object':'box0','attribute':'class','value':'synthetic_block','weight':1}]);attr['asset_registry_sha256']=registry
    spatial=task('spatial','spatial_relation',[region('box1'),p3])
    organize=task('organize','multi_object_organize',[region('box0'),{'type':'P2','object':'box0','weight':1},region('box1'),p3,{'type':'P5','checkpoints':checkpoints}],[{'type':'C1','kind':'non_target_displacement'}])
    tasks=[pick,attr,spatial,organize];engine=TrustedContractExtension();config=v3.default_config();package=create(root/'internal_skill')
    def factory(task,seed,mode):
        cp=next((p['checkpoints'] for p in task['predicates'] if p['type']=='P5'),[])
        return ManiSkillSimulatorAdapter(make_env,snapshot=physical_snapshot,spatial_provider=spatial_snapshot,metadata=metadata,scene_settings=SETTINGS,public_fields=['text_instruction'],observation_provider=ObservationProvider(approved_fields=['text_instruction']),checkpoints=cp,recovery=(mode=='final'),max_steps=80)
    factory.source='ACTUAL CPU Piper, trusted synthetic primitive scene; new extension contract closure'
    diagnostic=CandidateRunner(factory,package_zip=package,isolated=True,contract_extension=engine,timeout_s=30,max_steps=80).run({'config':config,'tasks':tasks},root/'closure',{'text_instruction':'move'})
    assert len(diagnostic['episodes'])==40 and 'extension_candidate_score_100' in diagnostic,diagnostic
    score=json.loads((root/'closure/extension_candidate_score.json').read_text());assert score['official_score'] is None
    assert len(score['tasks'])==4 and all(t['SR_final']==1 for t in score['tasks'][1:]),score
    assert abs(score['tasks'][0]['SR_final']-2/3)<1e-12,score
    evidence=json.loads((root/'closure/trusted_event_evidence.json').read_text());assert all(e['P5']['passed'] for e in evidence if e['task_index']==3 and e['mode']=='final')
    submission=json.loads((root/'closure/runs.json').read_text());mutated=copy.deepcopy(submission);mutated['runs'][0]['elapsed_s']+=1
    try:engine.evaluate({'config':config,'tasks':tasks},mutated,root/'closure')
    except ValueError:pass
    else:raise AssertionError('record tamper accepted')
    # State-file mutation is detected even inside the judge test process.
    path=root/'closure'/submission['runs'][0]['final_layout'];before=path.read_bytes();state=json.loads(before);state['objects'][0]['position'][0]+=1;path.write_text(json.dumps(state))
    try:
        try:engine.evaluate({'config':config,'tasks':tasks},submission,root/'closure')
        except ValueError:pass
        else:raise AssertionError('state tamper accepted')
    finally:path.write_bytes(before)
    result={'status':'PASS','new_function':'four PDF single-arm task types through original lifecycle, real Piper/PhysX, trusted Recorder and candidate predicate scoring','state_timing_tamper':'REJECTED by independent evidence replay','episodes':40,'per_task_SR_final':[t['SR_final'] for t in score['tasks']],
            'candidate_score_100':score['candidate_score_100'],'official_score':None,'P5':'trusted substep sequence PASS; no standalone weight','formula':'actual unmodified V3.summarize','scope':'synthetic functional fixture; not TTG difficulty calibration'}
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args().output)
