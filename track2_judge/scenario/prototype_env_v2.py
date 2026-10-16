"""Actual CPU PhysX package lifecycle + substep evidence + supplemental predicates."""
import os
for k,v in {'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'}.items():os.environ[k]=v
import argparse,copy,hashlib,json,sys
from pathlib import Path
import numpy as np
from track2_judge.util.common import ROOT,plain
from track2_judge.scenario.prototype_env import prepare_model,SETTINGS
from track2_judge.scoring.scorer_bridge import v3
from track2_judge.runner.candidate_runner import CandidateRunner
from track2_judge.toolapi.dispatch import ManiSkillSimulatorAdapter
from track2_judge.scenario.prototype_backend import physical_snapshot
from track2_judge.contract.trusted_state import spatial_snapshot
from track2_judge.observation.provider import ObservationProvider
from track2_tests.fixtures.make_internal_skills import create
from track2_judge.sandbox.rpc import encode

POLICY={'schema':'candidate-event-policy-v2',
        'collision':{'count':'contact_onset','forbidden_pairs':[['box0','table']],'min_impulse_Ns':1e-7},
        'forbidden_zone':{'count':'entry','zones':[{'id':'synthetic_forbidden','subject':'box0','membership':'point','aabb_m':[[-2,-2,-2],[2,2,2]]}]}}
REGION={'type':'P1','object':'box0','region_aabb_m':[[-2,-2,-2],[2,2,2]],'membership':'whole_aabb'}
ON={'type':'P3','object':'box0','reference':'table','relation':'on','tolerance_m':.01}
ATTRIBUTE={'type':'P4','object':'box0','attribute':'class','value':'synthetic_block'}
CHECKPOINTS=[{'name':'region','predicates':[REGION]},{'name':'supported','predicates':[ON]}]

def make_env(render=False,non_target_drop=False):
    import torch,sapien
    torch.set_num_threads(2)
    from track2_judge.scenario.piper_robot import PiperPrototype,REST
    PiperPrototype.urdf_path=str(prepare_model())
    from mani_skill.envs.sapien_env import BaseEnv
    class V2Env(BaseEnv):
        SUPPORTED_ROBOTS=[PiperPrototype.uid]
        @property
        def _default_sensor_configs(self):return []
        @property
        def _default_human_render_camera_configs(self):
            if not render:return []
            from mani_skill.sensors.camera import CameraConfig
            from mani_skill.utils import sapien_utils
            return CameraConfig('candidate_rgb',sapien_utils.look_at([1.5,-1.4,1.2],[.5,0,.15]),96,72,1,.01,10)
        def _load_agent(self,options):super()._load_agent(options,sapien.Pose())
        def evaluate(self):return {}
        def _get_obs_extra(self,info):return {}
        def _load_scene(self,options):
            self.objects={};self.metadata={};self.load_evidence={};self.spatial_actors={}
            labels={}
            for i in range(3):
                name='box'+str(i);b=self.scene.create_actor_builder();b.add_box_collision(half_size=[.025]*3);b.set_initial_pose(sapien.Pose([.8+i*.1,0,.025]))
                if render:b.add_box_visual(half_size=[.025]*3,material=sapien.render.RenderMaterial(base_color=[[1,0,0,1],[0,1,0,1],[0,0,1,1]][i]))
                a=b.build(name=name);self.objects[name]=a;self.metadata[name]={'pivot_m':[0,0,0]};self.load_evidence[name]={'dynamic':True}
                labels[name]={'class':'synthetic_block','color':['red','green','blue'][i]}
            b=self.scene.create_actor_builder();b.add_box_collision(half_size=[1,1,.025]);b.set_initial_pose(sapien.Pose([.5,0,-.025]))
            if render:b.add_box_visual(half_size=[1,1,.025],material=sapien.render.RenderMaterial(base_color=[.5,.5,.5,1]))
            self.spatial_actors['table']=b.build_static(name='table')
            digest=hashlib.sha256(json.dumps(labels,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            self.trusted_asset_registry={'objects':labels,'sha256':digest,'evidence_source':'judge-created primitive catalog; not visual classifier'}
        def _initialize_episode(self,env_idx,options):
            self.agent.reset(REST)
            for i,a in enumerate(self.objects.values()):
                d=self._episode_rng.uniform(-.001,.001,2);a.set_pose(sapien.Pose([.8+.1*i+d[0],d[1],.2 if non_target_drop and i==2 else .025]));a.set_linear_velocity([0,0,0]);a.set_angular_velocity([0,0,0])
    return V2Env(num_envs=1,robot_uids=PiperPrototype.uid,control_mode='pd_joint_pos',sim_backend='cpu',render_backend='cpu' if render else 'none',obs_mode='none',reward_mode='none',sim_config={'sim_freq':240,'control_freq':20},enhanced_determinism=True,render_mode='rgb_array' if render else None)

def main(output,events=True,instruction='move',non_target_drop=False):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    try:
        from track2_judge.diagnostics.state_export import export_layout
        probe=make_env(non_target_drop=non_target_drop);probe.reset(seed=0);initial=export_layout(physical_snapshot(probe)['objects'],probe.metadata,SETTINGS);meta=copy.deepcopy(probe.metadata);probe.close()
        target=copy.deepcopy(initial);target['objects'][0 if non_target_drop else 2]['position'][1]+=.2*100
        scope='moved' if non_target_drop else 'all'
        task=v3.make_task('synthetic/v2_physical_package',target,initial,scope=scope);config=v3.default_config();config['event_constraints']=['collision','forbidden_zone'] if events else []
        suite={**v3.report_meta(),'kind':'mesa_ctrc_suite','scope':scope,'moved_policy':v3.moved_policy(),'synthetic_demo_only':True,'config':config,'tasks':[task]}
        package=create(output/'internal_skill')
        def factory(task,seed,mode):
            provider=ObservationProvider(approved_fields=('text_instruction',))
            return ManiSkillSimulatorAdapter(lambda:make_env(non_target_drop=non_target_drop),snapshot=physical_snapshot,metadata=meta,scene_settings=SETTINGS,public_fields=('text_instruction',),
                observation_provider=provider,spatial_provider=spatial_snapshot,event_policy=POLICY if events else None,
                checkpoints=CHECKPOINTS,recovery=(mode=='final'))
        factory.source='ACTUAL LOCAL CPU PhysX/Piper + original manifest lifecycle'
        result=CandidateRunner(factory,isolated=True,timeout_s=30,max_steps=80,package_zip=package,
            supplemental_contracts={task['task_id']:{'predicates':[REGION,ON,ATTRIBUTE]}}).run(suite,output/'closure',{'text_instruction':instruction})
        if instruction=='move':
            assert len(result['episodes'])==10 and all(e['status']=='failure' for e in result['episodes']),result
            evidence=json.loads((output/'closure/trusted_event_evidence.json').read_text());supp=json.loads((output/'closure/supplemental_verification.json').read_text())
            life=json.loads((output/'closure/skill_lifecycle.json').read_text());trace=json.loads((output/'closure/physical_trace.json').read_text())
            assert all(e['events'][0:2]==['ready','initialize'] and e['events'][-1]=='shutdown' for e in life)
            assert all(e['coverage']=='physics_substep' and e['frames']==e['expected_frames'] and e['frames']>0 for e in evidence)
            if events:assert all(e['event_counts']['collision']>=1 and e['event_counts']['forbidden_zone']>=1 for e in evidence)
            assert all(all(p['passed'] for p in e['predicates']) for e in supp),supp
            assert all(e['P5']['passed'] for e in supp if e['mode']=='final')
            score=json.loads((output/'closure/frozen_score_diagnostic.json').read_text());assert abs(score['tasks'][0]['SR_final']-(0 if non_target_drop else 2/3))<1e-12
            assert v3.evaluate(suite,json.loads((output/'closure/runs.json').read_text()),output/'closure')==score
            paired_runs=json.loads((output/'closure/runs.json').read_text())['runs'];initial_hashes=[]
            for j in range(0,10,2):
                a=(output/'closure'/paired_runs[j]['initial_layout']).read_bytes();b=(output/'closure'/paired_runs[j+1]['initial_layout']).read_bytes()
                assert a==b;initial_hashes.append(hashlib.sha256(a).hexdigest())
            assert len(set(initial_hashes))==5
            q0=trace[0]['physical']['qpos'];motion=max(abs(t['physical']['qpos'][0]-q0[0]) for t in trace);assert motion>.005
            if non_target_drop:
                runs=json.loads((output/'closure/runs.json').read_text())['runs'];displacements=[]
                for run in runs:
                    a=json.loads((output/'closure'/run['initial_layout']).read_text());b=json.loads((output/'closure'/run['final_layout']).read_text())
                    start_box=next(x for x in a['objects'] if x['instance']=='box2');end_box=next(x for x in b['objects'] if x['instance']=='box2')
                    d=abs(start_box['position'][2]-end_box['position'][2]);assert d>2;displacements.append(d)
            final={'status':'PASS','scope':'actual package -> Piper -> trusted substeps -> original Recorder/V3, synthetic goals',
                'episodes':10,'distinct_paired_initial_states':5,'physics_frames':sum(e['frames'] for e in evidence),'joint_motion_rad':motion,'P1_P3_P4_P5':'supplemental PASS; NOT frozen scoring compatibility',
                'events':[{'mode':e['mode'],'counts':e['event_counts'],'frames':e['frames']} for e in evidence],
                'SR_final':score['tasks'][0]['SR_final'],'raw_diagnostic_score':score['score_100'],'official_score':None,'non_target_displacement_cm':displacements if non_target_drop else None,'policy':'candidate-event-policy-v2, intentionally forbidden support contact; official semantics pending','production_ready':False}
        else:
            assert result.get('scoring_status')=='REFUSED: missing trustworthy state'
            final={'status':'PASS','scope':'enabled constraints with no physics samples','scoring':'REFUSED; no invented zero events','production_ready':False}
    except Exception as exc:
        import traceback
        final={'status':'FAIL','error':type(exc).__name__,'details':str(exc),'trace':traceback.format_exc(),'production_ready':False}
    (output/'v2_physics_result.json').write_text(encode(final),encoding='utf8');print(json.dumps(final));return int(final['status']!='PASS')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--no-events',action='store_true');p.add_argument('--instruction',default='move');p.add_argument('--non-target-drop',action='store_true');a=p.parse_args();sys.exit(main(a.output,not a.no_events,a.instruction,a.non_target_drop))
