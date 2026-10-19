"""Local CPU PhysX closure. TTG reference uses only authorized external colliders.
TTG goals are synthetic offsets, NOT official hidden targets or official scores.
"""
import os
for key,value in {'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'}.items():os.environ[key]=value
import argparse, copy, hashlib, json, sys, time, xml.etree.ElementTree as ET
from pathlib import Path
from track2_judge.util.common import ROOT, plain
for key,name in {'XDG_CACHE_HOME':'cache/xdg','TMPDIR':'tmp','MS_ASSET_DIR':'cache/assets','TORCH_HOME':'cache/torch'}.items():
    path=ROOT/name;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
from track2_judge.sandbox.rpc import encode
from track2_judge.scoring.scorer_bridge import v3
from track2_judge.diagnostics.state_export import export_layout
from track2_judge.toolapi.dispatch import ManiSkillSimulatorAdapter
from track2_judge.runner.candidate_runner import CandidateRunner

SETTINGS={'units':'centimeters','up_axis':'Z','rotation_convention':'absolute_quaternion_xyzw','scene_offset':[0,0,0]}

def prepare_model():
    path=ROOT/'model/piper_sim.urdf';tree=ET.parse(path)
    for mesh in tree.findall('.//mesh'):
        original=mesh.get('filename');rel=original.split('/upstream/',1)[1]
        target=ROOT/'upstream'/rel
        if not target.is_file(): raise FileNotFoundError('public Piper mesh missing')
        mesh.set('filename',str(target))
    # Runtime generated copy only, immutable source stays intact.
    dest=ROOT/'tmp'/('piper_runtime_'+str(os.getpid())+'.urdf');dest.parent.mkdir(exist_ok=True)
    # Standard XML encoding plus per-process atomic publication prevents parser
    # failures and cross-test races on WSL paths containing Chinese characters.
    temporary=dest.with_suffix('.pending')
    tree.write(temporary,encoding='utf-8',xml_declaration=True);os.replace(temporary,dest)
    return dest

def run(args):
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    try:
        import torch, sapien, importlib.metadata, numpy as np
        torch.set_num_threads(2)
        from track2_judge.scenario.piper_robot import PiperPrototype
        PiperPrototype.urdf_path=str(prepare_model())
        from track2_judge.scenario.prototype_backend import create_env,physical_snapshot
        if args.ttg:
            previous=Path(args.collision_root).resolve()
            if not (previous/'original_manifest.json').is_file():raise RuntimeError('authorized TTG collider source missing')
            def env_factory(seed=0):
                delta=np.random.default_rng(seed).uniform(-.001,.001,3);delta[2]=0
                return create_env('',True,collision_root=previous,initial_delta=delta.tolist())[0]
        else:
            # Actual SAPIEN static table and dynamic boxes; no synthetic state updates.
            from mani_skill.envs.sapien_env import BaseEnv
            from track2_judge.scenario.piper_robot import REST
            class SyntheticPhysicsEnv(BaseEnv):
                SUPPORTED_ROBOTS=[PiperPrototype.uid]
                @property
                def _default_sensor_configs(self):return []
                @property
                def _default_human_render_camera_configs(self):return []
                def evaluate(self):return {}
                def _get_obs_extra(self,info):return {}
                def _load_scene(self,options):
                    self.objects={};self.metadata={};self.load_evidence={}
                    for i in range(3):
                        b=self.scene.create_actor_builder();b.add_box_collision(half_size=[.025,.025,.025]);b.set_initial_pose(sapien.Pose([.8+i*.1,0,.025]))
                        a=b.build(name='box'+str(i));self.objects['box'+str(i)]=a;self.metadata['box'+str(i)]={'pivot_m':[0,0,0]};self.load_evidence['box'+str(i)]={'dynamic':True}
                    b=self.scene.create_actor_builder();b.add_box_collision(half_size=[1,1,.025]);b.set_initial_pose(sapien.Pose([.5,0,-.025]));b.build_static(name='table')
                def _initialize_episode(self,env_idx,options):
                    self.agent.reset(REST)
                    for i,a in enumerate(self.objects.values()):
                        jitter=self._episode_rng.uniform(-.001,.001,2)
                        a.set_pose(sapien.Pose([.8+i*.1+jitter[0],jitter[1],.025]));a.set_linear_velocity([0,0,0]);a.set_angular_velocity([0,0,0])
            def env_factory(seed=0):return SyntheticPhysicsEnv(num_envs=1,robot_uids=PiperPrototype.uid,control_mode='pd_joint_pos',sim_backend='cpu',render_backend='none',obs_mode='none',reward_mode='none',sim_config={'sim_freq':240,'control_freq':20},enhanced_determinism=True)
        probe=env_factory();probe.reset(seed=0)
        initial=export_layout(physical_snapshot(probe)['objects'],probe.metadata,SETTINGS);metadata=copy.deepcopy(probe.metadata)
        load=plain(probe.load_evidence);probe.close()
        target=copy.deepcopy(initial)
        # A deliberately unreachable synthetic offset: genuine physical failure must score as such.
        for i,obj in enumerate(target['objects']):
            if args.goal=='unreachable' or (args.goal=='partial' and i>=2) or (args.goal=='below' and i>=1):obj['position'][1]+=20
        task=v3.make_task('authorized_internal/ttg_diagnostic' if args.ttg else 'synthetic/physical_boxes',target,initial,scope='all')
        suite={**v3.report_meta(),'kind':'mesa_ctrc_suite','scope':'all','moved_policy':v3.moved_policy(),'synthetic_demo_only':True,'config':v3.default_config(),'tasks':[task]}
        def factory(task,seed):return ManiSkillSimulatorAdapter(lambda:env_factory(seed),snapshot=physical_snapshot,metadata=metadata,scene_settings=SETTINGS,public_fields=('text_instruction',))
        factory.source='LOCAL CPU PhysX + ManiSkill + Piper; synthetic scoring target'
        diag=CandidateRunner(factory,isolated=True,timeout_s=.4 if args.instruction=='timeout' else 90,max_steps=1000).run(suite,output/'closure',{'text_instruction':args.instruction})
        traces=json.loads((output/'closure/physical_trace.json').read_text())
        episodes=diag['episodes']; assert len(episodes)==5
        expected='timeout' if args.instruction=='timeout' else 'error' if args.instruction=='illegal' else 'success' if args.goal=='stationary' else 'failure'
        assert all(e['status']==expected for e in episodes),episodes
        if expected=='failure':assert all(e['attempts']==1 for e in episodes)
        motion=0
        if args.instruction=='move':
            assert traces
            q0=traces[0]['physical']['qpos'];motion=max(abs(s['physical']['qpos'][0]-q0[0]) for s in traces)
            assert motion>.02
        # ★ 2026-10-18：冻结 V3 评分器要求 first/final 配对，单模式下**不再产分**。
        #   这条诊断路径明确断言「不产分」，与 `candidate_runner` 的行为一致。
        assert diag.get('scoring_status')=='REFUSED: frozen paired scorer retired (single-mode)',diag.get('scoring_status')
        assert 'raw_frozen_score_100' not in diag
        runs=json.loads((output/'closure/runs.json').read_text())['runs'];hashes=[]
        for run in runs:
            a=(output/'closure'/run['initial_layout']).read_bytes()
            hashes.append(hashlib.sha256(a).hexdigest())
        assert len(set(hashes))==5
        result={'status':'PASS','scope':'actual physics closure with synthetic goals; stationary success is not successful pick/place or official evaluation','goal_policy':args.goal,'instruction':args.instruction,'expected_execution_status':expected,'episodes':5,'recorded_control_steps':len(traces),'joint_motion_rad':motion,
                'all_failure':expected=='failure','distinct_initial_states':5,'official_score':None,'grasp_success':'NOT TESTED: joint/gripper diagnostic does not establish successful grasp',
                'versions':{k:importlib.metadata.version(k) for k in ('mani_skill','sapien','torch','numpy','scipy')},'engine':'cpu PhysX','GPU_used':False,
                'asset_collision_counts':{k:v.get('hull_count') for k,v in load.items()},'public_observation':'proprioception and approved internal instruction; camera NOT IMPLEMENTED'}
    except Exception as exc:
        import traceback
        result={'status':'FAIL','error':type(exc).__name__,'message':str(exc),'trace':traceback.format_exc(),'official_score':None}
    (output/'physical_result.json').write_text(encode(result),encoding='utf8');print(json.dumps(result));return 0 if result['status']=='PASS' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--ttg',action='store_true');p.add_argument('--collision-root');p.add_argument('--goal',choices=['unreachable','stationary','partial','below'],default='unreachable');p.add_argument('--instruction',choices=['move','noop','illegal','timeout'],default='move');args=p.parse_args();sys.exit(run(args))
