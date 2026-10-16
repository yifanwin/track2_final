"""Private representative collision-fixture control, NOT Frozen73 GT or assets.

Positive controls exercise legal isolated Skill -> Piper -> PhysX -> complete
Recorder/Verifier/Scorer repeats. They do not certify the original TTG assets.
"""
import argparse,copy,json,math,os
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
import numpy as np,sapien,torch
from mani_skill.envs.sapien_env import BaseEnv
from track2_judge.scenario.prototype_env import prepare_model,SETTINGS
from track2_judge.scenario.piper_robot import PiperPrototype,REST
from track2_judge.runner.candidate_runner import CandidateRunner
from track2_judge.toolapi.dispatch import ManiSkillSimulatorAdapter
from track2_judge.scenario.prototype_backend import physical_snapshot
from track2_judge.contract.trusted_state import spatial_snapshot
from track2_judge.observation.provider import ObservationProvider
from track2_judge.contract.extension import TrustedContractExtension,SCHEMA
from track2_judge.contract.profile import PROFILE,load_config
from track2_judge.scoring.result_gate import atomic_json

PiperPrototype.urdf_path=str(prepare_model());torch.set_num_threads(2)

def make_env(kind):
    class CalibrationEnv(BaseEnv):
        SUPPORTED_ROBOTS=[PiperPrototype.uid]
        @property
        def _default_sensor_configs(self):return []
        @property
        def _default_human_render_camera_configs(self):return []
        def evaluate(self):return {}
        def _get_obs_extra(self,info):return {}
        def _load_scene(self,options):
            self.objects={};self.metadata={};self.load_evidence={}
            mat=sapien.physx.PhysxMaterial(static_friction=.5,dynamic_friction=.5,restitution=0)
            b=self.scene.create_actor_builder()
            if kind=='box':b.add_box_collision(half_size=[.025]*3,material=mat,density=1000)
            else:
                # Convex segments retain an open bowl/mug; no filled convex hull.
                height=.05 if kind=='bowl' else .065;radius=.036 if kind=='bowl' else .025
                b.add_box_collision(pose=sapien.Pose([0,0,-height/2+.004]),half_size=[radius,radius,.004],material=mat,density=1000)
                for i in range(12):
                    a=i*math.pi/6;p=[radius*math.cos(a),radius*math.sin(a),0]
                    b.add_box_collision(pose=sapien.Pose(p,[math.cos(a/2),0,0,math.sin(a/2)]),half_size=[.004,radius*.28,height/2],material=mat,density=1000)
                if kind=='complex_mug':
                    for z in (-.02,.02):b.add_box_collision(pose=sapien.Pose([.041,0,z]),half_size=[.017,.006,.004],material=mat,density=1000)
                    b.add_box_collision(pose=sapien.Pose([.057,0,0]),half_size=[.004,.006,.024],material=mat,density=1000)
            z=.025 if kind=='box' else .025 if kind=='bowl' else .0325
            b.set_initial_pose(sapien.Pose([.32,0,z]));actor=b.build(name='calibration_'+kind)
            body=actor._bodies[0];body.mass=.125;body.inertia=[.0001]*3;body.set_solver_position_iterations(30);body.set_solver_velocity_iterations(4)
            self.objects['calibration_'+kind]=actor;self.metadata['calibration_'+kind]={'pivot_m':[0,0,0],'pivot_scaled':True};self.load_evidence['calibration_'+kind]={'dynamic':True}
            t=self.scene.create_actor_builder();t.add_box_collision(half_size=[.15,.2,.012],material=mat);t.set_initial_pose(sapien.Pose([.35,0,-.012]));table=t.build_static(name='calibration_table')
            self.objects['calibration_table']=table;self.metadata['calibration_table']={'pivot_m':[0,0,0]};self.load_evidence['calibration_table']={'dynamic':False}
        def _initialize_episode(self,env_idx,options):self.agent.reset(REST)
    return CalibrationEnv(num_envs=1,robot_uids=PiperPrototype.uid,control_mode='pd_joint_pos',sim_backend='cpu',render_backend='none',obs_mode='none',reward_mode='none',sim_config={'sim_freq':240,'control_freq':20},enhanced_determinism=True)

def main(kind,skill,output):
    out=Path(output);out.mkdir(mode=0o700,parents=True,exist_ok=False);name='calibration_'+kind
    z=.025 if kind in ('box','bowl') else .0325
    layout={'coordinate_frame':'scene_local_cm','scene_settings':SETTINGS,'objects':[{'instance':name,'position':[32,0,z*100],'rotation':[0,0,0,1]},{'instance':'calibration_table','position':[35,0,-1.2],'rotation':[0,0,0,1]}]}
    goal=copy.deepcopy(layout);goal['objects'][0]['position'][0]+=3.5
    task={'task_id':'SYNTHETIC_CALIBRATION_'+kind,'task_type':'pick_place','tier':'standard','contract_schema':SCHEMA,'scoring_profile':PROFILE,
          'initial_layout':layout,'target_layout':goal,'moved_objects':[name],'predicates':[{'type':'P1','object':name,'membership':'point','region_aabb_m':[[.30,-.05,-.03],[.40,.05,.09]],'weight':1},{'type':'P2','object':name,'weight':1}],'constraints':[{'type':'C1','kind':'non_target_displacement'}]}
    public={'text_instruction':'PUBLIC_CALIBRATION_35MM: diagnostic fixture, fixed public calibration prior; not a TTG initial observation','predicates':[{'type':'P1','object':name,'region':task['predicates'][0]['region_aabb_m']},{'type':'P2','object':name,'target_pose':[.355,0,z,0,0,0,1],'eps_pos':.03,'eps_rot_deg':10}],'constraints':[{'type':'C1','delta_max':.02}]}
    def factory(task,seed,mode):
        meta={name:{'pivot_m':[0,0,0]},'calibration_table':{'pivot_m':[0,0,0]}}
        return ManiSkillSimulatorAdapter(lambda:make_env(kind),snapshot=physical_snapshot,metadata=meta,scene_settings=SETTINGS,public_fields=tuple(public),observation_provider=ObservationProvider(approved_fields=tuple(public)),spatial_provider=spatial_snapshot)
    factory.source='ACTUAL CPU PhysX legal Skill representative SYNTHETIC calibration; not frozen TTG assets'
    r=CandidateRunner(factory,isolated=True,package_zip=skill,contract_extension=TrustedContractExtension(),timeout_s=90).run({'tasks':[task],'config':load_config()},out/'closure',public)
    trace=json.loads((out/'closure/physical_trace.json').read_text());initial_z=z;lift=max(t['physical']['objects'][name]['actor_p_m'][2]-initial_z for t in trace) if trace else 0
    details={'kind':kind,'scope':'synthetic representative positive control, not Frozen73 Case certificate','status':'PASS' if r['result_gate']['publishable'] else 'FAIL','episodes':len(r['episodes']),'task_successes':sum(e['status']=='success' for e in r['episodes']),'task_failures':sum(e['status']=='failure' for e in r['episodes']),'execution_errors':sum(e['status'] not in ('success','failure') for e in r['episodes']),'max_lift_m':lift,'grasp_motion_observed':lift>.04,'P1_P2_evaluated':True,'health_frames':sum(h['summary'].get('frames',0) for h in r['physics_health']),'health_flags':sorted({k for h in r['physics_health'] for k in h['summary'].get('flag_counts',{})}),'result_gate':r['result_gate']['status'],'frozen_rule_score_100':r.get('frozen_rule_score_100'),'formal_score':None}
    atomic_json(out/'sanitized_result.json',details);print(json.dumps(details));return int(details['status']!='PASS')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['box','bowl','complex_mug'],required=True);p.add_argument('--skill-zip',required=True);p.add_argument('--output',required=True);a=p.parse_args();raise SystemExit(main(a.kind,a.skill_zip,a.output))
