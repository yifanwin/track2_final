"""Trusted synthetic Piper solvability probe; does not claim original cup success."""
import os,sys,json,time,hashlib
from pathlib import Path
WORK=Path(sys.argv[1]).resolve(strict=True)
ROOT=Path(os.environ['TRACK2_STAGING_ROOT']).resolve(strict=True)
assert WORK.is_relative_to(ROOT) and WORK.stat().st_uid==os.getuid()
sys.path.insert(0,str(WORK/'source/Track2_ManiSkill_Evaluator_Candidate_V2'))
sys.path.insert(0,str(WORK/'private_diagnostics'))
os.umask(0o077)
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
import numpy as np,sapien
from track2_judge.scenario.prototype_env import prepare_model
from track2_judge.scenario.piper_robot import PiperPrototype,REST
from mani_skill.envs.sapien_env import BaseEnv
from track2_judge.toolapi.dispatch import ManiSkillSimulatorAdapter
from track2_judge.util.common import array,plain
from track2_judge.scenario.kinematics import Kinematics
PiperPrototype.urdf_path=str(prepare_model())
SEED=int(sys.argv[2]) if len(sys.argv)>2 else 0
INITIAL=np.array([.32,0,.025]);INITIAL[:2]+=np.zeros(2) if SEED==0 else np.random.default_rng(SEED).uniform(-.003,.003,2)
OUT=WORK/('synthetic_grasp' if SEED==0 else 'synthetic_grasp_seed'+str(SEED));OUT.mkdir(mode=0o700,exist_ok=False)
class CubeEnv(BaseEnv):
    SUPPORTED_ROBOTS=[PiperPrototype.uid]
    @property
    def _default_sensor_configs(self):return []
    @property
    def _default_human_render_camera_configs(self):return []
    def evaluate(self):return {}
    def _get_obs_extra(self,info):return {}
    def _load_scene(self,options):
        material=sapien.physx.PhysxMaterial(static_friction=.5,dynamic_friction=.5,restitution=0)
        b=self.scene.create_actor_builder();b.add_box_collision(half_size=[.025]*3,material=material,density=1000);b.set_initial_pose(sapien.Pose(INITIAL));self.cube=b.build(name='synthetic_cube')
        b=self.scene.create_actor_builder();b.add_box_collision(half_size=[.15,.2,.012],material=material);b.set_initial_pose(sapien.Pose([.35,0,-.012]));self.table=b.build_static(name='synthetic_table')
    def _initialize_episode(self,env_idx,options):
        self.agent.reset(REST);self.cube.set_pose(sapien.Pose(INITIAL));self.cube.set_linear_velocity([0,0,0]);self.cube.set_angular_velocity([0,0,0])
def factory():return CubeEnv(num_envs=1,robot_uids=PiperPrototype.uid,control_mode='pd_joint_pos',sim_backend='cpu',render_backend='none',obs_mode='none',reward_mode='none',sim_config={'sim_freq':240,'control_freq':20},enhanced_determinism=True)
def snapshot(e):
    forces=[array(e.scene.get_pairwise_contact_forces(f,e.cube)) for f in e.agent.fingers]
    return {'qpos':array(e.agent.robot.get_qpos()),'qvel':array(e.agent.robot.get_qvel()),'position':array(e.cube.pose.p),'linear_velocity':array(e.cube.linear_velocity),'angular_velocity':array(e.cube.angular_velocity),'finger_force_norms':[np.linalg.norm(f) for f in forces],'support_force':np.linalg.norm(array(e.scene.get_pairwise_contact_forces(e.table,e.cube)))}
a=None;trace=[];diags=[];start=time.monotonic()
result={'status':'FAIL','synthetic_seed':SEED,'original_CoACD32_cup_proven':False,'scope':'synthetic 50mm cube; actual original Piper parameters','physical_robot_parameters_changed':False,'object_setters_during_actions':False}
try:
    a=ManiSkillSimulatorAdapter(factory,snapshot=snapshot,max_steps=1800,max_sim_time_s=90);a.reset(seed=0)
    kin=Kinematics(prepare_model());previous=np.r_[np.asarray(a.get_final_state()['qpos'])[:6],.08].astype(float)
    rotation=np.diag([-1,1,-1]);grip=INITIAL.copy();last=None;phase_end={}
    phases=[('pregrasp',grip+[0,0,.10],.1,160),('approach',grip,.1,120),('close',grip,0,100),('lift',grip+[0,0,.1],0,120),('transfer',grip+[.035,0,.1],0,100),('lower',grip+[.035,0,0],0,120),('release',grip+[.035,0,0],.1,80),('retreat',grip+[.035,0,.12],.1,120),('stable',grip+[.035,0,.12],.1,60)]
    for phase,target,opening,steps in phases:
        points=[target] if last is None or phase in ('close','release','stable') else [last+(target-last)*f for f in np.linspace(1/6,1,6)]
        for point in points:
            q,d=kin.ik(point,rotation,previous[:6],seed_count=8 if len(points)==1 else 4);diags.append(d)
            wanted=np.r_[q,opening]
            for _ in range(max(18,steps//len(points))):
                previous+=np.clip(wanted-previous,-np.r_[[.03]*6,.004],np.r_[[.03]*6,.004]);previous[:6]=np.clip(previous[:6],kin.limits[:,0]+1e-6,kin.limits[:,1]-1e-6);previous[6]=np.clip(previous[6],0,.1)
                a.step({'joint_position':previous[:6].tolist(),'gripper_opening_m':float(previous[6])});s=a.get_final_state();s['phase']=phase;trace.append(s)
        phase_end[phase]=s;last=target.copy()
        print(json.dumps({'phase':phase,'completed':True}),flush=True)
    transfer=[s for s in trace if s['phase']=='transfer'][-20:];stable=[s for s in trace if s['phase']=='stable'];final=trace[-1]
    carried=all(s['position'][2]>.065 and min(s['finger_force_norms'])>.1 and s['support_force']<.1 for s in transfer)
    released=final['support_force']>.1 and max(final['finger_force_norms'])<.1
    speed=max(float(np.linalg.norm(s['linear_velocity'])) for s in trace)
    lift=float(phase_end['lift']['position'][2]-.025);distance=float(np.linalg.norm(np.array(phase_end['transfer']['position'])[:2]-grip[:2]));stable_ok=np.linalg.norm(final['linear_velocity'])<.01 and np.linalg.norm(final['angular_velocity'])<.1
    success=carried and released and stable_ok and lift>.04 and distance>.02 and speed<10
    raw=json.dumps(trace,allow_nan=False);(OUT/'internal_trace.json').write_text(raw)
    result.update(status='PASS' if success else 'FAIL',synthetic_cube_pick_place_proven=bool(success),continuous_airborne_bilateral=carried,released_with_table_support=released,final_stable=bool(stable_ok),lift_m=lift,horizontal_transport_m=distance,max_linear_speed_m_s=speed,max_IK_position_error_m=max(d['position_error_m'] for d in diags),max_IK_rotation_error_deg=float(np.degrees(max(d['rotation_error_rad'] for d in diags))),control_steps=len(trace),trace_sha256=hashlib.sha256(raw.encode()).hexdigest(),mass_kg=.125)
except Exception as e:
    import traceback
    (OUT/'internal_error.log').write_text(traceback.format_exc());result.update(error=type(e).__name__)
finally:
    if a is not None:a.close()
result['wall_s']=time.monotonic()-start
(WORK/'evidence'/('synthetic_grasp_sanitized.json' if SEED==0 else 'synthetic_grasp_seed'+str(SEED)+'_sanitized.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result))
