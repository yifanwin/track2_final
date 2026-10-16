"""Trusted CPU backend. No backend handle belongs in a Skill process."""
import json
from pathlib import Path
import numpy as np
import sapien
from mani_skill.envs.sapien_env import BaseEnv
from track2_judge.util.common import ROOT, array
from track2_judge.scenario.piper_robot import PiperPrototype, REST

def create_env(staging,ttg=False,base_distance=.42,initial_delta=None,initial_yaw_deg=0,collision_root=None):
    staging=Path(staging); previous=Path(collision_root) if collision_root else staging/'experiments/coacd_server_cpu_validation_20261011'
    original=json.loads((previous/'original_manifest.json').read_text()) if ttg else None
    mug=next(o for o in original['objects'] if o['id']=='blue_mug_on_table_000') if ttg else None
    base=np.array(mug['actor_pose_m']['p'])+[-base_distance,0,0] if ttg else np.array([0.,0.,0.])
    if ttg:base[2]=1.5285
    class PrototypeEnv(BaseEnv):
        SUPPORTED_ROBOTS=[PiperPrototype.uid]
        @property
        def _default_sensor_configs(self):return []
        @property
        def _default_human_render_camera_configs(self):return []
        def evaluate(self):return {}
        def _get_obs_extra(self,info):return {}
        def _load_agent(self,options):super()._load_agent(options,sapien.Pose(base))
        def _load_scene(self,options):
            self.objects={};self.metadata={};self.load_evidence={}
            if not ttg:return
            material=sapien.physx.PhysxMaterial(static_friction=.5,dynamic_friction=.5,restitution=0)
            for o in original['objects']:
                name=o['id']; dynamic=o['dynamic']; builder=self.scene.create_actor_builder()
                folder=previous/'assets'/name
                paths=sorted((folder/'coacd32').glob('*.obj')) if dynamic else sorted(folder.glob('*.obj'))
                if not paths:raise FileNotFoundError(folder)
                for path in paths:builder.add_convex_collision_from_file(str(path),material=material,density=1000)
                pose=o['actor_pose_m']; builder.set_initial_pose(sapien.Pose(pose['p'],pose['q_wxyz']))
                actor=builder.build(name=name) if dynamic else builder.build_static(name=name)
                if dynamic:
                    params=json.loads((previous/'physics'/name/'matched_parameters.json').read_text())
                    body=actor._bodies[0];body.mass=params['mass'];body.inertia=params['inertia'];body.cmass_local_pose=sapien.Pose(params['cmass_p'],params['cmass_q'])
                count=len(actor._bodies[0].collision_shapes)
                if count!=len(paths):raise RuntimeError('PhysX dropped collision shapes')
                self.objects[name]=actor;self.metadata[name]={'pivot_m':o['canonical_pivot_zup_m'],'pivot_scaled':True}
                self.load_evidence[name]={'dynamic':dynamic,'hull_count':count,'collision_paths':[str(p) for p in paths], 'mass_kg':float(actor._bodies[0].mass) if dynamic else None,'initial_pose':pose}
        def _initialize_episode(self,env_idx,options):
            # Judge initialization only. No setters are called during actions.
            self.agent.reset(REST);self.agent.robot.set_pose(sapien.Pose(base))
            if ttg:
                for o in original['objects']:
                    a=self.objects[o['id']];p=o['actor_pose_m'];position=np.array(p['p']);quaternion=p['q_wxyz']
                    if o['id']=='blue_mug_on_table_000':
                        from scipy.spatial.transform import Rotation
                        position+=np.asarray(initial_delta or [0,0,0]);quaternion=(Rotation.from_euler('z',initial_yaw_deg,degrees=True)*Rotation.from_quat(np.asarray(quaternion)[[1,2,3,0]])).as_quat()[[3,0,1,2]]
                    a.set_pose(sapien.Pose(position,quaternion))
                    if o['dynamic']:a.set_linear_velocity([0,0,0]);a.set_angular_velocity([0,0,0])
    env=PrototypeEnv(num_envs=1,robot_uids=PiperPrototype.uid,control_mode='pd_joint_pos',sim_backend='cpu',render_backend='none',obs_mode='none',reward_mode='none',sim_config={'sim_freq':240,'control_freq':20},enhanced_determinism=True)
    return env,base,original

def physical_snapshot(env):
    objects={}
    for name,a in env.objects.items():
        dynamic=env.load_evidence[name]['dynamic']
        objects[name]={'actor_p_m':array(a.pose.p),'q_wxyz':array(a.pose.q),'linear_velocity_m_s':array(a.linear_velocity) if dynamic else np.zeros(3),'angular_velocity_rad_s':array(a.angular_velocity) if dynamic else np.zeros(3)}
    contacts=[]
    for c in env.scene.get_contacts():
        contacts.append({'bodies':[b.entity.name.removeprefix('scene-0_') for b in c.bodies],'raw_body_names':[b.entity.name for b in c.bodies],'points':[{'separation_m':float(p.separation),'position_m':np.asarray(p.position),'normal':np.asarray(p.normal),'impulse_Ns':np.asarray(p.impulse)} for p in c.points]})
    finger_forces={}
    if 'blue_mug_on_table_000' in env.objects:
        for f in env.agent.fingers:finger_forces[f.name]=array(env.scene.get_pairwise_contact_forces(f,env.objects['blue_mug_on_table_000']))
    return {'qpos':array(env.agent.robot.get_qpos()),'qvel':array(env.agent.robot.get_qvel()),'tcp_p_m':array(env.agent.tcp.pose.p),'tcp_q_wxyz':array(env.agent.tcp.pose.q),'objects':objects,'contacts':contacts,'finger_forces_N':finger_forces}
