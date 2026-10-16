"""Judge-only selected-case backend. No casebank/actor enters Skill RPC.
Source physical properties are unchanged.
Confirmed scoring scope is distinct from production-switch authorization.

D2：不再逐 hull 校验 sha256（发布期哈希门禁已删）。
保留的路径包含 + 符号链接检查仍防路径穿越。
"""
import os,json
from pathlib import Path
import numpy as np,sapien,torch
from mani_skill.envs.sapien_env import BaseEnv
from track2_judge.scenario.piper_robot import PiperPrototype,REST
from track2_judge.scenario.prototype_env import prepare_model
PiperPrototype.urdf_path=str(prepare_model())
def create_case_env(bundle,casebank,seed=0,solver=(30,4),*,allow_unapproved_diagnostics=False,diagnostic_camera=None):
    from track2_judge.contract.profile import PROFILE,validate
    if bundle['contract'].get('scoring_profile')==PROFILE:validate(bundle['contract'])
    elif not allow_unapproved_diagnostics:raise RuntimeError("diagnostic profile explicit opt-in required")
    DATA=Path(casebank).resolve(strict=True)
    if DATA.stat().st_uid!=os.getuid() or DATA.stat().st_mode & 0o077:raise PermissionError("private casebank must be judge-owned 0700")
    class CaseEnv(BaseEnv):
        SUPPORTED_ROBOTS=[PiperPrototype.uid]
        @property
        def _default_sensor_configs(self):return []
        @property
        def _default_human_render_camera_configs(self):
            if diagnostic_camera is None:return []
            from mani_skill.sensors.camera import CameraConfig
            from mani_skill.utils import sapien_utils
            center=np.asarray(diagnostic_camera['center_m'])
            return CameraConfig('diagnostic_public_rgb',sapien_utils.look_at(center+[0,0,1.2],center,up=[0,1,0]),256,256,.7,.01,10)
        def evaluate(self):return {}
        def _get_obs_extra(self,info):return {}
        def _load_agent(self,options):super()._load_agent(options,sapien.Pose(bundle['robot_base_candidate_m'],bundle.get('robot_base_candidate_q_wxyz',[1,0,0,0])))
        def _load_scene(self,options):
            self.objects={};self.metadata={};self.load_evidence={}
            initial={r['instance']:r for r in bundle['initial_layouts'][seed]['objects']}
            material=sapien.physx.PhysxMaterial(static_friction=.5,dynamic_friction=.5,restitution=0)
            for name,item in bundle['objects'].items():
                b=self.scene.create_actor_builder()
                for ref in item['hulls']:
                    p=(DATA/ref['file']).resolve(strict=True)
                    if DATA not in p.parents or p.is_symlink():raise ValueError('private asset containment')
                    b.add_convex_collision_from_file(str(p),material=material,density=1000)
                    if diagnostic_camera is not None:
                        # Actor-local collision geometry at CURRENT physics pose,
                        # never original goal-baked GLB or segmentation IDs.
                        color=[.5,.5,.5,1]
                        for word,rgb in [('blue',[.05,.15,.85,1]),('green',[.05,.8,.1,1]),('red',[.8,.05,.05,1]),('white',[.8,.8,.8,1])]:
                            if word in name.lower():color=rgb;break
                        b.add_visual_from_file(str(p),material=sapien.render.RenderMaterial(base_color=color))
                pose=initial[name];b.set_initial_pose(sapien.Pose(np.array(pose['position'])/100,np.array(pose['rotation'])[[3,0,1,2]]))
                a=b.build_static(name=name) if item['static'] else b.build(name=name)
                body=a._bodies[0]
                if len(body.collision_shapes)!=len(item['hulls']):raise RuntimeError('engine collision shape count mismatch')
                if not item['static']:body.mass=item['mass_kg'];body.inertia=item['inertia_kg_m2'];body.set_solver_position_iterations(solver[0]);body.set_solver_velocity_iterations(solver[1])
                self.objects[name]=a;self.metadata[name]={'pivot_m':[0,0,0],'pivot_scaled':True}
                # Source canonical pivot is retained in bundle for private inverse
                # provenance; scoring uses this explicitly rebased asset root.
                self.load_evidence[name]={'dynamic':not item['static']}
            # Visible, static scene support fixtures from initial task configuration.
            # Built only during normal scene loading; never exposed as a Skill setter.
            self.spatial_actors={}
            for fixture in bundle.get('initial_static_supports',{}).get(str(seed),[]):
                name=fixture['name']
                if not name.startswith('world_support_') or name in self.objects:raise ValueError('support identity')
                half=np.asarray(fixture['half_size_m'],dtype=float);pos=np.asarray(fixture['position_m'],dtype=float)
                if half.shape!=(3,) or pos.shape!=(3,) or not np.isfinite(np.r_[half,pos]).all() or np.any(half<=0) or np.any(half>2):raise ValueError('support geometry')
                builder=self.scene.create_actor_builder();builder.add_box_collision(half_size=half,material=material)
                if diagnostic_camera is not None:builder.add_box_visual(half_size=half,material=sapien.render.RenderMaterial(base_color=[.55,.55,.55,1]))
                builder.set_initial_pose(sapien.Pose(pos));support=builder.build_static(name=name)
                self.spatial_actors[name]=support;self.metadata[name]={'pivot_m':[0,0,0]}
        def _initialize_episode(self,env_idx,options):self.agent.reset(np.array(bundle.get('initial_robot_qpos',REST),dtype=np.float32))
    return CaseEnv(num_envs=1,robot_uids=PiperPrototype.uid,control_mode='pd_joint_pos',sim_backend='cpu',render_backend='none' if diagnostic_camera is None else 'sapien_cuda:0',obs_mode='none',reward_mode='none',sim_config={'sim_freq':240,'control_freq':20},enhanced_determinism=True)
