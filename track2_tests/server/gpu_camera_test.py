"""Private TTG RGB acceptance: CPU PhysX, authorized GPU0 rasterizer, host readback."""
import os,sys
from pathlib import Path
WORK=Path(sys.argv[1]).resolve(strict=True)
ROOT=Path(os.environ['TRACK2_STAGING_ROOT']).resolve(strict=True)
assert WORK.is_relative_to(ROOT) and WORK.stat().st_uid==os.getuid()
SOURCE=WORK/'source/Track2_ManiSkill_Evaluator_Candidate_V2'
sys.path.insert(0,str(SOURCE))
os.umask(0o077)
AUTHORIZED_GPU_UUID=os.environ['CUDA_VISIBLE_DEVICES']
import hashlib,json,time,zipfile,traceback
import numpy as np
import sapien
from track2_judge.scenario.prototype_env import prepare_model
from track2_judge.scenario.piper_robot import PiperPrototype,REST
from track2_judge.scenario.prototype_backend import physical_snapshot
from track2_judge.toolapi.dispatch import ManiSkillSimulatorAdapter
from track2_judge.observation.provider import ObservationProvider
from track2_judge.sandbox.package import unpack_skill,PackageExecutor
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from PIL import Image
# The CPU-only physical_test helper intentionally clears CUDA visibility on
# import. Restore the caller's explicitly authorized device for rendering.
os.environ['CUDA_VISIBLE_DEVICES']=AUTHORIZED_GPU_UUID
ASSETS=ROOT/'experiments/coacd_server_cpu_validation_20261011'
manifest=json.loads((ASSETS/'original_manifest.json').read_text())
cup=next(o for o in manifest['objects'] if o['id']=='blue_mug_on_table_000')
center=np.asarray(cup['actor_pose_m']['p']);base=center+[-.42,0,0];base[2]=1.5285
PiperPrototype.urdf_path=str(prepare_model())
OUT=WORK/os.environ.get('TRACK2_CAMERA_OUTPUT','gpu_camera_retry');OUT.mkdir(mode=0o700,exist_ok=False)
class CameraEnv(BaseEnv):
    SUPPORTED_ROBOTS=[PiperPrototype.uid]
    @property
    def _default_sensor_configs(self):return []
    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig('ttg_rgb',sapien_utils.look_at(center+[-.55,-.65,.50],center+[0,0,.03]),160,120,1,.01,10)
    def evaluate(self):return {}
    def _get_obs_extra(self,info):return {}
    def _load_agent(self,options):super()._load_agent(options,sapien.Pose(base))
    def _load_scene(self,options):
        self.objects={};self.metadata={};self.load_evidence={};self.asset_hashes={}
        material=sapien.physx.PhysxMaterial(static_friction=.5,dynamic_friction=.5,restitution=0)
        for o in manifest['objects']:
            name=o['id'];folder=ASSETS/'assets'/name;paths=sorted((folder/'coacd32').glob('*.obj')) if o['dynamic'] else sorted(folder.glob('*.obj'))
            b=self.scene.create_actor_builder()
            for p in paths:b.add_convex_collision_from_file(str(p),material=material,density=1000)
            visual=Path(o['visual']);visual=visual if visual.is_absolute() else ROOT/visual
            # Authorized original GLB is judge-only, never mounted in the Skill.
            assert visual.resolve().is_relative_to(ROOT) and visual.is_file()
            original_visual=visual
            if os.environ.get('TRACK2_VISUAL_CACHE_DIR'):
                cache=Path(os.environ['TRACK2_VISUAL_CACHE_DIR']).resolve(strict=True);assert cache.is_relative_to(WORK)
                visual=cache/(name+'.glb');assert visual.is_file()
            b.add_visual_from_file(str(visual));self.asset_hashes[name]=hashlib.sha256(visual.read_bytes()).hexdigest()
            pose=o['actor_pose_m'];b.set_initial_pose(sapien.Pose(pose['p'],pose['q_wxyz']))
            a=b.build(name=name) if o['dynamic'] else b.build_static(name=name)
            if o['dynamic']:
                params=json.loads((ASSETS/'physics'/name/'matched_parameters.json').read_text());body=a._bodies[0]
                body.mass=params['mass'];body.inertia=params['inertia'];body.cmass_local_pose=sapien.Pose(params['cmass_p'],params['cmass_q'])
            assert len(a._bodies[0].collision_shapes)==len(paths)
            self.objects[name]=a;self.metadata[name]={'pivot_m':o['canonical_pivot_zup_m'],'pivot_scaled':True};self.load_evidence[name]={'dynamic':o['dynamic']}
    def _initialize_episode(self,env_idx,options):
        self.agent.reset(REST);self.agent.robot.set_pose(sapien.Pose(base))
        for o in manifest['objects']:
            p=o['actor_pose_m'];a=self.objects[o['id']];a.set_pose(sapien.Pose(p['p'],p['q_wxyz']))
            if o['dynamic']:a.set_linear_velocity([0,0,0]);a.set_angular_velocity([0,0,0])
def factory():
    return CameraEnv(num_envs=1,robot_uids=PiperPrototype.uid,control_mode='pd_joint_pos',sim_backend='cpu',render_backend='sapien_cuda:0',obs_mode='none',reward_mode='none',sim_config={'sim_freq':240,'control_freq':20},enhanced_determinism=True)
def rgb(env):
    # CPU PyTorch cannot consume CUDA image tensors. SAPIEN's documented host
    # readback works while rendering remains strictly on the chosen GPU.
    env.scene.update_render();camera=env._human_render_cameras['ttg_rgb'];camera.capture()
    value=camera.camera._render_cameras[0].get_picture('Color')
    if value.dtype!=np.uint8:value=np.clip(value[:,:,:3]*255,0,255).astype(np.uint8)
    return value[:,:,:3]
skill='''import base64,hashlib,struct
def initialize(context):
    assert 'seed' not in context
def run(observation,tools):
    for o in [observation,tools.observe()]:
        assert set(o)<= {'robot','task_inputs','control_steps','simulation_time_s','first_frame','camera_status'}
        f=o['first_frame'];b=base64.b64decode(f['data_base64'],validate=True)
        assert b[:8]==b'\\x89PNG\\r\\n\\x1a\\n' and hashlib.sha256(b).hexdigest()==f['sha256']
        assert struct.unpack('>II',b[16:24])==(160,120)
        assert o['camera_status']=='PASS' and f['source']=='trusted_camera_rgb'
    q=observation['robot']['joint_position'];tools.piper_gripper_control(q[6]-q[7])
    return {'status':'completed','rgb_received':True}
def shutdown(context):pass
'''
a=None;proc=None;result={'status':'FAIL','actual_TTG001_RGB':False,'physical_gpu':'authorized GPU 0','physical_parameters_changed':False}
try:
    path=OUT/'internal_camera_skill.zip'
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as z:
        z.writestr('camera_skill/manifest.json',json.dumps({'skill_name':'internal_rgb_boundary_test','version':'2.0-candidate','entrypoint':'main.py','api_version':'ctrc-skill-v1'}))
        z.writestr('camera_skill/main.py',skill)
    package=unpack_skill(path,OUT/'skill_staging')
    shapes=[];hashes=[];stats=[]
    for seed in (0,1,2):
        provider=ObservationProvider(approved_fields=('text_instruction',),camera=rgb,camera_required=True)
        a=ManiSkillSimulatorAdapter(factory,snapshot=physical_snapshot,observation_provider=provider,public_fields=('text_instruction',),max_steps=4)
        a.reset(seed=seed,public_task_inputs={'text_instruction':'Internal RGB interface test; no GT'})
        first=a.observe();assert first['first_frame']['width']==160 and first['first_frame']['height']==120
        proc=PackageExecutor(package,OUT/('skill_work'+str(seed)),isolated=True,timeout_s=10)
        proc.initialize({'skill_api_version':'ctrc-skill-v1','mode':'first','environment':'sim','timeout_s':10,'tool_call_budget':4})
        response=proc.run(first,a.dispatch_public,timeout_s=10,max_requests=4)
        assert response.get('rgb_received') is True,response
        proc.close();proc=None
        env=a._ManiSkillSimulatorAdapter__env
        for frame in range(3):
            robot=a.observe()['robot']['joint_position'];target=robot[:6];target[0]+=.015
            a.step({'joint_position':target,'gripper_opening_m':float(robot[6]-robot[7])})
            picture=rgb(env);assert picture.shape==(120,160,3) and float(picture.std())>5
            shapes.append(list(picture.shape));hashes.append(hashlib.sha256(picture.tobytes()).hexdigest());stats.append(float(picture.std()))
            if os.environ.get('TRACK2_VISUAL_CACHE_DIR'):assert float(np.mean(np.ptp(picture.astype(float),axis=2)>2))>.01
            if seed==0 and frame==0:Image.fromarray(picture).save(OUT/'TTG001_first_frame_private.png')
        visual_hashes=env.asset_hashes.copy();a.close();a=None
    result.update(status='PASS',actual_TTG001_RGB=True,reset_count=3,continuous_frames=9,frame_shapes=shapes,frame_hashes=hashes,distinct_frame_hashes=len(set(hashes)),nonblank_std=stats,isolated_skill_RGB_transfer='PASS',CPU_torch_host_readback='PASS',visual_hashes=visual_hashes,
        imagery_release='PRIVATE: original initial scene imagery not included in public evaluator ZIP',camera_pose='judge-chosen diagnostic look_at; official camera config pending',production_ready=False)
except Exception as e:
    (OUT/'private_camera_error.log').write_text(traceback.format_exc());result.update(error=type(e).__name__,message=str(e)[:400])
finally:
    if proc is not None:proc.close()
    if a is not None:a.close()
(OUT/'camera_result_sanitized.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
