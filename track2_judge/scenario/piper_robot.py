import numpy as np
import sapien
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent
from track2_judge.util.common import ROOT

REST=np.array([0,1.2,-1.2,0,.5,0,.04,-.04],dtype=np.float32)
@register_agent()
class PiperPrototype(BaseAgent):
    uid='piper_track2_prototype_v1'
    urdf_path=str(ROOT/'model/piper_sim.urdf')
    keyframes={'rest':Keyframe(pose=sapien.Pose(),qpos=REST)}
    # Retain self collision; adjacent links excluded by loader's normal semantics.
    disable_self_collisions=False
    urdf_config={'_materials':{'finger':{'static_friction':.5,'dynamic_friction':.5,'restitution':0}},'link':{'gripper_link1':{'material':'finger'},'gripper_link2':{'material':'finger'}}}
    @property
    def _controller_configs(self):
        return {'pd_joint_pos':PDJointPosControllerConfig(
            joint_names=['joint'+str(i) for i in range(1,7)]+['gripper_joint1','gripper_joint2'],
            lower=None,upper=None,stiffness=[2000]*6+[2000]*2,damping=[100]*6+[50]*2,
            force_limit=[100]*6+[15]*2,normalize_action=False,interpolate=True)}
    def _after_init(self):
        links={l.name:l for l in self.robot.get_links()}
        self.tcp=links['diagnostic_tcp']; self.fingers=[links['gripper_link1'],links['gripper_link2']]
