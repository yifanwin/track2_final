"""Explicit synthetic adapter harness; NEVER physical competition evidence."""
import copy
from types import SimpleNamespace
import numpy as np
from track2_judge.toolapi.dispatch import ManiSkillSimulatorAdapter
from track2_judge.scoring.scorer_bridge import v3

SETTINGS={'units':'centimeters','up_axis':'Z','rotation_convention':'absolute_quaternion_xyzw','scene_offset':[0,0,0]}

def suite():
    target={'scene_settings':SETTINGS,'objects':[{'instance':n,'position':[x,20,5],'rotation':[0,0,0,1]} for n,x in [('a',15),('b',35),('c',55),('background',85)]]}
    initial=copy.deepcopy(target)
    for obj in initial['objects'][:3]: obj['position'][1]+=20
    task=v3.make_task('synthetic/internal',target,initial)
    return {**v3.report_meta(),'kind':'mesa_ctrc_suite','scope':'moved','moved_policy':v3.moved_policy(),'synthetic_demo_only':True,'config':v3.default_config(),'tasks':[task]}

class FakeEnv:
    control_timestep=.05
    def __init__(self, task, count):
        self.task=task;self.count=count;self.steps=0
        self.q=np.array([0.,1.2,-1.2,0.,.5,0.,.04,-.04])
        self.agent=SimpleNamespace(robot=SimpleNamespace(get_qpos=lambda:self.q.copy(),get_qvel=lambda:np.zeros(8),get_qlimits=lambda:np.array([[-2.6,2.6],[0,3.14],[-2.96,0],[-1.74,1.74],[-1.22,1.22],[-2.09,2.09],[0,.05],[-.05,0]])),tcp=SimpleNamespace(pose=SimpleNamespace(p=np.zeros(3),q=np.array([1,0,0,0]))))
    def reset(self,seed): self.seed=seed
    def step(self,action):self.steps+=1;self.q=action.copy()
    def close(self):pass
    def snapshot(self):
        layout=copy.deepcopy(self.task['initial_layout'])
        for obj in layout['objects']:obj['position'][0]+=self.seed*.1
        # Synthetic state transition is exclusively a harness behavior, never an external action.
        if self.steps:
            for i in range(self.count):layout['objects'][i]=copy.deepcopy(self.task['target_layout']['objects'][i])
        return {'objects':{o['instance']:{'actor_p_m':np.array(o['position'])/100,'q_wxyz':[1,0,0,0]} for o in layout['objects']},'synthetic_only':True}

def factory(count=0,max_steps=2400):
    def make(task,seed):
        env=FakeEnv(task,count)
        return ManiSkillSimulatorAdapter(lambda:env,max_steps=max_steps,snapshot=lambda e:e.snapshot(),metadata={o['instance']:{'pivot_m':[0,0,0]} for o in task['initial_layout']['objects']},scene_settings=SETTINGS,public_fields=('text_instruction',))
    make.source='SYNTHETIC: no physics'
    return make
