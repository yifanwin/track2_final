"""Create only internal synthetic compliant test Skill ZIPs, never public release."""
import json,zipfile
from pathlib import Path

ROOT=Path(__file__).parent

SOURCE='''import os,time
mode='noop'
def initialize(context):
    if context.get('skill_api_version')!='ctrc-skill-v1' or 'seed' in context:raise ValueError('context boundary')
def run(observation,tools):
    mode=observation['task_inputs'].get('text_instruction','noop')
    if mode=='timeout':time.sleep(30)
    if mode=='illegal':tools.call('set_pose',{'instance':'synthetic','position':[1,2,3]})
    if mode=='private':tools.call('get_final_state',{})
    if mode=='error':return {'status':'error'}
    if mode=='forge':return {'status':'completed','score':100,'final_layout':{'fabricated':True}}
    if mode=='blocked_reader':
        os.write(1,b'{"type":"tool_call","tool":"observe","arguments":{}}\\n');time.sleep(30)
    if mode=='move':
        obs=tools.observe();q=obs['robot']['joint_position'];target=q[:6];opening=q[6]-q[7]
        # 强制 Action Chunk：攒成 8 步一次提交（原地重复补足下限）
        targets=[]
        for i in range(8):
            target[0]+=.01
            targets.append({'joint_position':list(target),'gripper_opening_m':float(opening)})
        tools.joint_trajectory(targets)
        tools.piper_gripper_control(max(0.,opening-.002))
    return {'status':'completed'}
def shutdown(context):pass
'''

def create(output=None):
    root=Path(output) if output else ROOT/'internal_packages/v2_minimal'
    root.mkdir(parents=True,exist_ok=True)
    manifest={'skill_name':'internal_piper_interface_test','version':'2.0.0-candidate','entrypoint':'main.py','api_version':'ctrc-skill-v1','tool_profile':'piper-joint-v2-candidate'}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8');(root/'main.py').write_text(SOURCE,encoding='utf8')
    (root/'README.md').write_text('Internal synthetic judge test only. NOT a public reference Skill. No task GT included.\n',encoding='utf8')
    path=root.with_suffix('.zip')
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.iterdir()):z.write(p,root.name+'/'+p.name)
    return path

if __name__=='__main__':print(create())
