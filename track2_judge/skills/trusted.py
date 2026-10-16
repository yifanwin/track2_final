"""Internal joint/gripper test. No object setters, simulator or judge access."""
import json, sys, time

def call(method,payload=None):
    print(json.dumps({'method':method,'payload':payload},allow_nan=False),flush=True)
    return json.loads(sys.stdin.readline())['result']

for line in sys.stdin:
    obs=json.loads(line)['observation']; inputs=obs['task_inputs']; kind=inputs.get('text_instruction','noop')
    if kind=='timeout': time.sleep(30)
    elif kind=='illegal': call('step',{'set_pose':[1,2,3]})
    elif kind=='private': call('get_final_state')
    elif kind=='forge':
        print(json.dumps({'done':True,'score':100}),flush=True); continue
    elif kind=='oversized':
        print('x'*70000,flush=True);time.sleep(30)
    elif kind!='noop':
        q=obs['robot']['joint_position']; current=q[:6]+[q[6]-q[7]]
        # Last commanded target avoids slew violations caused by tracking lag.
        schedule=[([q[0]+.15]+q[1:6],.08,40),([q[0]+.15]+q[1:6],0.,40),
                  ([q[0]+.15,q[1]+.1,q[2]-.1]+q[3:6],0.,40), (q[:6],.08,40)]
        for target,g,n in schedule:
            for _ in range(n):
                goal=target+[g]
                for j in range(7): current[j]+=max(-(.02 if j<6 else .003),min((.02 if j<6 else .003),goal[j]-current[j]))
                current[6]=max(0.,min(.1,current[6]))
                call('step',{'joint_position':current[:6],'gripper_opening_m':current[6]})
    print(json.dumps({'done':True}),flush=True)
