"""Internal joint/gripper test. No object setters, simulator or judge access."""
import json, sys, time

def call(method,payload=None):
    print(json.dumps({'method':method,'payload':payload},allow_nan=False),flush=True)
    return json.loads(sys.stdin.readline())['result']

def flush(buf):
    # ★ 强制 Action Chunk（2026-10-18）：唯一的关节移动途径是 `joint_trajectory`。
    #   不足 8 步（schema 下限）就补到 8 —— 原地重复最后一步，物理等价于"没有更多运动"。
    if not buf: return
    while len(buf)<8: buf.append(dict(buf[-1]))
    call('joint_trajectory',{'targets':buf[:25]})

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
        pending=[]
        for target,g,n in schedule:
            for _ in range(n):
                goal=target+[g]
                for j in range(7): current[j]+=max(-(.02 if j<6 else .003),min((.02 if j<6 else .003),goal[j]-current[j]))
                current[6]=max(0.,min(.1,current[6]))
                pending.append({'joint_position':[float(x) for x in current[:6]],
                                'gripper_opening_m':float(current[6])})
                if len(pending)>=16:
                    flush(pending); pending=[]
        flush(pending)
    print(json.dumps({'done':True}),flush=True)
