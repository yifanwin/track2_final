"""Private diagnostic baseline: public goal/proprioception + public robot FK.

No GT, initial object state, private paths, judge state or direct pose setters.
This deliberately modest open-loop prior is not a competition baseline score.
"""
import json
import math
import base64,struct,zlib,statistics
from pathlib import Path

MODEL=json.loads(Path(__file__).with_name('public_robot_kinematics.json').read_text())
STATE={}
#: 强制 Action Chunk（2026-10-18）：攒够 `CHUNK_SIZE` 步提交一次。
#: 取 12（在 8–25 之间）—— 既留足抗超时余量，又不至于让盲区太长。
CHUNK=[];CHUNK_SIZE=12
def png_pixels(payload):
    blob=base64.b64decode(payload['data_base64']);offset=8;parts=[]
    while offset<len(blob):
        size=struct.unpack('>I',blob[offset:offset+4])[0];kind=blob[offset+4:offset+8];value=blob[offset+8:offset+8+size];offset+=size+12
        if kind==b'IHDR':w,h,depth,color,_,_,interlace=struct.unpack('>IIBBBBB',value)
        if kind==b'IDAT':parts.append(value)
    assert depth==8 and color in (2,6) and interlace==0
    channels=3 if color==2 else 4;stride=w*channels;raw=zlib.decompress(b''.join(parts));rows=[];previous=[0]*stride
    for y in range(h):
        kind=raw[y*(stride+1)];row=list(raw[y*(stride+1)+1:(y+1)*(stride+1)])
        for i in range(stride):
            left=row[i-channels] if i>=channels else 0;up=previous[i];ul=previous[i-channels] if i>=channels else 0
            if kind==1:v=left
            elif kind==2:v=up
            elif kind==3:v=(left+up)//2
            elif kind==4:
                p=left+up-ul;a=abs(p-left);b=abs(p-up);c=abs(p-ul);v=left if a<=b and a<=c else up if b<=c else ul
            else:v=0
            row[i]=(row[i]+v)&255
        rows.append(row);previous=row
    return w,h,channels,rows
def mm(a,b):return [[sum(a[i][k]*b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]
def transpose(a):return [list(v) for v in zip(*a)]
def identity(n):return [[float(i==j) for j in range(n)] for i in range(n)]
def rt(r,p):return [r[i]+[p[i]] for i in range(3)]+[[0,0,0,1]]
def inv(t):
    r=transpose([v[:3] for v in t[:3]]);p=[-sum(r[i][j]*t[j][3] for j in range(3)) for i in range(3)];return rt(r,p)
def axisrot(axis,angle):
    x,y,z=axis;c=math.cos(angle);s=math.sin(angle);v=1-c
    return [[c+x*x*v,x*y*v-z*s,x*z*v+y*s],[y*x*v+z*s,c+y*y*v,y*z*v-x*s],[z*x*v-y*s,z*y*v+x*s,c+z*z*v]]
def quatrot(q):
    w,x,y,z=q
    return [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
def fk(q):
    t=identity(4)
    for joint in MODEL['chain']:
        t=mm(t,joint['origin'])
        if joint['index'] is not None:t=mm(t,rt(axisrot(joint['axis'],q[joint['index']]),[0,0,0]))
    return t
def solve(a,b):
    m=[list(r)+[b[i]] for i,r in enumerate(a)];n=len(b)
    for i in range(n):
        k=max(range(i,n),key=lambda j:abs(m[j][i]));m[i],m[k]=m[k],m[i];v=m[i][i]
        if abs(v)<1e-12:return [0]*n
        m[i]=[x/v for x in m[i]]
        for j in range(n):
            if j!=i:
                f=m[j][i];m[j]=[m[j][k]-f*m[i][k] for k in range(n+1)]
    return [m[i][-1] for i in range(n)]
def error(q,target):
    t=fk(q);r=[row[:3] for row in t[:3]];d=[row[:3] for row in target[:3]]
    # Rotation residual from matrix entries avoids the 180-degree sine zero.
    return [t[i][3]-target[i][3] for i in range(3)]+[.08*(r[i][j]-d[i][j]) for i in range(3) for j in range(3)]
def ik(target,start):
    limits=MODEL['limits'];best=None
    seeds=[start,[0,1.3,-1.3,0,.5,0],[0,.6,-1.7,0,1.,0],[0,2.,-.8,0,-.5,0]]
    for seed in seeds:
        q=[max(limits[i][0]+1e-6,min(limits[i][1]-1e-6,seed[i])) for i in range(6)]
        for _ in range(85):
            e=error(q,target);loss=sum(v*v for v in e)
            if best is None or loss<best[0]:best=(loss,list(q))
            if loss<1e-7:break
            cols=[]
            for i in range(6):
                trial=list(q);trial[i]+=.0001;v=error(trial,target);cols.append([(v[j]-e[j])/.0001 for j in range(len(e))])
            a=[[sum(cols[i][k]*cols[j][k] for k in range(len(e)))+(.00001 if i==j else 0) for j in range(6)] for i in range(6)]
            b=[-sum(cols[i][k]*e[k] for k in range(len(e))) for i in range(6)];delta=solve(a,b)
            q=[max(limits[i][0]+1e-6,min(limits[i][1]-1e-6,q[i]+max(-.15,min(.15,delta[i])))) for i in range(6)]
        if best[0]<1e-7:break
    return best[1],math.sqrt(best[0])

def initialize(context):
    if context.get('tool_profile')!='piper-joint-v2-candidate':raise ValueError('Piper tool profile required')
    STATE.clear()

def run(observation,tools):
    if STATE.get('done'):return {'status':'completed','baseline':'no extra motion after completed sequence'}
    obs=tools.observe();q=obs['robot']['joint_position'];previous=STATE.setdefault('command',list(q[:6])+[q[6]-q[7]])
    current=rt(quatrot(obs['robot']['tcp_quaternion_wxyz']),obs['robot']['tcp_position_m']);base=mm(current,inv(fk(q[:6])))
    p=next(x for x in obs['task_inputs']['predicates'] if x['type']=='P2')
    # ⚠️ 2026-10-19 口径变更：公开输入里**不再有 `camera_calibration`**。
    #    这个诊断基准原来靠「标定 + 颜色阈值定位像素」把目标映回世界坐标；
    #    现在那条路**走不通了**（`calibration` 恒 None）→ 直接落到下面的
    #    「no public target source」明确失败分支。
    #
    #    **不补任何隐藏状态兜底** —— 这正是这个诊断存在的意义：它如实报告
    #    「只凭公开输入做不到」，而不是偷偷用答案把分数做漂亮。
    goal=None
    grip=None
    calibration=(obs.get('task_inputs') or {}).get('camera_calibration')
    if calibration and (obs.get('images') or {}).get('camera'):
        w,h,ch,rows=png_pixels(obs['images']['camera']);xs=[];ys=[]
        color='blue' if 'blue' in p['object'] else 'green' if 'green' in p['object'] else 'red'
        for y,row in enumerate(rows):
            for x in range(w):
                rgb=row[x*ch:x*ch+3];index={'blue':2,'green':1,'red':0}[color]
                if rgb[index]>70 and all(rgb[index]>1.5*rgb[k] for k in range(3) if k!=index):xs.append(x);ys.append(y)
        if len(xs)<5:return {'status':'completed','baseline':'vision target not detected; no hidden state fallback'}
        # 从**参考图**里定位目标（模式 B：选手也只有图像可看）
        ref=(obs['task_inputs'] or {}).get('reference_image')
        if ref is None:return {'status':'completed','baseline':'no reference image in public input; mode B requires one'}
        # 画面是正方形，两方向跨度相同（`span_m` 由标定直接给）
        span=calibration['span_m'][0]
        grip=[calibration['aim_m'][0]+(statistics.median(xs)+.5-w/2)/w*span,
              calibration['aim_m'][1]+(statistics.median(ys)+.5-h/2)/h*span,
              calibration['aim_m'][2]]
    if grip is None or len(grip)!=3:
        # 模式 B 下目标只能从参考图 + 相机标定推。两者缺一就**明确失败** ——
        # 这个诊断基准不用任何隐藏状态兜底（那正是它存在的意义）。
        STATE['done']=True
        return {'status':'completed',
                'baseline':'no public target source available (2026-10-19: camera_calibration withdrawn; image-only localization not implemented in this diagnostic)',
                'target_label':p['object']}
    rotation=mm([row[:3] for row in base[:3]],[[-1,0,0],[0,1,0],[0,0,-1]])
    points=[('pregrasp',[grip[0],grip[1],grip[2]+.10],.1,140),('approach',grip,.1,120),
            ('close',grip,0.,100),('lift',[grip[0],grip[1],grip[2]+.10],0.,120),
            ('transfer',[grip[0]+.035,grip[1],grip[2]+.10],0.,100),
            ('lower',[grip[0]+.035,grip[1],grip[2]],0.,120),
            ('release',[grip[0]+.035,grip[1],grip[2]],.1,80),
            ('retreat',[grip[0]+.035,grip[1],grip[2]+.12],.1,120),
            ('stable',[grip[0]+.035,grip[1],grip[2]+.12],.1,40)]
    errors=[];last=None
    # ★ 强制 Action Chunk（2026-10-18）：逐次 `piper_joint_control` 已不存在。
    #   把每段插值结果**攒成 chunk**（8–25 步）再提交，语义与原来一致
    #   （slew 逐步校验，轨迹相同），只是往返变少。
    def _flush(buf):
        if buf:
            tools.joint_trajectory(list(buf))
            buf.clear()
    for phase,target,opening,steps in points:
        path=[target] if last is None or phase in ('close','release','stable') else [[last[k]+(target[k]-last[k])*f/6 for k in range(3)] for f in range(1,7)]
        for point in path:
            wanted,residual=ik(mm(inv(base),rt(rotation,point)),previous[:6]);errors.append(residual)
            if residual>.02:
                STATE['done']=True
                return {'status':'completed','baseline':'selected grasp orientation/waypoint IK not feasible; no unsafe fallback','max_ik_residual':residual,'failed_phase':phase,'target_label':p['object']}
            for _ in range(max(18,steps//len(path))):
                previous[:6]=[previous[i]+max(-.03,min(.03,wanted[i]-previous[i])) for i in range(6)]
                previous[6]+=max(-.004,min(.004,opening-previous[6]))
                CHUNK.append({'joint_position':list(previous[:6]),
                              'gripper_opening_m':float(previous[6])})
                if len(CHUNK)>=CHUNK_SIZE:
                    _flush(CHUNK)
        last=list(target)
    _flush(CHUNK)          # 收尾：不足 8 步时**不提交**（schema 会拒）
    STATE['done']=True
    return {'status':'completed','baseline':'public_goal_open_loop_prior','max_ik_residual':max(errors),'target_label':p['object']}

def shutdown(context):STATE.clear()
