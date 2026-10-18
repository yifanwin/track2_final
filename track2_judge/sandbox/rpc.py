"""Bounded JSON lines RPC; contestant code never imports the adapter."""
import json, os, queue, subprocess, sys, threading, time
from pathlib import Path
from track2_judge.util.common import plain

class SkillTimeout(TimeoutError): pass


class BudgetTimeout(SkillTimeout):
    """**正常的预算耗尽** —— 与链路故障区分（2026-10-18）。

    触发源全部是**比赛定义的预算**：episode 墙钟（600 s）、
    RPC/控制步数预算。语义是「到时间了 → 停止执行 → 冻结当前物理状态 → 照常评分」，
    所以它**不是**执行异常，`ResultGate` 应当放行。

    ⚠️ 刻意与三类**基础设施故障**分开，它们仍然阻断：
      * `ActionStoppedUnknown` —— 动作超时且**物理停没停说不清**（Q13 的 unknown）
      * `ActionTimeout` —— 动作超时但**已确认物理已停**（围栏已释放、可判）
      * 其余 Exception —— 丢包 / 503 / 协议错，一律 infrastructure
    """


class SecurityNotReady(RuntimeError): pass

def encode(value): return json.dumps(plain(value),allow_nan=False,separators=(',',':'))+'\n'
def decode(line):
    return json.loads(line,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))

class SkillProcess:
    def __init__(self, skill, work, *, isolated=True, trusted=False, package_root=None, entrypoint=None, spec=None):
        skill=Path(skill).resolve(); work=Path(work).resolve(); work.mkdir(parents=True,exist_ok=True)
        if not isolated and not trusted: raise SecurityNotReady('SECURITY_NOT_READY: trusted internal tests only')
        # ⚠️ 先给默认值 —— 下面只在 `if isolated:` 分支里赋值，
        #    非隔离路径读到它们会 UnboundLocalError（踩过，表现成一件无关的
        #    安全测试 rc=1，很容易误判成安全回归）。
        runtime=None; spec_path=None
        if isolated:
            if sys.platform!='linux': raise SecurityNotReady('SECURITY_NOT_READY: Linux namespaces required')
            # ⚠️ spec JSON 必须在 **chroot 之前**写到一个宿主可见的位置 ——
            #    bootstrap 一开始（还没 chroot）读它，之后那个路径就不可见了。
            #    同理 stdlib 档的 runtime 拷贝也走 work/。
            if spec is not None and getattr(spec,'profile','')=='official':
                spec_path=str(spec.write(work/'sandbox_spec.json'))
            else:
                # stdlib 档：仍先拷一份私有 stdlib（旧行为，21 项回归跑的就是它）
                from track2_judge.sandbox.runtime import prepare_runtime
                runtime=prepare_runtime()
            root=work/'rootfs'; root.mkdir(exist_ok=True)
            cmd=['unshare','--user','--map-root-user','--mount']
            # ⚠️ 官方档**不加 `--net`**，即**共用宿主的网络命名空间**。
            #
            #    这一点容易想反：新建 netns 里 `lo` 是自己那条 loopback，
            #    `127.0.0.1:19110` 指的是**沙箱自己**，连不到宿主上的 gateway。
            #    所以「只能访问 Gateway 端口」只能靠「不隔离网络 + 只放行
            #    socket/connect」实现，并**接受沙箱能看到宿主能到的一切网络**。
            #
            #    这是本方案隔离强度最低的一处（`Track2_完整方案.md` §4.7.4
            #    明确写「本机评测先放宽，安全强化留到多机部署」）。
            #    后续收严的路径：给沙箱配 veth + 只放行 gateway 的地址段，
            #    或者把 HTTP 换回 JSON-lines 桥接（那样就不用放开网络）。
            if spec is None or not getattr(spec,'allow_network',False):
                cmd.append('--net')
            # ⚠️ `--mount-proc` 是 **CUDA 能不能用**的前提，不是可选项。
            #    `--pid` 之后若 `/proc` 仍是宿主那份，驱动 `cuInit` 返回
            #    304（CUDA_ERROR_OPERATING_SYSTEM），表现为
            #    `torch.cuda.is_available() == False` —— 看起来像「没插显卡」。
            #    让 unshare 在**建完 PID 命名空间后立刻**挂一份新 procfs 就正常
            #    （实测：加 `--mount-proc` 通过，不加必然 304）。
            cmd += ['--pid','--mount-proc','--fork','--kill-child=SIGKILL',
                 '/usr/bin/python3','-I','-B',str(Path(__file__).with_name('bootstrap.py')),str(root),str(skill)]
            if package_root is not None:cmd += [str(Path(package_root).resolve()),entrypoint]
        else: cmd=[sys.executable,'-I','-B',str(skill)]
        if not isolated and package_root is not None:cmd += ['--entrypoint',str(Path(package_root).resolve()/entrypoint)]
        # ⚠️ 外层（chroot 之前）那个 `/usr/bin/python3` **保持不动** ——
        #    bootstrap.py 只用标准库（spec JSON 也自己解析），所以宿主的 3.10 足够。
        #    沙箱**里面**被 exec 的才是官方 runtime 的解释器（bind 到 /usr/bin/python3）。
        #    混淆这两层会得到「3.10 跑 3.12 的 site-packages」这种莫名其妙的错误。
        self.p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                                text=True,encoding='utf8',cwd=work,close_fds=True,
                                env={'PATH':os.environ.get('PATH',''),'SystemRoot':os.environ.get('SystemRoot',''),
                                     'PYTHONDONTWRITEBYTECODE':'1',
                                     **({'TRACK2_SANDBOX_RUNTIME':str(runtime)} if isolated and runtime else {}),
                                     **({'TRACK2_SANDBOX_SPEC':spec_path} if spec_path else {})},start_new_session=True)
        self.q=queue.Queue(maxsize=16)
        def reader():
            try:
                while True:
                    line=self.p.stdout.readline(65538)
                    if not line: self.q.put(None,timeout=.5); break
                    if len(line)>65536 or not line.endswith('\n'): self.q.put(None,timeout=.5); self.p.kill(); break
                    self.q.put(line,timeout=.5)
            except queue.Full:
                self.p.kill()
            except Exception: pass
        threading.Thread(target=reader,daemon=True).start()
    def send(self,value,timeout_s=2):
        # A hostile Skill can request a large approved image then stop reading.
        # Bound pipe writes as well as reads; never block the judge indefinitely.
        payload=encode(value)
        if len(payload.encode('utf8'))>3*1024**2:raise ValueError('outbound RPC budget')
        done=threading.Event();errors=[]
        def write():
            try:self.p.stdin.write(payload);self.p.stdin.flush()
            except (OSError,ValueError) as exc:errors.append(exc)
            finally:done.set()
        threading.Thread(target=write,daemon=True).start()
        if not done.wait(max(.001,min(2,timeout_s))):
            self.close();raise SkillTimeout('Skill not reading bounded RPC')
        if errors:raise RuntimeError('Skill input unavailable') from None
    def receive(self,timeout_s):
        try:line=self.q.get(timeout=max(.001,timeout_s))
        # 2026-10-18：调用方给的截止就是**比赛预算**（episode 墙钟 / 控制步预算），
        #   耗尽属可评分终态 —— 归 `BudgetTimeout`（`SkillTimeout` 的子类）。
        except queue.Empty:raise BudgetTimeout('skill deadline') from None
        if line is None:raise RuntimeError('skill process unavailable')
        try:message=decode(line)
        except (ValueError,RecursionError):raise ValueError('invalid JSON RPC') from None
        if type(message) is not dict:raise ValueError('RPC object required')
        return message
    def run(self, observation, dispatch, timeout_s=5, max_requests=2400):
        deadline=time.monotonic()+timeout_s; count=0
        self.send({'observation':observation},timeout_s)
        while True:
            message=self.receive(deadline-time.monotonic())
            if message=={'done':True}: return
            if set(message)!={'method','payload'}: raise ValueError('RPC envelope rejected')
            count+=1
            if count>max_requests: raise BudgetTimeout('RPC budget')
            # No exception text containing judge paths/state is returned to Skill.
            result=dispatch(message['method'],message['payload'])
            self.send({'result':result})
            if time.monotonic()>deadline: raise BudgetTimeout('skill deadline')
    def close(self):
        if sys.platform=='linux':
            import signal
            try:
                os.killpg(self.p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        elif self.p.poll() is None:self.p.kill()
        self.p.wait(timeout=5)
        self.p.stdin.close();self.p.stdout.close()
