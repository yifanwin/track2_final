"""Bounded JSON lines RPC; contestant code never imports the adapter."""
import json, os, queue, subprocess, sys, threading, time
from pathlib import Path
from track2_judge.util.common import plain

class SkillTimeout(TimeoutError): pass
class SecurityNotReady(RuntimeError): pass

def encode(value): return json.dumps(plain(value),allow_nan=False,separators=(',',':'))+'\n'
def decode(line):
    return json.loads(line,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))

class SkillProcess:
    def __init__(self, skill, work, *, isolated=True, trusted=False, package_root=None, entrypoint=None):
        skill=Path(skill).resolve(); work=Path(work).resolve(); work.mkdir(parents=True,exist_ok=True)
        if not isolated and not trusted: raise SecurityNotReady('SECURITY_NOT_READY: trusted internal tests only')
        if isolated:
            if sys.platform!='linux': raise SecurityNotReady('SECURITY_NOT_READY: Linux namespaces required')
            from track2_judge.sandbox.runtime import prepare_runtime
            runtime=prepare_runtime()
            root=work/'rootfs'; root.mkdir(exist_ok=True)
            cmd=['unshare','--user','--map-root-user','--mount','--net','--pid','--fork','--kill-child=SIGKILL',
                 '/usr/bin/python3','-I','-B',str(Path(__file__).with_name('bootstrap.py')),str(root),str(skill)]
            if package_root is not None:cmd += [str(Path(package_root).resolve()),entrypoint]
        else: cmd=[sys.executable,'-I','-B',str(skill)]
        if not isolated and package_root is not None:cmd += ['--entrypoint',str(Path(package_root).resolve()/entrypoint)]
        self.p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                                text=True,encoding='utf8',cwd=work,close_fds=True,
                                env={'PATH':os.environ.get('PATH',''),'SystemRoot':os.environ.get('SystemRoot',''),
                                     'PYTHONDONTWRITEBYTECODE':'1', **({'TRACK2_SANDBOX_RUNTIME':str(runtime)} if isolated else {})},start_new_session=True)
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
        except queue.Empty:raise SkillTimeout('skill deadline') from None
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
            if count>max_requests: raise SkillTimeout('RPC budget')
            # No exception text containing judge paths/state is returned to Skill.
            result=dispatch(message['method'],message['payload'])
            self.send({'result':result})
            if time.monotonic()>deadline: raise SkillTimeout('skill deadline')
    def close(self):
        if sys.platform=='linux':
            import signal
            try:
                os.killpg(self.p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        elif self.p.poll() is None:self.p.kill()
        self.p.wait(timeout=5)
        self.p.stdin.close();self.p.stdout.close()
