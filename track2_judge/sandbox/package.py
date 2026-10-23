"""Reuse original manifest/AST contract, with bounded safe ZIP unpack and real sandbox."""
import hashlib, os, shutil, stat, time, zipfile,json
from pathlib import Path, PurePosixPath
from track2_judge.runner.skill_manifest import SkillPackage
from track2_judge.runner.version import SKILL_API_VERSION
from track2_judge.sandbox.rpc import BudgetTimeout, SkillError, SkillProcess, SkillTimeout

def unpack_skill(source,destination):
    source=Path(source).resolve();destination=Path(destination).resolve()
    if destination.exists():raise ValueError('Skill staging destination must be new')
    destination.mkdir(parents=True)
    if not source.is_file() or source.suffix.lower()!='.zip':raise ValueError('Skill ZIP required')
    with zipfile.ZipFile(source) as archive:
        members=archive.infolist()
        if len(members)>200 or sum(m.file_size for m in members)>20*1024**2:raise ValueError('ZIP budget exceeded')
        seen=set()
        for member in members:
            name=member.filename;path=PurePosixPath(name)
            mode=member.external_attr>>16
            if '\\' in name or ':' in name or path.is_absolute() or '..' in path.parts or stat.S_ISLNK(mode):raise ValueError('ZIP path/symlink rejected')
            if name in seen:raise ValueError('duplicate ZIP entry')
            seen.add(name)
            if member.file_size>5*1024**2 or (member.compress_size and member.file_size/member.compress_size>200):raise ValueError('ZIP expansion rejected')
            dest=(destination/Path(*path.parts)).resolve()
            if destination not in dest.parents:raise ValueError('ZIP path outside staging')
            if member.is_dir():dest.mkdir(parents=True,exist_ok=True);continue
            if stat.S_IFMT(mode) not in (0,stat.S_IFREG):raise ValueError('special file rejected')
            dest.parent.mkdir(parents=True,exist_ok=True)
            with archive.open(member) as src,dest.open('xb') as out:
                size=0
                while True:
                    block=src.read(65536)
                    if not block:break
                    size+=len(block)
                    if size>member.file_size:raise ValueError('ZIP declared size mismatch')
                    out.write(block)
            if size!=member.file_size:raise ValueError('ZIP incomplete file')
    manifests=list(destination.rglob('manifest.json'))
    if len(manifests)!=1:raise ValueError('exactly one manifest required')
    root=manifests[0].parent
    # All files must belong to the chosen root, avoiding hidden secondary packages.
    if any(root not in p.parents for p in destination.rglob('*') if p.is_file()):raise ValueError('files outside Skill root')
    # Check lexical entrypoint containment before the legacy validator's is_file
    # call, so a malicious manifest cannot probe judge paths by existence/error.
    if manifests[0].stat().st_size>65536:raise ValueError('manifest budget exceeded')
    try:manifest=json.loads(manifests[0].read_text(encoding='utf-8-sig'))
    except (ValueError,RecursionError):raise ValueError('manifest parse rejected') from None
    entry=manifest.get('entrypoint') if isinstance(manifest,dict) else None
    if not isinstance(entry,str) or not entry or '\\' in entry or ':' in entry or PurePosixPath(entry).is_absolute() or '..' in PurePosixPath(entry).parts:
        raise ValueError('manifest entrypoint containment rejected')
    entry_file=root/Path(*PurePosixPath(entry).parts)
    if not entry_file.is_file() or entry_file.stat().st_size>128*1024:raise ValueError('entrypoint validation budget exceeded')
    try:package=SkillPackage(root)
    except (ValueError,RuntimeError,RecursionError):raise ValueError('manifest/lifecycle validation rejected') from None
    if any(p.is_symlink() for p in root.rglob('*')):raise ValueError('symlink rejected')
    return package

class PackageExecutor:
    def __init__(self,package,work,*,isolated=True,trusted=False,timeout_s=10,spec=None):
        self.package=package;self.closed=False;self.lifecycle=[];self.default_timeout=timeout_s;self.tool_service_s=0.
        # `spec` 决定沙箱档（第 4 步 / Q18）：
        #   None → stdlib 档（第 4 步之前的默认，21 项回归跑的就是它）
        #   official_spec(...) → 官方 runtime 档，torch + CUDA 可用
        self.process=SkillProcess(Path(__file__).with_name('package_host.py'),work,isolated=isolated,trusted=trusted,
            package_root=package.root,entrypoint=package.entrypoint.relative_to(package.root).as_posix(),spec=spec)
        try:
            ready=self.process.receive(timeout_s)
            if ready.get('type')!='ready' or set(ready)-{'type','pid'}:raise RuntimeError('Skill readiness rejected')
            self.lifecycle.append('ready')
        except Exception:self.process.close();raise
    def initialize(self,context,timeout_s=None):
        if 'seed' in context:raise ValueError('private random seed must not enter Skill context')
        self.process.send({'type':'init','context':context})
        result=self.process.receive(timeout_s or self.default_timeout)
        if result!={'type':'init_result','result':{'ok':True}}:raise RuntimeError('Skill initialization failed')
        self.lifecycle.append('initialize')
    def run(self,observation,dispatch,timeout_s=10,max_requests=2400):
        self.process.send({'type':'run','observation':observation});deadline=time.monotonic()+timeout_s;count=0;hard_deadline=time.monotonic()+600
        while True:
            remaining=min(deadline,hard_deadline)-time.monotonic()
            if remaining<=0:raise BudgetTimeout('package deadline')
            msg=self.process.receive(remaining)
            if msg.get('type')=='tool_call' and set(msg)=={'type','tool','arguments'}:
                count+=1
                if count>max_requests:raise BudgetTimeout('package RPC budget')
                if type(msg['tool']) is not str or type(msg['arguments']) is not dict:raise ValueError('malformed package tool')
                name=msg['tool'];payload=msg['arguments']
                # Same ToolClient lifecycle; legacy object teleport calls stay forbidden.
                # ★ 2026-10-19 修：`observe` 的 payload 是**关键字参数**（如 `include_rgb`），
                #   不是「非法参数」。ToolSpec 里 `observe` 本就声明了 `include_rgb`
                #   （`schema.py`），`package_host.ToolClient.observe(**kwargs)` 与
                #   `SandboxBridge` 也都按 kwargs 透传 —— 只有这里曾硬拒任何 payload，
                #   导致**照文档写 `tools.observe(include_rgb=True)` 的合规提交被判
                #   `ValueError: observe has no arguments` → 整 case BLOCKED**。
                #   （评测器自带的 skill/测试都调无参 `observe()`，所以一直没暴露。）
                if name=='observe':
                    payload=payload or {}
                service_start=time.monotonic()
                try:result=dispatch(name,payload)
                except Exception as _exc:
                    import sys as _sys
                    print(f"[track2-package] tool {name!r} rejected: "
                          f"{type(_exc).__name__}: {_exc}", file=_sys.stderr, flush=True)
                    self.process.send({'type':'tool_result','ok':False,'error':'tool request rejected'})
                    raise
                finally:
                    service=time.monotonic()-service_start;self.tool_service_s+=service;deadline+=service
                self.process.send({'type':'tool_result','ok':True,'result':result})
            elif msg.get('type')=='run_result' and set(msg)=={'type','result'} and isinstance(msg['result'],dict):
                self.lifecycle.append('run')
                if msg['result'].get('status')=='error':raise RuntimeError('Skill reported failure')
                # All claimed state, scores and evidence are ignored. Parent judges physics.
                return msg['result']
            else:
                # ⚠️ 把**收到的是什么**带进错误里。原来只写
                #    `package execution/protocol rejected`，看不出是
                #    宿主报了错、还是协议真的被违反 —— 排查时要重跑一遍
                #    才能加打印。这里带上有界摘要（不含裁判路径/私有状态，
                #    因为摘要来自**子进程**发给我们的消息）。
                kind=str(msg.get('type'))[:32]
                category=str(msg.get('category') or msg.get('error') or '')[:120]
                # ★ F2（2026-10-18）：**选手包自己抛了异常**（子进程发来的
                #   `error` 消息）与「协议/链路坏了」是两回事：
                #   * 选手包崩 → 该 episode 记 `skill_error`，**仍按终态评分**（可评分）
                #   * 协议/链路坏 → 仍是 `infrastructure_timeout`（阻断）
                #   不分开的话，迁移期一个没改过的老包会让 50 个 case 全 BLOCKED。
                if kind == 'error':
                    raise SkillError('contestant package error: category=%s' % category)
                raise RuntimeError('package execution/protocol rejected: '
                                   'type=%s category=%s' % (kind,category))
    def close(self):
        if self.closed:return
        self.closed=True
        try:
            if self.process.p.poll() is None:
                self.process.send({'type':'shutdown'})
                result=self.process.receive(min(2,self.default_timeout))
                if result=={'type':'shutdown_result','result':{'ok':True}}:self.lifecycle.append('shutdown')
        except (OSError,RuntimeError,ValueError,TimeoutError):pass
        finally:self.process.close()
    def abort(self):
        self.closed=True;self.lifecycle.append('terminated');self.process.close()
