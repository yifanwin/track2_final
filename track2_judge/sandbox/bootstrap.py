"""Linux rootless user/mount/network/PID namespace with minimal chroot.

只让「一份 Python 运行时 + 选中的 Skill 文件」进沙箱。
**不挂**宿主 HOME、裁判根目录、procfs、env_ready 或任何资产。

## 两档（第 4 步新增官方档）

| 档 | 运行时来源 | 网络 | 内存上限 | 谁用 |
|---|---|---|---|---|
| `stdlib`（默认） | 拷贝出来的纯标准库 | `--net` + 禁 socket | `RLIMIT_AS` 256 MiB | 不带模型的选手 |
| `official` | 官方 ML runtime（bind，只读） | 只剩 loopback + 允许 socket | **不设**（见下） | 带模型推理的选手 |

档由 `TRACK2_SANDBOX_SPEC` 指向的 JSON 决定（`sandbox/official_runtime.py`
生成）。**没有这个环境变量时走 stdlib 档，代码路径与第 4 步之前逐字相同** ——
21 项回归必须零差异。

## 两档共同的底线（不放宽）

- user / mount / PID 命名空间
- chroot 到 tmpfs，**不挂**宿主 `/`、`/mnt`、`/home`、`/proc`
- 丢光 bounding + effective/permitted/inheritable capabilities
- `no_new_privs`
- 只读挂载选手代码（`/skill`）与宿主 runtime

## 官方档为什么可以放宽这些

**网络**：不是「解开 `--net`」，而是**保留 netns 并只拉起 loopback**。
新建的 netns 里没有路由、没有对端，`connect()` 到任何非 127.0.0.1 地址
内核直接 `ENETUNREACH` —— 这是**内核保证**，比 seccomp 过滤地址可靠
（地址在内存里，不在寄存器里，seccomp 根本看不到）。

**内存**：原方案写「放宽到 32 GiB」。**那是错的** —— CUDA 版 torch 一加载就
reserve 几十 GiB 虚拟地址空间（不是常驻内存）。设任何 `RLIMIT_AS` 都会得到
「模型还没跑就 OOM」。所以官方档设 `memory_limit_bytes = None`，
改用墙钟超时 + 父进程可杀，`RLIMIT_CPU` 只给一个很宽的值兜底。

**clone/fork**：官方档**必须**放行 —— torch 一 import 就有 OpenMP 线程池。
但单独放行 `clone` 等于放行 `clone(CLONE_NEWUSER)`，所以对 `clone`/`clone3`
加**参数过滤**：flags 里带任何 `CLONE_NEW*` 就直接 `EPERM`。
`unshare` / `setns` 仍然全禁。
"""
import ctypes
import json
import os
import pathlib
import resource
import subprocess
import sys

# ───────────────────────────────────────────────── 常量

_SCMP_ACT_ALLOW = 0x7FFF0000
_SCMP_ACT_ERRNO = 0x00050001
_SCMP_ACT_ENOSYS = 0x00050026          # SCMP_ACT_ERRNO(ENOSYS) = 38
_SCMP_CMP_MASKED_EQ = 7

_CLONE_NEW_FLAGS = {
    "CLONE_NEWNS": 0x00020000,
    "CLONE_NEWCGROUP": 0x02000000,
    "CLONE_NEWUTS": 0x04000000,
    "CLONE_NEWIPC": 0x08000000,
    "CLONE_NEWUSER": 0x10000000,
    "CLONE_NEWPID": 0x20000000,
    "CLONE_NEWNET": 0x40000000,
}

# 两档都禁 —— 与旧实现逐字一致，顺序也保持
_BASE_DENY = (
    "fork", "vfork", "clone", "clone3", "unshare", "setns", "mount", "umount2",
    "pivot_root", "chroot", "ptrace", "process_vm_readv", "process_vm_writev",
    "open_by_handle_at", "socket", "connect", "bind", "listen", "accept",
    "accept4", "bpf", "perf_event_open", "keyctl", "add_key", "request_key",
)

# 官方档额外放行（只是从 _BASE_DENY 里拿掉）。
#
# ⚠️ **`bind` / `listen` 是必须放行的**，否则 CUDA 起不来。
#    NVIDIA 驱动在 `cuInit` 里会探测性地建 listener socket，
#    `bind` 被 EPERM 挡掉就直接返回 304（CUDA_ERROR_OPERATING_SYSTEM），
#    表现为 `torch.cuda.is_available() == False` —— 看起来像「没插显卡」。
#
# ⚠️ **`clone3` 也必须在这里**（虽然它最终仍被 ENOSYS 挡着，见下）。
#    libseccomp 对**同一个 syscall 的多条同优先级规则**是按**插入顺序**
#    取第一条命中的。`clone3` 若留在 `_BASE_DENY` 的无条件循环里，就会先拿到
#    一条无条件 EPERM 规则，把后面那条 ENOSYS 规则**永久遮蔽** ——
#    而 glibc 只在 `errno == ENOSYS` 时才退回 `clone`。结果是 CUDA 起不来。
#    所以要从循环里拿掉，只留 `_install_seccomp` 里那条 ENOSYS 规则。
#
# 实测（`work/scripts/debug_seccomp_cuda.py`，逐条二分）：
#   * 官方档真实 blocked 集合                     → child cuInit = 304
#   * 同一集合，只把 `clone3` 拿出去              → child cuInit = 0
#   * 单独禁 `socket` / `bind` / `listen` 任一    → 304
#   * 单独禁 `connect` / `accept` / `accept4`     → 0（可以继续禁）
_OFFICIAL_UNBLOCK = frozenset({
    "socket", "bind", "listen",                 # CUDA 驱动探测需要
    "connect",                                   # 连本机 gateway 需要
    "clone", "fork", "vfork",                    # torch 线程池
    "clone3",                                    # 走 ENOSYS，不用无条件 EPERM
})


def command(*args):
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ───────────────────────────────────────────────── seccomp


class _ScmpArgCmp(ctypes.Structure):
    _fields_ = [
        ("arg", ctypes.c_uint),
        ("op", ctypes.c_int),
        ("datum_a", ctypes.c_uint64),
        ("datum_b", ctypes.c_uint64),
    ]


def _load_seccomp():
    sec = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    sec.seccomp_init.restype = ctypes.c_void_p
    sec.seccomp_init.argtypes = [ctypes.c_uint32]
    sec.seccomp_syscall_resolve_name.restype = ctypes.c_int
    sec.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    # ⚠️ 用 `seccomp_rule_add_array` 而不是变参的 `seccomp_rule_add` ——
    #    后者在 varargs 里要按值传 `struct scmp_arg_cmp`，ctypes 传不了
    #    （会把结构体指针压栈）。`_array` 版收指针，语义清晰也安全。
    sec.seccomp_rule_add_array.restype = ctypes.c_int
    sec.seccomp_rule_add_array.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint,
        ctypes.POINTER(_ScmpArgCmp),
    ]
    sec.seccomp_load.restype = ctypes.c_int
    sec.seccomp_load.argtypes = [ctypes.c_void_p]
    return sec


def _rule(sec, ctx, action, name):
    nr = sec.seccomp_syscall_resolve_name(name.encode())
    if nr >= 0 and sec.seccomp_rule_add_array(ctx, action, nr, 0, None) != 0:
        raise RuntimeError("seccomp rule failure: " + name)


def _rule_masked(sec, ctx, action, name, arg_index, mask, value):
    """带一个参数比较的规则：`(arg[arg_index] & mask) == value` → action。"""
    nr = sec.seccomp_syscall_resolve_name(name.encode())
    if nr < 0:
        return
    comparison = _ScmpArgCmp(arg_index, _SCMP_CMP_MASKED_EQ, mask, value)
    if sec.seccomp_rule_add_array(ctx, action, nr, 1, ctypes.byref(comparison)) != 0:
        raise RuntimeError("seccomp masked rule failure: " + name)


def _install_seccomp(profile):
    sec = _load_seccomp()
    ctx = sec.seccomp_init(_SCMP_ACT_ALLOW)
    if not ctx:
        raise RuntimeError("seccomp unavailable")

    blocked = [n for n in _BASE_DENY
               if profile != "official" or n not in _OFFICIAL_UNBLOCK]
    for name in blocked:
        _rule(sec, ctx, _SCMP_ACT_ERRNO, name)

    if profile == "official":
        # `clone` 放行了（torch 的 OpenMP 线程池要它），但**不许带任何
        # CLONE_NEW\*** —— 否则等于把 unshare 的活绕过去了。
        # clone 的 flags 在 arg0，加一条带掩码比较的规则即可。
        for bit in sorted(_CLONE_NEW_FLAGS.values()):
            _rule_masked(sec, ctx, _SCMP_ACT_ERRNO, "clone", 0, bit, bit)
        # clone3 的 flags 在**指针指向的结构体**里，seccomp 比较不到。
        #
        # ⚠️ 这里**必须回 ENOSYS 而不是 EPERM**。glibc 只在
        #    `errno == ENOSYS` 时才退回 clone；回 EPERM 它会当成
        #    「权限不足」直接失败（`ENOSYS` 对**所有** clone3 都生效，
        #    覆盖面比 EPERM 更大，安全性只增不减）。
        #
        #    退回的 clone 已经被上面那条掩码规则盯着（不许带 CLONE_NEW*）。
        _rule(sec, ctx, _SCMP_ACT_ENOSYS, "clone3")

    if sec.seccomp_load(ctx) != 0:
        raise RuntimeError("seccomp load failure")


# ───────────────────────────────────────────────── 挂载


class _Spec:
    """`TRACK2_SANDBOX_SPEC` 指向的 JSON 的**只读壳**。

    ⚠️ 刻意不 import `track2_judge.sandbox.official_runtime` ——
    bootstrap 是以 `/usr/bin/python3 -I -B bootstrap.py` 起的，
    `-I` 会把脚本目录和 PYTHONPATH 都从 `sys.path` 拿掉，
    import 不到包。而且这个脚本必须是**纯标准库**，才能在裸沙箱里跑。
    字段名与那边保持一致，改一处要两边一起改（有条自检盯着）。
    """

    def __init__(self, raw):
        self.profile = str(raw.get("profile", "stdlib"))
        self.binds = [
            (pathlib.Path(b["source"]), b["target"], b.get("kind", "dir"))
            for b in raw.get("binds", ())
        ]
        self.symlinks = [(s["path"], s["target"]) for s in raw.get("symlinks", ())]
        self.devices = tuple(raw.get("devices", ()))
        self.allow_network = bool(raw.get("allow_network", False))
        self.gateway_port = int(raw.get("gateway_port", 0))
        self.memory_limit_bytes = raw.get("memory_limit_bytes")
        self.cpu_limit_s = int(raw.get("cpu_limit_s", 30))
        self.mount_shm_bytes = int(raw.get("mount_shm_bytes", 0))
        self.mount_proc = bool(raw.get("mount_proc", False))


def _load_spec(path):
    try:
        with open(path, encoding="utf8") as handle:
            return _Spec(json.load(handle))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"TRACK2_SANDBOX_SPEC unreadable: {exc}")


def _bind_entry(entry):
    """`_Spec.binds` 的一条 → `(source, target, kind)`。"""
    return entry[0], entry[1], entry[2]


def _bind(source, target, kind, *, read_only=True):
    """把 `source` bind 到 `target`。`target` 不存在时按 `kind` 先建出来。"""
    target = pathlib.Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        if kind == "file":
            target.touch()
        else:
            target.mkdir(parents=True, exist_ok=True)
    command("mount", "--bind", str(source), str(target))
    if read_only:
        command("mount", "-o", "remount,bind,ro,nosuid,nodev", str(target))


def _mount_spec(root, spec):
    """按 spec 布置官方档的只读挂载。"""
    for entry in spec.binds:
        source, target, kind = _bind_entry(entry)
        _bind(source, pathlib.Path(str(root)) / target.lstrip("/"), kind)
    # 软链：bind 建不出软链，而沙箱根是空 tmpfs，只能在这里造。
    # `/lib -> usr/lib` 与 `/lib64 -> usr/lib64` 是 **ELF 解释器**能起来的前提。
    for path, target in spec.symlinks:
        link = pathlib.Path(str(root)) / path.lstrip("/")
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)


def _mount_devices(root, spec):
    """设备节点：只能 bind **已存在** 的节点文件。

    rootless 用户命名空间里没法 `mknod`（没有 CAP_MKNOD），`mount --dev-bind`
    也不行 —— 所以先 `touch` 出普通文件占位，再把真实节点 bind 上去。
    bind 保留原 inode，权限仍是设备节点。
    """
    dev = pathlib.Path(str(root)) / "dev"
    dev.mkdir(parents=True, exist_ok=True)
    for name in spec.devices:
        source = pathlib.Path("/dev") / name
        if not source.exists():
            continue
        target = dev / name
        target.touch()
        command("mount", "--bind", str(source), str(target))


# ───────────────────────────────────────────────── 沙箱主体


def _drop_privileges():
    """丢光 capability + `no_new_privs`。两档共用。"""
    libc = ctypes.CDLL(None, use_errno=True)
    for cap in range(41):
        libc.prctl(24, cap, 0, 0, 0)

    class Header(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

    class Data(ctypes.Structure):
        _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32),
                    ("inheritable", ctypes.c_uint32)]

    header = Header(0x20080522, 0)
    data = (Data * 2)()
    if libc.capset(ctypes.byref(header), ctypes.byref(data)) != 0:
        raise RuntimeError("capset failed")
    if libc.prctl(38, 1, 0, 0, 0) != 0:
        raise RuntimeError("no_new_privs failed")



def _launch_argv(package, entry, skill):
    if package is None:
        return ["python3", "-I", "-B", "/skill/main.py"]
    return ["python3", "-I", "-B", "/host/main.py", "--entrypoint", "/skill/" + entry]


def _legacy_main(root, skill, package, entry):
    """**第 4 步之前的路径，逐字保留。** 21 项回归跑的就是这条。"""
    root = pathlib.Path(root).resolve()
    skill = pathlib.Path(skill).resolve()
    command("mount", "--make-rprivate", "/")
    command("mount", "-t", "tmpfs", "-o", "size=32m,nosuid,nodev", "tmpfs", str(root))
    runtime = pathlib.Path(os.environ["TRACK2_SANDBOX_RUNTIME"]).resolve(strict=True)
    for name in ("usr", "lib", "lib64", "bin"):
        source = runtime / name
        if not source.exists():
            continue
        dest = root / name
        dest.mkdir(exist_ok=True)
        # Private copied tree: never recursive-bind host directories or mounts.
        command("mount", "--bind", str(source), str(dest))
        command("mount", "-o", "remount,bind,ro,nosuid,nodev", str(dest))
    (root / "skill").mkdir()
    if package is None:
        (root / "skill/main.py").touch()
        command("mount", "--bind", str(skill), str(root / "skill/main.py"))
        command("mount", "-o", "remount,bind,ro,nosuid,nodev", str(root / "skill/main.py"))
    else:
        command("mount", "--bind", str(package), str(root / "skill"))
        command("mount", "-o", "remount,bind,ro,nosuid,nodev", str(root / "skill"))
        (root / "host").mkdir()
        (root / "host/main.py").touch()
        command("mount", "--bind", str(skill), str(root / "host/main.py"))
        command("mount", "-o", "remount,bind,ro,nosuid,nodev", str(root / "host/main.py"))
    (root / "tmp").mkdir()
    (root / "etc").mkdir()
    (root / "dev").mkdir()
    (root / "dev/null").touch()
    command("mount", "--bind", "/dev/null", str(root / "dev/null"))
    os.chroot(root)
    os.chdir("/tmp")
    _drop_privileges()
    _install_seccomp("stdlib")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 ** 2, 256 * 1024 ** 2))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 ** 2, 1024 ** 2))
    os.execve("/usr/bin/python3", _launch_argv(package, entry, skill),
              {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "TMPDIR": "/tmp",
               "PYTHONIOENCODING": "utf-8"})


def _nvidia_library_dirs(root):
    """官方 runtime 里 CUDA 运行时的 `.so` 目录（**沙箱内**路径）。

    cu126 那批 `nvidia-*-cu12` 包把 `libcublas.so` / `libcufile.so.0` 这类
    共享库放在 `site-packages/nvidia/<pkg>/lib/`，**不在** `/usr/lib`。
    torch 只 preload 其中一部分，`libcufile.so.0` 就得靠动态链接器找 ——
    所以要把这些目录塞进 `LD_LIBRARY_PATH`。

    ⚠️ **必须在 `os.chroot` 之前调用**，而且**不能用 `glob`** ——
    chroot 之后 `sys.path` 上的标准库是宿主那份（3.10），而沙箱里只挂了
    官方 runtime 的 3.12 stdlib，此时再 import 任何新模块都会
    `ModuleNotFoundError`（实测：`No module named 'glob'`）。
    所以这里只用已经加载好的 `os`，手写一层目录遍历。
    """
    root = str(root).rstrip("/")
    base = os.path.join(root, "usr", "lib")
    try:
        version_dirs = sorted(d for d in os.listdir(base) if d.startswith("python3."))
    except OSError:
        return []
    directories = []
    for version_dir in version_dirs:
        packages = os.path.join(base, version_dir, "site-packages", "nvidia")
        try:
            names = sorted(os.listdir(packages))
        except OSError:
            continue
        for name in names:
            library = os.path.join(packages, name, "lib")
            if os.path.isdir(library):
                directories.append(library[len(root):] or "/")
    return directories


def _official_main(root, spec, skill, package, entry):
    """Q18 官方 runtime 档。只读挂载官方运行时 + 可选 GPU，其余照旧隔离。"""
    root = pathlib.Path(root).resolve()
    skill = pathlib.Path(skill).resolve()
    command("mount", "--make-rprivate", "/")
    # 根 tmpfs 要容得下 `/tmp` 的临时文件（torch 解压内核模块、编译 triton
    # 缓存都会往这里写）。64 MiB 会中途 ENOSPC，给 2 GiB。
    command("mount", "-t", "tmpfs", "-o", "size=2g,nosuid,nodev", "tmpfs", str(root))

    _mount_spec(root, spec)

    # /dev/shm —— CUDA 与多进程 dataloader 都要。**必须单独挂**：
    # 沙箱根的 tmpfs 只有 64 MiB，在它上面开 shm 会让 torch 一加载就 ENOSPC。
    if spec.mount_shm_bytes > 0:
        shm = root / "dev/shm"
        shm.mkdir(parents=True, exist_ok=True)
        os.chmod(shm, 0o777)
        command("mount", "-t", "tmpfs",
                "-o", f"size={spec.mount_shm_bytes},mode=1777,nosuid,nodev",
                "tmpfs", str(shm))

    # procfs：CUDA 驱动**必须**能读 `/proc/driver/nvidia` 才肯初始化
    # （不挂 procfs 时 `cuInit` 返回 304 = CUDA_ERROR_OPERATING_SYSTEM，
    # 表现为 `torch.cuda.is_available() == False` —— 看起来像「没插显卡」）。
    #
    # ⚠️ 挂的是**沙箱自己 PID 命名空间里的 procfs**（`unshare --pid` 已经保证），
    #    所以 `/proc/*` 只列出沙箱内的进程，看不到宿主与裁判侧的进程。
    #    `hidepid=2` 再加一道「非本用户进程不可见」。
    if getattr(spec, "mount_proc", False):
        proc = root / "proc"
        proc.mkdir(exist_ok=True)
        command("mount", "-t", "proc", "-o", "nosuid,nodev,noexec,hidepid=2",
                "proc", str(proc))

    (root / "tmp").mkdir(exist_ok=True)
    os.chmod(root / "tmp", 0o1777)
    # 设备节点：`null`/`zero`/`random`/`urandom` + GPU 那几个。
    # ⚠️ 顺序要紧 —— `/dev/shm` 是上面单独挂的 tmpfs，`_mount_devices`
    #    会 `mkdir(exist_ok=True)`，不会把它盖掉；但如果反过来先挂设备再挂 shm，
    #    tmpfs 会把 `/dev` 里已挂好的节点遮住。
    _mount_devices(root, spec)

    (root / "skill").mkdir(exist_ok=True)
    # 选手代码：只读挂载。**不挂裁判目录** —— 与 stdlib 档一致。
    command("mount", "--bind", str(package if package is not None else skill.parent),
            str(root / "skill"))
    command("mount", "-o", "remount,bind,ro,nosuid,nodev", str(root / "skill"))
    if package is not None:
        (root / "host").mkdir(exist_ok=True)
        (root / "host/main.py").touch()
        command("mount", "--bind", str(skill), str(root / "host/main.py"))
        command("mount", "-o", "remount,bind,ro,nosuid,nodev", str(root / "host/main.py"))

    # ⚠️ **在 chroot 之前**算 LD_LIBRARY_PATH —— chroot 之后能 import 的
    #    模块只剩已经加载过的那几个，连 `glob` 都会 ModuleNotFoundError。
    nvidia_dirs = _nvidia_library_dirs(root)

    # ⚠️ **在 chroot 之前**算 LD_LIBRARY_PATH —— chroot 之后能 import 的
    #    模块只剩已经加载过的那几个，连 `glob` 都会 ModuleNotFoundError。
    nvidia_dirs = _nvidia_library_dirs(root)

    os.chroot(root)
    os.chdir("/tmp")
    _drop_privileges()
    _install_seccomp("official")

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # ⚠️ 不设 RLIMIT_AS —— CUDA 的虚拟地址预留会让任何合理值都误判成 OOM
    #    （见模块头与 `Track2_完整方案.md` §4.7.1 的勘误）
    resource.setrlimit(resource.RLIMIT_CPU, (spec.cpu_limit_s, spec.cpu_limit_s))
    # torch 会开很多 fd；stdlib 档的 32 在这里必然不够
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 ** 3, 1024 ** 3))

    environ = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    # CUDA 运行时的 `.so` 分散在 `site-packages/nvidia/<pkg>/lib/` ——
    # 不把它们加进 LD_LIBRARY_PATH，torch 一 import 就
    # `ImportError: libcufile.so.0: cannot open shared object file`。
    # （`nvidia_dirs` 已在 chroot 之前算好，路径是沙箱内的 `/` 起头。）
    if nvidia_dirs:
        environ["LD_LIBRARY_PATH"] = ":".join(
            ["/usr/lib/x86_64-linux-gnu", "/usr/lib", *nvidia_dirs])
    if spec.devices:
        # 只 bind 了指定 GPU 的设备节点 —— 显式对齐 CUDA 的可见性判断
        gpus = [d[len("nvidia"):] for d in spec.devices
                if d.startswith("nvidia") and d[6:].isdigit() and d != "nvidiactl"]
        if gpus:
            environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    if spec.gateway_port:
        environ["TRACK2_TOOL_GATEWAY_URL"] = f"http://127.0.0.1:{spec.gateway_port}"

    os.execve("/usr/bin/python3", _launch_argv(package, entry, skill), environ)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # ⚠️ 不设 RLIMIT_AS —— CUDA 的虚拟地址预留会让任何合理值都误判成 OOM
    resource.setrlimit(resource.RLIMIT_CPU, (spec.cpu_limit_s, spec.cpu_limit_s))
    # torch 会开很多 fd；stdlib 档的 32 在这里必然不够
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 ** 3, 1024 ** 3))

    environ = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    # CUDA 运行时的 `.so` 分散在 `site-packages/nvidia/<pkg>/lib/` ——
    # 不把它们加进 LD_LIBRARY_PATH，torch 一 import 就
    # `ImportError: libcufile.so.0: cannot open shared object file`。
    # （`nvidia_dirs` 已经在 chroot 之前算好，见上。）
    if nvidia_dirs:
        environ["LD_LIBRARY_PATH"] = ":".join(
            ["/usr/lib/x86_64-linux-gnu", "/usr/lib", *nvidia_dirs])
    if spec.devices:
        # 只 bind 了 GPU0 的设备节点 —— 显式对齐 CUDA 的可见性判断
        gpus = [d[len("nvidia"):] for d in spec.devices
                if d.startswith("nvidia") and d[6:].isdigit() and d != "nvidiactl"]
        if gpus:
            environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    if spec.gateway_port:
        environ["TRACK2_TOOL_GATEWAY_URL"] = f"http://127.0.0.1:{spec.gateway_port}"

    if os.environ.get("TRACK2_SANDBOX_STOP_BEFORE_EXEC"):
        raise SystemExit(0)
    os.execve("/usr/bin/python3", _launch_argv(package, entry, skill), environ)


def main():
    root = pathlib.Path(sys.argv[1]).resolve()
    skill = pathlib.Path(sys.argv[2]).resolve()
    package = pathlib.Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else None
    entry = sys.argv[4] if package is not None else None

    spec_path = os.environ.get("TRACK2_SANDBOX_SPEC", "").strip()
    if spec_path:
        spec = _load_spec(spec_path)
        if spec.profile == "official":
            _official_main(root, spec, skill, package, entry)
            return
    _legacy_main(root, skill, package, entry)


if __name__ == "__main__":
    main()
