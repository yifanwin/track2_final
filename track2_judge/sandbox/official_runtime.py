"""官方 ML runtime 沙箱档（第 4 步 / Q18）。

## 它解决什么

第 4 步之前，沙箱只挂一份**只拷贝标准库**的运行时
（`sandbox/runtime.py`，排除 `site-packages`/`dist-packages`）。这个前提与
「选手可以自带模型在沙箱内推理」（Q6）**直接冲突** —— torch 不在标准库里。

Q18 把这个问题的形状收窄了：**依赖由官方镜像提供，选手只带权重**。
所以这里不需要「支持任意 Python 环境」，只需要**把官方 runtime 挂进沙箱**。

## 为什么是 bind 而不是 copy

官方 runtime 是 8.7 GiB（`toolenv/.venv/lib/python3.12/site-packages`，含 CUDA 版 torch）。
拷贝一遍既慢又占盘。而 bind-mount 的语义正好够用：

```text
宿主（只读、可白名单）                   沙箱内
<uv>/bin/python3.12              ─bind─▶ /usr/bin/python3
<uv>/lib/                        ─bind─▶ /lib
<venv>/lib/python3.12/site-packages ─bind─▶ /lib/python3.12/site-packages
/dev/nvidia0                     ─bind─▶ /dev/nvidia0        （只给 GPU 0）
```

**只挂这四项**，宿主其余部分（HOME、裁判目录、casebank、模型权重之外的一切）
仍然看不见。这比「递归 bind 宿主 /usr」安全得多，也是旧实现刻意避免的做法
（`sandbox/runtime.py` 的注释明说 "No host runtime directory mounts"）。

## 与 stdlib 档的关系

**默认仍是 stdlib 档**（`--net` 隔离 + seccomp 禁网 + 只拷标准库），
第 4 步之前的所有测试与回归都跑在那一档上，行为**一个字都没动**。
官方档是**新增**的一档，只在 `requires_gpu` 或显式要求时启用。

## 诚实说明：官方档的隔离强度

| 维度 | stdlib 档 | official 档 |
|---|---|---|
| user/mount/pid 命名空间 | ✅ | ✅（不变） |
| 网络 | `--net` + seccomp 禁 socket | **不隔离网络**，只放行 socket/connect（见下） |
| 宿主文件系统 | 只挂拷贝出来的 stdlib | 只挂上面那几项 |
| 内存上限 | `RLIMIT_AS` 256 MiB | **不设**（CUDA 的 VA 预留远超任何合理值，见下） |
| cap / no_new_privs | 全丢 | 全丢（不变） |

**网络：与 stdlib 档一样保留 `--net`（自己的网络命名空间）。**

实测（`work/scripts/check_sandbox_official.py`）：

| | 外网 | 宿主 `127.0.0.1:19110` |
|---|---|---|
| 有 `--net` | ✗ 不可达 | ✗ 不可达 |
| 无 `--net` | **✓ 可达** | ✓ 可达 |

`cuInit` 在**两种情况下都等于 0**（`check_sandbox_official.py` 直接验过），
所以没有理由为了 CUDA 放弃网络隔离。

**代价**：沙箱内连不到宿主上的 gateway，所以工具调用**不能**走 HTTP。
它走的仍是既有的 JSON-lines 桥：

```text
沙箱内 tools.joint_trajectory(targets)
  → 协议 stdout/stdin（package_host.py:ToolClient）
  → 评测器进程 toolapi/client.py:SandboxBridge
  → Gateway → 物理
```

`Track2_完整方案.md` §4.7.4 把「去掉网络隔离」列为**可选**，这里选择了**不去掉** ——
隔离强度更高，且不影响功能。

`connect` 那条 seccomp 规则**保留放行**（不是必需，但既然进了独立 netns，
留它没有额外暴露，而以防万一某些 CUDA 路径需要它）。

**内存这条的坑**（原方案 §4.7.1 已勘误）：`RLIMIT_AS = 32 GiB` 是**错的**。
CUDA 版 torch 一加载就 reserve 几十 GiB 虚拟地址空间（不是常驻内存），
设任何 `RLIMIT_AS` 都会得到「模型还没跑就 OOM」。所以官方档**不设地址空间上限**，
改用墙钟超时 + 父进程可杀（`RLIMIT_CPU` 给一个宽值兜底）。
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "RuntimeSpecError", "BindMount", "SandboxSpec", "stdlib_spec",
    "official_spec", "resolve_official_runtime", "OFFICIAL_RUNTIME_ENV",
    "GPU_DEVICE_ENV", "SPEC_ENV",
]

# 官方 runtime 的位置（由 `skill.yaml` / start.sh 注入，或这里探测）
OFFICIAL_RUNTIME_ENV = "TRACK2_SANDBOX_RUNTIME"
SPEC_ENV = "TRACK2_SANDBOX_SPEC"
# 要暴露给沙箱的 GPU（默认只 GPU 0）
GPU_DEVICE_ENV = "TRACK2_SANDBOX_GPU"

_DEFAULT_GPU = "0"
# CUDA 至少需要这三个；`/dev/nvidiaN` 按 GPU_DEVICE_ENV 追加
_CUDA_CONTROL_DEVICES = ("nvidiactl", "nvidia-uvm", "nvidia-uvm-tools")
# 基础设备节点。官方 runtime（torch/CUDA）会用到 `/dev/urandom`
# （`torch.rand` 的种子、OpenMP 的哈希种子），缺了直接
# `RuntimeError: Unable to open /dev/urandom`。stdlib 档只挂 `/dev/null` 够用，
# 官方档必须多挂这几个 —— 它们都是**只读字符设备**，没有授予任何额外权限。
_BASE_DEVICES = ("null", "zero", "random", "urandom")


class RuntimeSpecError(RuntimeError):
    """沙箱档不能被满足 —— **宁可起不来，也不要悄悄降级**。"""


@dataclass(frozen=True)
class BindMount:
    """一条 bind 规则。`target` 是沙箱内的**绝对路径**。"""

    source: Path
    target: str
    kind: str                      # "file" | "dir"

    def as_dict(self) -> dict[str, str]:
        return {"source": str(self.source), "target": self.target, "kind": self.kind}


@dataclass(frozen=True)
class SandboxSpec:
    """bootstrap 需要知道的全部沙箱参数。

    序列化成 JSON 通过 `TRACK2_SANDBOX_SPEC` 传给 bootstrap ——
    用文件而不是 argv，因为条目数不固定，且更容易在日志里看清。
    """

    profile: str                   # "stdlib" | "official"
    binds: tuple[BindMount, ...] = ()
    # 沙箱根是**新建的 tmpfs**，里面什么都没有。ELF 解释器与 `python3` 的
    # 硬编码路径要靠这些软链才解析得了 —— bind 建不出软链，只能由 bootstrap 建。
    symlinks: tuple[tuple[str, str], ...] = ()
    devices: tuple[str, ...] = ()
    # ⚠️ 保持 False：官方档**也保留 `--net`**（自己的网络命名空间）。
    #    实测（`work/scripts/check_sandbox_official.py` 的 C/G2 项）：
    #      * 有 `--net` → 外网不可达；宿主 gateway 也不可达
    #      * 无 `--net` → **外网可达**（宿主能到哪它就能到哪）
    #    CUDA 在**两种情况下都正常**（`cuInit=0`），所以这里没有理由放弃隔离。
    #    代价是沙箱内连不到宿主上的 gateway —— 工具调用仍走既有的
    #    JSON-lines 桥（`sandbox/package_host.py` 的 `ToolClient` +
    #    评测器侧的 `toolapi/client.py:SandboxBridge`），那条路本来就不需要网络。
    #    这也正是 `Track2_完整方案.md` §4.7.4 把「去掉网络隔离」列为**可选项**的原因。
    allow_network: bool = False
    gateway_port: int = 0
    # None = 不设地址空间上限（CUDA 必须如此）
    memory_limit_bytes: int | None = None
    cpu_limit_s: int = 30
    mount_shm_bytes: int = 0       # 0 = 不单独挂 /dev/shm
    # CUDA 驱动**要求**能读 `/proc/driver/nvidia` 才肯初始化。
    # 挂的是**沙箱自己 PID 命名空间里的 procfs**，看不到宿主进程；见 bootstrap 的说明。
    mount_proc: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "binds": [b.as_dict() for b in self.binds],
            "symlinks": [{"path": p, "target": t} for p, t in self.symlinks],
            "devices": list(self.devices),
            "allow_network": self.allow_network,
            "gateway_port": self.gateway_port,
            "memory_limit_bytes": self.memory_limit_bytes,
            "cpu_limit_s": self.cpu_limit_s,
            "mount_shm_bytes": self.mount_shm_bytes,
            "mount_proc": self.mount_proc,
        }

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf8")
        path.chmod(0o600)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SandboxSpec":
        raw = json.loads(Path(path).read_text(encoding="utf8"))
        return cls(
            profile=str(raw["profile"]),
            binds=tuple(BindMount(Path(b["source"]), b["target"], b["kind"])
                        for b in raw.get("binds", ())),
            symlinks=tuple((s["path"], s["target"]) for s in raw.get("symlinks", ())),
            devices=tuple(raw.get("devices", ())),
            allow_network=bool(raw.get("allow_network", False)),
            gateway_port=int(raw.get("gateway_port", 0)),
            memory_limit_bytes=raw.get("memory_limit_bytes"),
            cpu_limit_s=int(raw.get("cpu_limit_s", 30)),
            mount_shm_bytes=int(raw.get("mount_shm_bytes", 0)),
            mount_proc=bool(raw.get("mount_proc", False)),
        )


# ───────────────────────────────────────────────── 两档的构造


def stdlib_spec(runtime_root: str | Path, *, cpu_limit_s: int = 30) -> SandboxSpec:
    """默认档：**与第 4 步之前的行为完全一致**。

    单条 bind 把整个拷贝出来的 runtime 挂成 `/`。所有旧语义（`--net`、
    seccomp 禁 socket、`RLIMIT_AS` 256 MiB、`RLIMIT_CPU` 30 s）都在 bootstrap
    的 stdlib 分支里逐字保留。
    """
    root = Path(runtime_root).resolve()
    if not root.is_dir():
        raise RuntimeSpecError(f"stdlib runtime root missing: {root}")
    return SandboxSpec(profile="stdlib",
                       binds=(BindMount(root, "/", "dir"),),
                       cpu_limit_s=cpu_limit_s)


def _cuda_devices(gpu_index: str = _DEFAULT_GPU) -> tuple[str, ...]:
    """官方档要 bind 的设备节点。

    基础节点（null/zero/random/urandom）在前 —— 它们是通用需求，不是 GPU 需求；
    缺 `/dev/urandom` 会让 torch 直接起不来（实测）。
    """
    devices = list(_BASE_DEVICES)
    devices.extend(_CUDA_CONTROL_DEVICES)
    if gpu_index and gpu_index.lower() not in ("none", ""):
        for index in gpu_index.split(","):
            index = index.strip()
            if not index:
                continue
            if not index.isdigit():
                raise RuntimeSpecError(
                    f"{GPU_DEVICE_ENV} must be a comma-separated list of GPU indices, "
                    f"got {gpu_index!r}")
            devices.append(f"nvidia{index}")
    return tuple(devices)


def resolve_official_runtime(site_packages: str | Path | None = None) -> tuple[Path, Path]:
    """找官方 runtime 的**基础解释器安装目录**与 **site-packages**。

    这两者是**两个不同的位置**，这是最容易搞错的一点：

    ```text
    ~/.local/share/uv/python/cpython-3.12.14-.../   ← 基础安装（stdlib 在这里）
    ├── bin/python3.12
    └── lib/python3.12/            ← 标准库
       /libpython3.12.so

    wenyifan/toolenv/.venv/                          ← venv（只有第三方包）
    ├── pyvenv.cfg                 ← home= 指向上面的基础安装
    └── lib/python3.12/site-packages/                ← torch / mani_skill / ...
    ```

    沙箱里要复原的正是这个组合：`/usr/bin/python3` = 基础解释器，
    `/usr/lib/python3.12/` = 基础 stdlib，再让 venv 的 `site-packages`
    落在 `/usr/lib/python3.12/site-packages`。

    返回 `(base_install, site_packages)`。
    """
    explicit_site = (Path(site_packages).expanduser().resolve()
                     if site_packages else None)
    if explicit_site is not None and not explicit_site.is_dir():
        raise RuntimeSpecError(f"site-packages is not a directory: {explicit_site}")

    override = os.environ.get(OFFICIAL_RUNTIME_ENV, "").strip()
    if override:
        override_path = Path(override).expanduser().resolve()
        if not override_path.is_dir():
            raise RuntimeSpecError(
                f"{OFFICIAL_RUNTIME_ENV} is not a directory: {override_path}")
        site = explicit_site or override_path
        return _base_install_for(override_path), site

    if explicit_site is not None:
        return _base_install_for(explicit_site), explicit_site

    # 退路：当前解释器自己（评测机在标定环境里跑时走这条）
    venv_root = Path(sys.prefix)
    site = _site_packages_for(venv_root)
    return _base_install_for(venv_root), site


def _venv_root_of(path: Path) -> Path | None:
    """从任意内层目录往上找 venv 根（含 `pyvenv.cfg` 的那层）。"""
    for candidate in (path, *path.parents):
        if (candidate / "pyvenv.cfg").is_file():
            return candidate
    return None


def _base_install_for(path: Path) -> Path:
    """给定 venv（或任意内层目录），返回**基础解释器安装目录**。

    优先读 `pyvenv.cfg` 的 `home=` —— 那是 venv 记下的基础解释器所在，最可靠。
    读不到再退到「从路径里找含 `bin/python3.*` 的层」，最后看符号链接的最终指向。
    """
    path = path.resolve()

    venv_root = _venv_root_of(path)
    if venv_root is not None:
        home = _read_pyvenv_home(venv_root / "pyvenv.cfg")
        if home is not None:
            resolved = home.resolve()
            if (resolved / "lib").is_dir():
                return resolved
            # 有些布局把 venv 建在基础安装内部，再往上找一层
            for candidate in (resolved, *resolved.parents):
                if (candidate / "lib").is_dir():
                    return candidate
            return resolved
        # pyvenv.cfg 里没有 home：从 venv 的 bin/python 符号链接追
        try:
            return (venv_root / "bin" / "python").resolve().parent.parent
        except OSError:
            pass

    # 不是 venv：找含 bin/python3.* 的层
    for candidate in (path, *path.parents):
        bin_dir = candidate / "bin"
        if bin_dir.is_dir() and (any(bin_dir.glob("python3.*")) or (bin_dir / "python").exists()):
            return candidate
    raise RuntimeSpecError(f"cannot locate a Python install dir from {path}")


def _read_pyvenv_home(cfg: Path) -> Path | None:
    """读 `pyvenv.cfg` 的 `home = ...`。读不到返回 None。"""
    try:
        for line in cfg.read_text(encoding="utf8", errors="replace").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "home" and value.strip():
                candidate = Path(value.strip()).expanduser()
                return candidate if candidate.is_dir() else None
    except OSError:
        return None
    return None


def _site_packages_for(install: Path) -> Path:
    """在 venv（或安装目录）里找 site-packages。"""
    for pattern in ("lib/python3.*/site-packages", "lib64/python3.*/site-packages"):
        matches = sorted(install.glob(pattern))
        if matches:
            return matches[0]
    raise RuntimeSpecError(
        f"no site-packages under {install}; pass it explicitly "
        f"(from skill.yaml environment / TRACK2_SANDBOX_RUNTIME)")


def _stdlib_dir(base_install: Path) -> Path:
    """基础安装里的标准库目录（`lib/python3.X`）。"""
    matches = sorted(base_install.glob("lib/python3.*"))
    for candidate in matches:
        if (candidate / "os.py").is_file():
            return candidate
    raise RuntimeSpecError(
        f"no stdlib (lib/python3.*/os.py) under {base_install}")


def official_spec(
    *,
    python_install: str | Path | None = None,
    site_packages: str | Path | None = None,
    gpu: str = _DEFAULT_GPU,
    gateway_port: int = 0,
    cpu_limit_s: int = 3600,
    mount_shm_bytes: int = 1024 ** 3,
    system_libs: str | Path | None = None,
    loader: str | Path | None = None,
) -> SandboxSpec:
    """Q18 官方 runtime 档。

    挂载计划（沙箱内用**标准 Linux 布局**，这样 CPython 自己就能找到 stdlib）：

    ```text
    <base>/bin/python3.12      ─bind─▶ /usr/bin/python3        （文件）
    <base>/lib/python3.12/     ─bind─▶ /usr/lib/python3.12      （stdlib）
    <venv>/.../site-packages   ─bind─▶ /usr/lib/python3.12/site-packages
    <base>/lib/libpython3.12.so* ─bind─▶ /usr/lib/…
    /usr/lib/x86_64-linux-gnu  ─bind─▶ /usr/lib/x86_64-linux-gnu（libstdc++ 等）
    /dev/nvidia*               ─bind─▶ 同名                     （只给指定 GPU）
    ```

    **三处「不这么做就跑不起来」的细节**（都是实测出来的）：

    1. **用 `/usr/lib` 而不是 `/lib`** —— CPython 从 `/usr/bin/python3` 推出
       `prefix=/usr`，再去 `prefix/lib/python3.12` 找 `os.py`。放 `/lib` 它**找不到**。
    2. **软链要自己建**（`spec.symlinks`）—— 沙箱根是全新 tmpfs，里面空无一物。
       ELF 解释器 `/lib64/ld-linux-x86-64.so.2` 是**硬编码在可执行文件里**的，
       `lib → usr/lib`、`lib64 → usr/lib64` 这两条链接不建就 `No such file or directory`。
    3. **`/dev/nvidia*` 得 bind 进去** —— `--dev-bind` 在 rootless 用户命名空间里
       用不了（设备节点无法在 userns 里创建），只能靠 bind 一个**已存在**的节点文件。
       所以设备目标要在挂载前 `touch` 出来。
    """
    if python_install is not None:
        base = Path(python_install).expanduser().resolve()
        site = (Path(site_packages).expanduser().resolve() if site_packages
                else _site_packages_for(_venv_root_of(base) or base))
    else:
        base, site = resolve_official_runtime(site_packages)

    if not base.is_dir():
        raise RuntimeSpecError(f"official runtime python install missing: {base}")
    if not site.is_dir():
        raise RuntimeSpecError(f"official runtime site-packages missing: {site}")

    stdlib = _stdlib_dir(base)
    interpreter = _pick_interpreter(base)
    version_dir = stdlib.name                    # "python3.12"

    binds: list[BindMount] = [
        BindMount(interpreter, "/usr/bin/python3", "file"),
        BindMount(stdlib, f"/usr/lib/{version_dir}", "dir"),
        # site-packages 落在 stdlib 之内（Python 的标准查找位置）
        BindMount(site, f"/usr/lib/{version_dir}/site-packages", "dir"),
    ]
    if interpreter.name != "python3":
        binds.append(BindMount(interpreter, f"/usr/bin/{interpreter.name}", "file"))

    # libpython 本体（在 base/lib 顶层的 .so）+ tcl/tk 那类解释器自带的库
    for shared in sorted(base.glob("lib/*.so*")):
        binds.append(BindMount(shared, f"/usr/lib/{shared.name}", "file"))

    # 系统库（libstdc++ / libgcc_s / libm / libGL …）：torch 与 SAPIEN 都要。
    # 只挂**一个目录**，仍然是只读 —— 不递归挂宿主 /usr。
    libs_dir = _system_libs_dir(system_libs)
    if libs_dir is not None:
        binds.append(BindMount(libs_dir, "/usr/lib/x86_64-linux-gnu", "dir"))

    # ELF 解释器：从可执行文件里读 PT_INTERP，再把它挂到沙箱内同一路径。
    # 读不到就退回宿主的标准位置（绝大多数发行版都一样）。
    loader_path = Path(loader).expanduser().resolve() if loader else _elf_interpreter(interpreter)
    symlinks: list[tuple[str, str]] = [
        # `/lib` 与 `/lib64` 是指向 `usr/...` 的**顶层软链** —— 宿主就是这样，
        # 而 ELF 的 INTERP 是硬编码的绝对路径，少一条就 `No such file or directory`。
        ("/lib", "usr/lib"),
        ("/lib64", "usr/lib64"),
    ]
    if loader_path is not None and loader_path.is_file():
        binds.append(BindMount(loader_path, f"/usr/lib/x86_64-linux-gnu/{loader_path.name}",
                               "file"))
        # ⚠️ `/usr/lib64/ld-linux-x86-64.so.2` 指向**沙箱内**的
        #    `/lib/x86_64-linux-gnu/...`（绝对路径）。不能照搬宿主那份 ——
        #    宿主 `/lib64` 自己就是软链，照搬会得到「软链指向软链」的循环解析。
        #    也不能把 `/usr/lib64` 做成软链：它得是**真目录**才能有东西放进去。
        symlinks.append(("/usr/lib64/ld-linux-x86-64.so.2",
                         f"/lib/x86_64-linux-gnu/{loader_path.name}"))

    return SandboxSpec(
        profile="official",
        binds=tuple(binds),
        symlinks=tuple(symlinks),
        devices=_cuda_devices(gpu),
        allow_network=False,
        gateway_port=int(gateway_port),
        # 不设地址空间上限 —— CUDA 的 VA 预留会让任何合理值都误判成 OOM
        memory_limit_bytes=None,
        cpu_limit_s=cpu_limit_s,
        mount_shm_bytes=mount_shm_bytes,
        mount_proc=True,
    )


def _elf_interpreter(binary: Path) -> Path | None:
    """读 ELF 的 `PT_INTERP`（动态链接器路径）。读不到返回 None。

    用 `readelf` 有依赖，所以自己解析：ELF 头 64 位、小端时，
    `e_phoff` 在偏移 32（8 字节），程序头 56 字节，`p_type == 3` 是 PT_INTERP。
    """
    import struct

    try:
        data = binary.read_bytes()[:4096]
    except OSError:
        return None
    if len(data) < 64 or data[:4] != b"\x7fELF":
        return None
    if data[4] != 2:                              # 只要 64 位
        return None
    little = data[5] == 1
    endian = "<" if little else ">"
    try:
        (phoff,) = struct.unpack_from(endian + "Q", data, 32)
        (phentsize,) = struct.unpack_from(endian + "H", data, 54)
        (phnum,) = struct.unpack_from(endian + "H", data, 56)
    except struct.error:
        return None
    # 用完整文件读程序头（前 4 KiB 可能不够）
    try:
        blob = binary.read_bytes()
    except OSError:
        return None
    for index in range(phnum):
        offset = phoff + index * phentsize
        try:
            (p_type,) = struct.unpack_from(endian + "I", blob, offset)
        except struct.error:
            return None
        if p_type != 3:                           # PT_INTERP
            continue
        try:
            (p_offset,) = struct.unpack_from(endian + "Q", blob, offset + 8)
            (p_filesz,) = struct.unpack_from(endian + "Q", blob, offset + 32)
        except struct.error:
            return None
        raw = blob[p_offset:p_offset + p_filesz].split(b"\0", 1)[0]
        if not raw:
            return None
        candidate = Path(raw.decode("utf8", "replace"))
        return candidate if candidate.is_file() else None
    return None


def _pick_interpreter(base: Path) -> Path:
    """基础安装里的解释器可执行文件。优先带版本号的那个（是真实文件）。"""
    bin_dir = base / "bin"
    for pattern in ("python3.*", "python3", "python"):
        for candidate in sorted(bin_dir.glob(pattern)):
            if candidate.is_file() or candidate.is_symlink():
                return candidate
    raise RuntimeSpecError(f"no python interpreter under {bin_dir}")


def _system_libs_dir(override: str | Path | None) -> Path | None:
    """宿主 glibc/toolchain 共享库目录。找不到返回 None（不挂，让自检报缺）。"""
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_dir():
            raise RuntimeSpecError(f"system libs dir missing: {path}")
        return path
    for candidate in ("/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/lib/x86_64-linux-gnu"):
        path = Path(candidate)
        if path.is_dir():
            return path
    return None


# ───────────────────────────────────────────────── 环境自检


def check_official_runtime(spec: SandboxSpec) -> list[str]:
    """返回**缺失**项列表。空列表 = 这份 spec 能落地。

    在起沙箱**之前**调用 —— 缺东西要在启动时就报，不要在选手第一次
    `import torch` 时才炸（那种失败的现场是「选手代码有问题」的样子，会误导人）。
    """
    problems: list[str] = []
    for bind in spec.binds:
        if not bind.source.exists():
            problems.append(f"bind source missing: {bind.source} -> {bind.target}")
    if spec.devices and _DEFAULT_GPU is not None:
        for device in spec.devices:
            path = Path("/dev") / device
            if not path.exists():
                problems.append(f"device missing: {path}")
    if spec.allow_network and not shutil.which("unshare"):
        problems.append("unshare not on PATH (network namespace needs it)")
    return problems


def main(argv: list[str] | None = None) -> int:
    """CLI：`python -m track2_judge.sandbox.official_runtime [--write PATH]`。

    给运维/评测机用：打印官方档的 bind 计划并自检。
    """
    import argparse

    parser = argparse.ArgumentParser(description="official sandbox runtime spec")
    parser.add_argument("--site-packages", default=None)
    parser.add_argument("--python-install", default=None)
    parser.add_argument("--gpu", default=os.environ.get(GPU_DEVICE_ENV, _DEFAULT_GPU))
    parser.add_argument("--gateway-port", type=int, default=0)
    parser.add_argument("--write", default=None, help="把 spec 写到这个路径")
    args = parser.parse_args(argv)

    try:
        spec = official_spec(python_install=args.python_install,
                             site_packages=args.site_packages,
                             gpu=args.gpu, gateway_port=args.gateway_port)
    except RuntimeSpecError as exc:
        print(f"REJECTED: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(spec.as_dict(), indent=2))
    problems = check_official_runtime(spec)
    for problem in problems:
        print(f"MISSING: {problem}", file=sys.stderr)
    if args.write:
        spec.write(args.write)
        print(f"written: {args.write}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
