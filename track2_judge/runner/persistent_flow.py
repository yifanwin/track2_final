"""**常驻 flow 的 Judge 侧驱动**（C2 / D1）—— 循环归 Judge，flow 只起一次。

## 它取代了什么

`select_case.run()` 原来在**裁判进程内**造 env、装 provider、跑 skill
（`CandidateRunner`）。那是「第二套执行语义」：判官自己当了一次 runtime。
边界决策 §2 要的是：

```text
Judge owns loop
      │  POST /episode/reset          ← B1 私有通道
      ▼
┌─────────────────────────────┐
│  Persistent PAOS Flow       │  起**一次**，之后不再重启
│  runtime → gateway → player │
└──────────────┬──────────────┘
               │ 选手跑完 → player 节点报 END
               ▼
             Judge
               │ GET /episode/snapshot  ← B2 judge-only 真值
               │ reset(next seed)
               └──────────────► 同一个 flow
```

## 为什么值得

实测起停一次 flow **32.8 s**。全量是 `73 case × 5 seed × 2 mode = 730` 个
episode，`final` 还要追加执行 —— 每 episode 起一次 flow 光启动开销就 ≈ 6.1 h。
常驻之后启动开销是**一次**。

## 它会碰的三样东西，都不许泄漏给选手

| 通道 | 方向 | 谁看得见 |
|---|---|---|
| `POST /episode/reset` | Judge → runtime | **只有 Judge**（loopback + 可选 token） |
| `GET /episode/snapshot` | Judge ← runtime | **只有 Judge**（选手沙箱无网络，见 B5） |
| `[track2-player] 结束：{...}` | 从**日志**里读 | 这是裁判自己的日志，选手看不到 |

选手侧只有一条路：gateway 的三个 Tool（`observe` / `joint_trajectory` /
`gripper_control`）。
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

__all__ = ["PersistentFlow", "FlowError", "free_port"]

#: player 节点每跑完一个 episode 打的那一行（`node.py::_report_summary`）。
class FlowError(RuntimeError):
    """常驻 flow 起不来 / 中途挂了 —— **明确失败，不降级**。"""


def free_port() -> int:
    """要一个空闲端口。**不写死** —— 并发评测时写死必然撞。"""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _set_yaml_scalar(text: str, key: str, value: str, *,
                     add_if_missing: bool = False) -> str:
    """把顶层标量 `key: ...` 替换成 `key: value`（逐次渲染 `runtime.yaml` 用）。

    只认**行首无缩进**的那一行 —— 模板里的键都在顶层，缩进的同名字段
    （如果有）不该被碰。`add_if_missing` 时不慎重复追加也会被 `re.M` 的首个匹配挡住。
    """
    import re
    pattern = rf"^{re.escape(key)}:[ \t]*.*$"
    if re.search(pattern, text, flags=re.M):
        return re.sub(pattern, f"{key}: {value}", text, count=1, flags=re.M)
    if not add_if_missing:
        raise FlowError(
            f"`runtime.example.yaml` 里没有 `{key}:` —— 模板与 PersistentFlow 不同步")
    if not text.endswith("\n"):
        text += "\n"
    return text + f"{key}: {value}\n"


class PersistentFlow:
    """一个常驻 dora flow 的生命周期 + 两个私有端点。"""

    def __init__(self, *, bundle_root, physics_root, python, log_path,
                 gateway_port=None, control_port=None, runtime_gpu="0",
                 work_dir=None, env=None, token=None, name=None,
                 player_package=None, camera=True, oracle_planning=False):
        self.bundle_root = Path(bundle_root).resolve()
        self.physics_root = Path(physics_root).resolve()
        self.python = str(python)
        self.log_path = Path(log_path)
        # ★ 2026-10-19 并行化：端口默认**从本 worker 的 `gateway.yaml` 读**
        #   （单一真源）。以前这里写死 `19110`，而 `resident_runner` 又不传 ——
        #   于是 8 路并行时：worker bundle 的 `gateway.yaml` 已改成 19120，
        #   但**客户端仍连 19110**，gateway 永远等不到。读 profile 就自动跟随。
        self.gateway_port = int(gateway_port) if gateway_port is not None \
            else self._read_gateway_port()
        self.control_port = int(control_port or free_port())
        self.runtime_gpu = str(runtime_gpu)
        #: 要不要建相机（G3）。**渲染进 `runtime.yaml` 的 `camera_required`** ——
        #: 不能走环境变量（`dora start --detach` 下节点由长命的 dora daemon spawn，
        #: daemon 环境里 0 个 `TRACK2*`，会**静默地**不生效）。见 `start()`。
        self.camera = bool(camera)
        #: 要不要在 runtime 侧做 GT 规划（G2，裁判私有通道）。
        self.oracle_planning = bool(oracle_planning)
        self.work_dir = Path(work_dir) if work_dir else self.log_path.parent
        self.token = token
        self.name = name or f"track2-resident-{os.getpid()}-{self.control_port}"
        #: 选手包 ZIP。**注入是 start.sh 的活**（`TRACK2_PLAYER_PACKAGE`），
        #: 不是 dora 的 —— 启动前 `_inject_player()` 会跑一遍 `start.sh`。
        self.player_package = Path(player_package).resolve() if player_package else None
        self._env = dict(env or {})
        self._proc: subprocess.Popen | None = None
        self._log_handle = None
        self._offset = 0
        self._stopped = False
        self._cli_rc: int | None = None

    # ── 起 / 停

    def _read_gateway_port(self) -> int:
        """从本 worker 的 `profiles/sim/gateway.yaml` 读 `port:`。

        ⚠️ **不要退回写死值** —— 那是并行化前的 bug 源：端口在 yaml 里，
        客户端却用常量。读不到就**明确报错**（比连错端口好查）。
        """
        import re
        path = self.bundle_root / "profiles" / "sim" / "gateway.yaml"
        try:
            text = path.read_text(encoding="utf8")
        except OSError as exc:
            raise RuntimeError(f"读不到 gateway.yaml: {path} ({exc})") from exc
        m = re.search(r"^port:\s*(\d+)\s*$", text, flags=re.MULTILINE)
        if not m:
            raise RuntimeError(f"gateway.yaml 里没有 `port:` 行: {path}")
        return int(m.group(1))

    def _flow_yaml(self) -> Path:
        """生成 flow YAML（把两个 `${...}` 替掉）。

        ⚠️ dora 的 `path:` **只做** `${FORGE_RUNTIME_BIN}` / `${PAOS_SKILL_ROOT}`
        两种字符串替换（PAOS installer 里写死的那几个），**不展开任意环境变量** ——
        所以 `TRACK2_PYTHON` 这类必须由包装脚本（`run_runtime.sh`）自己读。
        """
        profile = self.bundle_root / "profiles" / "sim"
        source = (profile / "dataflow.yaml").read_text(encoding="utf8")
        gateway_bin = self._gateway_bin()
        rendered = (source
                    .replace("${PAOS_SKILL_ROOT}", str(self.bundle_root))
                    .replace("${FORGE_RUNTIME_BIN}", str(gateway_bin.parent)))
        # The detached Dora daemon does not inherit this launcher's environment.
        # Bind interpreter/physics paths on the nodes themselves, as PAOS's
        # SkillEnvironmentBuilder does. This also supports installed bundles,
        # whose parent directory does not contain toolenv or track2_final.
        import yaml
        flow = yaml.safe_load(rendered)
        for node in flow["nodes"]:
            if node.get("id") in ("runtime", "player"):
                node["env"] = {**(node.get("env") or {}),
                               "TRACK2_PYTHON": self.python,
                               "TRACK2_PHYSICS_ROOT": str(self.physics_root)}
        rendered = yaml.safe_dump(flow, sort_keys=False, allow_unicode=True)
        target = profile / ".resident_flow.yaml"
        target.write_text(rendered, encoding="utf8")
        return target

    def _gateway_bin(self) -> Path:
        """官方 gateway 二进制的位置。

        ⚠️ **不要靠相对位置猜**（第一版就是这么写的，结果 `work_dir.parent`
        算出来的是 `wenyifan/` 而真实路径在 `wenyifan/work/paos_home/...`）。
        从 `TRACK2_PAOS_HOME` 起找，找不到再沿三层候选目录扫一遍。

        正常路径由 `paos_home` 下 forge_runtime 的版本目录给出；全都找不到就
        明确报错 —— 猜错的表现是 flow 起不来但看不出为什么。
        """
        roots = []
        if os.environ.get("TRACK2_PAOS_HOME"):
            roots.append(Path(os.environ["TRACK2_PAOS_HOME"]))
        if os.environ.get("HOME"):
            roots.append(Path(os.environ["HOME"]) / ".PhyAgentOS")
        for base in (self.work_dir, self.work_dir.parent, self.bundle_root):
            roots += [base / "paos_home/.PhyAgentOS", base / ".PhyAgentOS"]
        for root in roots:
            versions = root / "forge_runtime/nodes/gateway/versions"
            candidates = sorted(versions.glob("gateway-*/gateway")) if versions.is_dir() else []
            if candidates:
                return candidates[-1]
        raise FlowError(
            "找不到官方 gateway 二进制。找过：" + "; ".join(str(r) for r in roots))

    def start(self, *, timeout_s: float = 420.0) -> "PersistentFlow":
        import shutil
        if shutil.which("dora") is None:
            raise FlowError("dora 不在 PATH 里（评测机需要 ~/.dora/bin）")
        self._inject_player()
        profile = self.bundle_root / "profiles" / "sim"
        example = profile / "runtime.example.yaml"
        if example.is_file():
            # ★ **node 侧的两项开关都渲染进 `runtime.yaml`**，不靠环境变量。
            #
            #   两件事都要走文件，理由是同一条：
            #   `dora start --detach` 之后节点由**长命的 dora daemon** spawn，
            #   而 daemon 环境里没有评测时设的变量（实测 `/proc/<daemon>/environ`
            #   里 0 个 `TRACK2*`）—— 走环境变量会**静默地**不生效：
            #     * `judge_control_port` 不开控制口 → gateway 起来了但 /health 连不上
            #     * `camera_required` 不生效 → 假开关（这正是 G3 的形态，2026-10-17 修）
            #
            #   `runtime.yaml` 由本方法逐次渲染，所以一定到得了节点。
            import re
            rendered = example.read_text(encoding="utf8")
            rendered = _set_yaml_scalar(
                rendered, "judge_control_port", str(self.control_port),
                add_if_missing=True)
            rendered = _set_yaml_scalar(
                rendered, "camera_required",
                "true" if self.camera else "false")
            # ★ 2026-10-19 并行化（八卡）：渲染本 worker 的**渲染 GPU 卡号**。
            #
            #   为什么走文件而不是环境变量：与上面两条同因 —— `--detach` 下
            #   节点由长命 daemon spawn，daemon 环境里 **0 个 `TRACK2*`**
            #   （实测 `/proc/<daemon>/environ`）。`TRACK2_RENDER_GPU` 虽然被
            #   塞进了 `dora start` 的 env，但**到不了节点**（实测节点
            #   environ 里没有它）。渲染进 `runtime.yaml` 才一定到得了。
            #
            #   节点侧用它设 `CUDA_VISIBLE_DEVICES`，让写死的
            #   `render_backend='sapien_cuda:0'` 落在**本 worker 的卡**上。
            #   串行时 `runtime_gpu` 默认 "0"，行为与旧版一致。
            rendered = _set_yaml_scalar(
                rendered, "render_gpu", str(self.runtime_gpu),
                add_if_missing=True)
            if self.oracle_planning:
                # G2：只有显式开启时才让 runtime 具备 GT 规划能力。
                rendered = _set_yaml_scalar(
                    rendered, "oracle_planning", "true", add_if_missing=True)
            (profile / "runtime.yaml").write_text(rendered, encoding="utf8")

        # ★ 控制口端口的**第二份落点**：`player` 节点也要知道它。
        #
        #   两端拿端口的路径不能都是环境变量（`--detach` 下 daemon 环境里
        #   什么都没有，见 `_control_url` 的说明）。所以走文件：
        #   runtime 从 `runtime.yaml` 读，player 从本文件读；收工由 `stop()` 删。
        #   它在 Bundle 的 `profiles/<p>/` 下，而选手沙箱只挂 `player/` —— 选手看不到。
        (profile / ".judge_control.json").write_text(
            json.dumps({"port": self.control_port, "flow": self.name}),
            encoding="utf8")
        # 结果文件**起 flow 之前清空** —— 否则会把上一轮的行当成本轮结果
        # （`drain_results` 是增量读，但它从 0 开始）。
        try:
            self.results_path.unlink()
        except OSError:
            pass

        flow = self._flow_yaml()
        env = {
            **os.environ,
            **self._env,
            "PAOS_SKILL_ROOT": str(self.bundle_root),
            "PAOS_SKILL_PROFILE": "sim",
            "TRACK2_PYTHON": self.python,
            "TRACK2_PHYSICS_ROOT": str(self.physics_root),
            "TRACK2_RENDER_GPU": self.runtime_gpu,
            # ★ B1：**给了端口就开**常驻（player 节点据此从「跑一次」变成「等 episode」）
            "TRACK2_JUDGE_CONTROL_PORT": str(self.control_port),
        }
        if self.token:
            env["TRACK2_JUDGE_CONTROL_TOKEN"] = self.token
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("ab", buffering=0)
        # `dora start --name ... --detach`：**必须带名字** —— `stop()` 靠它把
        # STOP 事件发给正在跑的 flow（只有那条路能让节点走到收尾）。
        #
        # ⚠️ `--detach` 下 CLI **立刻退出**（rc=0），flow 由 daemon 接着跑。
        #    所以「`self._proc` 还在不在」**不能**用来判断 flow 活没活 ——
        #    第一版就是这么写的，于是每次都在这里报「flow 进程提前退出」。
        #    活没活只能问 gateway（见下）。
        self._proc = subprocess.Popen(
            ["dora", "start", str(flow), "--name", self.name, "--detach"],
            cwd=str(profile), env=env,
            stdout=self._log_handle, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            self._proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            pass                        # 没 detach 成功就照常等（下面的轮询兜底）
        self._cli_rc = self._proc.returncode
        if self._cli_rc not in (0, None):
            raise FlowError(
                f"`dora start` 退出码 {self._cli_rc}，日志：{self.log_path}")

        deadline = time.monotonic() + timeout_s
        last = "no attempt"
        while time.monotonic() < deadline:
            try:
                body = self._get(self.gateway_url + "/tools", timeout=3.0)
                revision = (body.get("data") or {}).get("revision")
                if revision:
                    # 控制口也要能应答 —— 它由 runtime 节点起，比 gateway 稍晚。
                    try:
                        self.health()
                        return self
                    except Exception as exc:                    # noqa: BLE001
                        last = f"控制口未就绪（{type(exc).__name__}）"
                else:
                    last = f"revision={revision!r}"
            except Exception as exc:                            # noqa: BLE001
                last = f"{type(exc).__name__}: {str(exc)[:80]}"
            time.sleep(2.0)
        raise FlowError(f"gateway/控制口未就绪（{last}），日志：{self.log_path}")

    def stop(self, *, grace_s: float = 30.0) -> None:
        """停 flow，并**确认没有残留节点**（C4）。

        ⚠️ 两件事都要做，缺一不可：

        1. **给运行中的 flow 发 STOP**（`dora stop --name`）—— 这是唯一能让
           player / runtime 节点**自己走到收尾**的方式。只杀进程等于把它们
           半路砍掉：fd 不关、`unregister` 不发、gateway 里留着租约。
        2. **按节点模块名收残留** —— daemon 起的节点是它自己的孩子，
           detach 之后本地没有它们的 pid；只能按 `-m track2_*` 找。

        第一版只做了「杀本地进程组」，而 `--detach` 下那个 CLI **早就退出了**，
        于是每次 e2e 都漏 2 个节点（C4 明确要验这条）。
        """
        if self._stopped:
            return
        self._stopped = True
        self._dora_stop(grace_s=grace_s)
        self._kill_stragglers()
        # 端口标记要删 —— 留着的话下一轮若有人起**不带**控制口的 flow，
        # player 会照旧去连一个没人监听的端口，然后静默地一直空转。
        try:
            (self.bundle_root / "profiles" / "sim" / ".judge_control.json").unlink()
        except OSError:
            pass
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except OSError:
                pass
            self._log_handle = None

    def _inject_player(self) -> None:
        """跑一遍 Bundle 的 `start.sh` 注入选手包（`TRACK2_PLAYER_PACKAGE`）。

        **注入是 Bundle 自己的活**，不是 dora 的：`start.sh` 会解包 + 用
        `track2_judge.runner.player_manifest` 校验（与裁判侧**同一份判据**，
        不在这里重写一遍）。没给选手包就跳过 —— 自测形态照旧能跑。

        校验失败**必须中止**（`start.sh` 非零退出）：带着一个残缺的选手包
        跑完 10 个 episode，比直接失败糟得多。
        """
        if self.player_package is None:
            return
        script = self.bundle_root / "start.sh"
        if not script.is_file():
            raise FlowError(f"Bundle 里没有 start.sh：{self.bundle_root}")
        env = {
            **os.environ,
            "PAOS_SKILL_ROOT": str(self.bundle_root),
            "PAOS_SKILL_PROFILE": "sim",
            "TRACK2_PYTHON": self.python,
            "TRACK2_PHYSICS_ROOT": str(self.physics_root),
            "TRACK2_PLAYER_PACKAGE": str(self.player_package),
        }
        result = subprocess.run(["bash", str(script), self.bundle_root.name, "0.0.0"],
                                env=env, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise FlowError(
                "start.sh 注入选手包失败（rc=%d）：\n%s"
                % (result.returncode, (result.stdout + result.stderr)[-1500:]))

    def _dora_stop(self, *, grace_s: float) -> None:
        try:
            subprocess.run(["dora", "stop", "--name", self.name,
                            "--grace-duration", f"{max(5, int(grace_s / 2))}s"],
                           capture_output=True, timeout=grace_s + 30)
        except (OSError, subprocess.TimeoutExpired):
            # dora 不在 / 停不掉都不致命 —— 下面的 kill 兜底
            pass

    def _kill_stragglers(self, *, settle_s: float = 2.0) -> list[str]:
        """按**节点模块名**收残留（F18 的教训），并**只收本 worker 的**。

        `run_runtime.sh` / `run_player.sh` 都以 `exec python -u -m ...` 结尾，
        argv 早被 exec 换掉 —— 所以 `pkill -f run_runtime.sh` 是空操作。
        必须按 `-m track2_*` 匹配。

        ## ★ 2026-10-19 并行化修复：**必须按 worker 过滤**

        裸 `pgrep -f` 是**全局**匹配 —— 8 路并行时，任一 worker 调 `stop()`
        会把**其他 7 路**的 player/runtime 节点一起 SIGKILL。表现是「随机的
        case 莫名其妙失败」，极难定位。

        过滤依据用 **`TRACK2_PROFILE_DIR`**（每 worker 的 Bundle profile 目录
        不同 ⇒ 天然唯一）。它由 `_node_env()` 注入到每个节点进程，
        比控制口端口更稳（端口只在 `start.sh` 注入路径上存在，
        而 `TRACK2_PROFILE_DIR` 在所有节点进程的 `/proc/<pid>/environ` 里）。

        读不到 environ（进程已退 / 权限）时**跳过该 pid**，不误杀。
        """
        import signal
        patterns = ("python -u -m track2_player_node",
                    "python -u -m track2_maniskill_runtime")
        want = str(self.bundle_root / "profiles" / "sim")

        def _mine(pid_text: str) -> bool:
            try:
                blob = Path(f"/proc/{int(pid_text)}/environ").read_bytes()
            except (OSError, ValueError):
                return False        # 读不到 ⇒ 不是我能确定的，跳过（不误杀）
            for entry in blob.split(b"\0"):
                if entry.startswith(b"TRACK2_PROFILE_DIR="):
                    return entry.split(b"=", 1)[1].decode("utf-8", "replace") == want
            return False

        def _collect() -> list[int]:
            out: list[int] = []
            for pattern in patterns:
                try:
                    listing = subprocess.run(["pgrep", "-f", pattern],
                                             capture_output=True, text=True,
                                             timeout=10).stdout.split()
                except (OSError, subprocess.TimeoutExpired):
                    continue
                out.extend(int(t) for t in listing if t.isdigit() and _mine(t))
            return out

        found: list[str] = []
        for pid in _collect():
            try:
                os.kill(pid, signal.SIGTERM)
                found.append(f"{want}:{pid}")
            except (ProcessLookupError, PermissionError, ValueError):
                continue
        if found:
            time.sleep(settle_s)
            for pid in _collect():
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, ValueError):
                    continue
        return found

    # ── 私有端点（B1 / B2）

    @property
    def control_url(self) -> str:
        return f"http://127.0.0.1:{self.control_port}"

    @property
    def gateway_url(self) -> str:
        return f"http://127.0.0.1:{self.gateway_port}"

    def _headers(self) -> dict:
        return {"X-Track2-Judge-Token": self.token} if self.token else {}

    def _get(self, url: str, *, timeout: float = 10.0):
        request = urllib.request.Request(url, headers=self._headers())
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise FlowError(f"GET {url} → {exc.code}: {exc.read()[:200]!r}") from None

    def _post(self, path: str, payload: dict, *, timeout: float = 600.0):
        data = json.dumps(payload).encode("utf8")
        request = urllib.request.Request(
            self.control_url + path, data=data, method="POST",
            headers={"Content-Type": "application/json", **self._headers()})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise FlowError(
                f"POST {path} → {exc.code}: {exc.read()[:300]!r}") from None

    def health(self) -> dict:
        return self._get(self.control_url + "/health", timeout=5.0)

    def reset(self, case_id: str, seed: int) -> dict:
        """下发一个 episode。**只有 Judge 能调**（见模块说明）。

        ⚠️ 这个调用会**阻塞到场景重建完**（2.7 s）—— `endpoints.reset_episode`
        走的是单线程物理 executor，而它和 Tool 调用排同一条队列，
        所以 reset 不会与选手的 `joint_trajectory` 撞在一起（Q9）。
        """
        return self._post("/episode/reset",
                          {"case_id": case_id, "seed": seed})

    def snapshot(self) -> dict:
        """judge-only 真值（B2）。**不含**任何进 gateway 的字段。"""
        return self._get(self.control_url + "/episode/snapshot", timeout=120.0)

    def timings(self) -> dict:
        """Q14：本 flow 至今所有 Action 的执行耗时分布（judge-only）。"""
        return self._get(self.control_url + "/episode/timings", timeout=30.0)

    # ── 收 episode 结果（从日志）

    def drain_results(self) -> list[dict]:
        """读 `.episode_results.jsonl` 里**新的**行（有界增量读）。

        ⚠️ **不能靠日志文件**。正式形态是 `dora start --detach`：那种形态下
        节点 stdout 住在 **daemon** 里（要 `dora logs` 才取得到），
        Judge 自己那个日志文件里只有 spawn 阶段的几行 ——
        第一版就是读日志的，于是 `wait_episode()` 一直等不到东西
        （表现是「episode 失败：flow 退出」，而 flow 其实活得很好）。
        """
        path = self.results_path
        if path is None or not path.is_file():
            return []
        results = []
        with path.open("r", encoding="utf8", errors="replace") as handle:
            handle.seek(self._offset)
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(json.loads(line))
                except ValueError:
                    continue
            self._offset = handle.tell()
        return results

    def wait_episode(self, *, timeout_s: float = 1200.0,
                     poll_s: float = 1.0, dead_after_s: float = 120.0) -> dict:
        """等 player 报下一个 episode 的结果。

        ⚠️ **两种「等不到」要分开**：

        * 正常在跑 —— 选手一轮可能几百秒，继续等；
        * **flow 已经死了** —— `self._proc` 不可用（`--detach` 下 CLI 早就退出），
          所以要主动探活。判据：控制口连续 `dead_after_s` 秒不应答。
          没有这条的话，player 节点崩掉之后 Judge 会**干等到超时**
          （实测：一次 player 的 `AttributeError` 让 e2e 挂住 8 分钟）。
        """
        deadline = time.monotonic() + timeout_s
        last_alive = time.monotonic()
        while time.monotonic() < deadline:
            pending = self.drain_results()
            if pending:
                return pending[-1]
            try:
                self.health()
                last_alive = time.monotonic()
            except Exception:                                   # noqa: BLE001
                if time.monotonic() - last_alive > dead_after_s:
                    raise FlowError(
                        f"控制口已连续 {dead_after_s:.0f} s 不应答 —— flow 很可能挂了。"
                        f"日志：{self.log_path}")
            time.sleep(poll_s)
        raise FlowError(f"等 episode 结果超时（{timeout_s} s）：{self.results_path}")

    def attempt_episode(self, case_id: str, seed: int, *,
                        max_attempts: int = 1,
                        verdict=None,
                        timeout_s: float = 1200.0) -> dict:
        """下发一个 episode，按 `mode` 跑 1 次或最多 `max_attempts` 次。

        `verdict(attempt, snapshot) -> bool` 是**裁判的判定回调**：
        返回 True = 这一轮成功，**提前收工**。
        ⚠️ 取消 first/final 后**不再有**「失败后不 reset 追加执行」——
        `max_attempts` 恒为 1。判定的结果由 `resident_runner` 决定要不要
        **重测整轮**（那是 `reset` 之后重来，不是在同一状态上追加）。

        ⚠️ **判定必须由裁判做，不能看选手自己报的 `status`。**
        `snapshot["player"]["result"]["status"]` 只说明「选手进程正常退出了」，
        选手永远可以返回 `{"status": "done"}`。
        真正的判定是拿 `snapshot` 跑 `FrozenVerifier` / `TrustedContractExtension`
        —— 这正是「循环归 Judge」的字面意思：**中间那次 verify 在裁判侧**。

        返回值里 `attempts` 是**实际执行了几次**（单模式下恒为 1）。
        """
        self.drain_results()
        record = self.reset(case_id, seed)
        rounds = []
        # ★ 2026-10-18 取消 first/final：**不再有**「失败后追加执行」。
        #   `max_attempts` 恒为 1（保留参数只为调用方签名兼容）。
        for attempt in range(1, max_attempts + 1):
            played = self.wait_episode(timeout_s=timeout_s)
            snapshot = self.snapshot()
            snapshot = {**snapshot, "player": played.get("result"),
                        "player_seq": played.get("seq"),
                        "player_attempt": played.get("attempt"),
                        "player_summary": played}
            rounds.append({"attempt": attempt, "played": played,
                           "snapshot": snapshot})
            if verdict is not None:
                try:
                    done = bool(verdict(attempt, snapshot))
                except Exception as exc:                        # noqa: BLE001
                    # 判定本身出错 = 这次评测不可信 → **明确失败**，不吞掉
                    raise FlowError(
                        f"裁判判定回调抛异常（attempt {attempt}）："
                        f"{type(exc).__name__}: {str(exc)[:200]}") from exc
                if done:
                    break
        return {"episode": record, "rounds": rounds,
                "attempts": len(rounds),
                "plan_revisions": len(rounds) - 1,
                # 便利键：**最后一轮**的选手返回值（判分只看最后一轮的终态）
                "player": (rounds[-1]["played"].get("result") or {}),
                "player_summary": rounds[-1]["played"],
                "snapshot": rounds[-1]["snapshot"]}

    def run_episode(self, case_id: str, seed: int, *,
                    timeout_s: float = 1200.0) -> dict:
        """下发一个 episode 并**等它真的跑完**，返回 judge 侧需要的三样。

        ⚠️ **必须先等上一轮结束再下发下一轮**。player 节点是用 `(seq, attempt)`
        判断「有新的一轮了」的：如果连着下发 seq=1、seq=2，而 player 还在
        上一轮里，它下一轮只会看到 seq=2 —— 中间那一轮就被**静默跳过**了。
        这个方法把「下发 → 等结果 → 取快照」串起来，就是防这个。

        只跑 1 次 `run()`（= 边界决策里的 `first`）。要跑 `final` 的多轮
        用 `attempt_episode()`。
        """
        return self.attempt_episode(case_id, seed, max_attempts=1,
                                    timeout_s=timeout_s)

    @property
    def results_path(self) -> Path | None:
        return self.bundle_root / "profiles" / "sim" / ".episode_results.jsonl"
