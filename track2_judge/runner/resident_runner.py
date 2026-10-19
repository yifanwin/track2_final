"""**常驻链路的裁判侧运行器**（C2）—— 只编排、不执行。

## 它取代了什么

`CandidateRunner.run()` 在**裁判进程内**造 env、装 provider、跑 skill
（`runner/candidate_runner.py` 第 47–169 行那一段）。那是「第二套执行语义」：
判官自己当了一次 runtime，而选手可见的东西也随之变成「判官决定」而不是
「PAOS 决定」。实测两条路漂了**四个**字段（`live=` / `camera=` / `max_steps=` /
`max_sim_time_s=`），并且公开输入硬编码两项、`run(observation)` 收空 dict。

本模块把那段换成：

```text
PersistentFlow.reset(seed, mode)   ← 下发 episode（B1 私有通道）
        ↓ 选手在 PAOS 沙箱里跑（正式执行链，唯一一条）
PersistentFlow.wait_episode()      ← 等这一轮真的跑完
PersistentFlow.snapshot()          ← judge-only 真值（B2）
        ↓
裁判判定（verify）+ 记录 + 评分 + 门禁
```

## 判分要的数据，一样都不少

| 判分需要 | 从哪来 |
|---|---|
| `initial_layout` / `final_layout` | `snapshot["initial_layout"]` / `["layout"]` |
| `initial_state` / `final_state` | `snapshot["initial_physical"]` / `["physical"]` |
| `event_evidence` | `snapshot["event_evidence"]` |
| `elapsed_s` | **裁判自己测的墙钟**（本模块） |
| `plan_revisions` / `attempts` | **裁判自己数的**（`final` 最多 3 次） |
| `status` | **裁判自己的 verify 结论**（不是选手报的 `status`） |

⚠️ 最后一条最容易搞错：选手永远可以返回 `{"status": "done"}`。
`status` 只能是 `verification['success']` 的映射。

## 为什么不复用 `CandidateRunner`

它把「造场景」「跑 skill」「记录」「评分」焊在一个方法里，而前三样现在都不归
裁判。硬要复用就会把第二套语义带回来 —— 那正是这一步要删的东西。

共用的是**判分与产物那一半**：`TrustedRecorder` / `TrustedContractExtension` /
`ResultGate` 全部原样复用，**一个字节的判定语义都没改**。
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path

__all__ = ["ResidentRunner", "ResidentRunError", "EPISODE_RETRIES"]

#: **链路故障的 episode 级重测次数**（2026-10-19）。
#:
#: 只对 `infrastructure_timeout`（丢包 / 503 / 动作超时且物理状态不明）生效 ——
#: 那是**基础设施的问题，不是选手的问题**，不该让他丢一整个 case。
#: 实测该故障发生率约 **0.8%**（250 episode 里 2 次），所以 1 次重试已足够；
#: 给 2 次是留余量。
#:
#: ⚠️ **`skill_error` / `budget_timeout` / `failure` 都**不重试 —— 那些是选手
#:   或物理的真实结果，重试等于篡改成绩。
EPISODE_RETRIES = 2


class ResidentRunError(RuntimeError):
    """编排层失败（flow 起不来 / 等 episode 超时 / 判定回调崩）——**明确失败**。"""


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    text = json.dumps(value, sort_keys=True, allow_nan=False)
    path.write_text(text, encoding="utf8")
    path.chmod(0o600)


def _player_terminal_status(summary) -> str:
    """从 judge 侧的 episode 记录里取出**选手节点的终态**。

    ⚠️ 形状陷阱（2026-10-18 实测踩到）：`PersistentFlow.attempt_episode` 给的
    `player_summary` 是 `.episode_results.jsonl` 的**整行** ——

        {"seq": 3, "attempt": 1, "result": {"status": "budget_timeout", ...}}

    也就是选手的 `status` 在 **`result` 里面**，不在顶层。
    写成 `summary.get("status")` 会**永远拿到 None**，于是终态分类静默失效
    （第一次 e2e 实测：10 个 episode 全被记成 `unknown`，
    超时/链路故障的区分根本没有生效）。
    """
    if not isinstance(summary, dict):
        return "unknown"
    inner = summary.get("result")
    if isinstance(inner, dict) and inner.get("status"):
        return str(inner["status"])
    return str(summary.get("status") or "unknown")


class ResidentRunner:
    """一次 `case × 5 seed` 的评测：起**一次** flow，跑完所有 episode。"""

    def __init__(self, *, case_id, casebank, reference_root, bundle_root,
                 physics_root, python, skill_zip, output,
                 flow=None, seeds=None, diagnostic_rgb=True, runtime_gpu="0",
                 skill_timeout_s=20, login: Path | None = None,
                 oracle_planning=False):
        from track2_judge.contract.extension import TrustedContractExtension
        from track2_judge.contract.profile import load_config
        from track2_judge.runner.persistent_flow import PersistentFlow

        self.case_id = str(case_id)
        self.casebank = Path(casebank).resolve(strict=True)
        self.reference_root = Path(reference_root)
        self.bundle_root = Path(bundle_root).resolve()
        self.physics_root = Path(physics_root).resolve()
        self.python = str(python)
        self.skill_zip = Path(skill_zip).resolve(strict=True)
        self.output = Path(output).resolve()
        #: 要不要建相机（G3）。**真的会生效** —— 传进 `PersistentFlow`
        #: → 渲染 `runtime.yaml` 的 `camera_required` → runtime 节点。
        self.diagnostic_rgb = bool(diagnostic_rgb)
        #: 裁判侧 Oracle 的 GT 规划（G2）：规划由 **runtime 节点**在它自己的 env 上做。
        self.oracle_planning = bool(oracle_planning)
        self.runtime_gpu = str(runtime_gpu)
        self.skill_timeout_s = float(skill_timeout_s)
        self.seeds = tuple(range(5) if seeds is None else seeds)
        self.config = load_config()
        self.engine = TrustedContractExtension()
        self._flow_factory = flow or (lambda **kw: PersistentFlow(**kw))

        bundle_path = self.casebank / "cases" / self.case_id / "case.json"
        if not bundle_path.is_file():
            raise ResidentRunError(f"case not found: {self.case_id}")
        self.bundle = json.loads(bundle_path.read_text(encoding="utf8"))
        self.task = self.bundle["contract"]
        #: 静止件（桌面等）—— **不算非目标物体**（终态连续评分的保持项口径）。
        self.static_objects = tuple(self.bundle.get("static_objects") or ())
        self.engine.validate_task(self.task)
        self._task_config = self.engine.config_for_task(self.task, self.config)

    # ── 公开入口

    def run(self) -> dict:
        """跑完全部 episode 并返回 `diagnosis`（与旧路径**同形状**）。"""
        from track2_judge.runner.recorder import TrustedRecorder
        from track2_judge.runner.legacy_util import canonical_json_hash
        from track2_judge.scoring.result_gate import ResultGate, block, write_receipt

        output = self.output
        if output.exists() and any(output.iterdir()):
            raise ResidentRunError("output must be new/empty")
        output.mkdir(parents=True, mode=0o700, exist_ok=True)
        if os.name == "posix":
            if output.stat().st_uid != os.getuid():
                raise PermissionError("judge output owner mismatch")
            output.chmod(0o700)

        suite = {"config": copy.deepcopy(self.config), "tasks": [self.task]}
        skill_hash = hashlib.sha256(self.skill_zip.read_bytes()).hexdigest()
        recorder = TrustedRecorder(output / "artifacts", skill_hash=skill_hash,
                                   suite_hash=canonical_json_hash(suite),
                                   runs_dir=output)
        runs_log: list[dict] = []
        health_reports: list[dict] = []
        event_reports: list[dict] = []
        lifecycles: list[dict] = []
        trace: list[dict] = []

        flow = self._flow_factory(
            bundle_root=self.bundle_root, physics_root=self.physics_root,
            python=self.python, log_path=output / "flow.log",
            runtime_gpu=self.runtime_gpu, work_dir=self.physics_root.parent / "work",
            player_package=self.skill_zip,
            # ★ G3：把「要不要相机」接到 node 侧（渲染进 `runtime.yaml`）。
            #   以前它停在 `self.diagnostic_rgb` 上没人读 —— 那是个假开关。
            camera=self.diagnostic_rgb,
            # ★ G2：Oracle 的 GT 规划在 runtime 侧做（裁判私有通道）。
            oracle_planning=self.oracle_planning)
        try:
            flow.start()
            # ★ 2026-10-18 取消 first/final：每 seed **1** 次（单 case = 5 episode）。
            #
            # ★ 2026-10-19 episode 级重测：`infrastructure_timeout`（链路故障：
            #   丢包 / 503 / 动作超时且物理状态不明）**不是选手的错**，不该让他
            #   丢掉一整个 case。同一个 seed **重建场景重跑**，最多
            #   `EPISODE_RETRIES` 次；只有**全部重试仍失败**才如实判 FAILED。
            #
            #   ⚠️ 重跑必须**干净地丢弃**上一次的产物（见 `_rollback_episode`）——
            #   否则 `recorder.runs` / `engine.packets` 里会留下两次 attempt，
            #   `extension.evaluate()` 的 `len(runs) != len(tasks)*R` 直接 `ValueError`。
            for seed in self.seeds:
                    retries = 0
                    for attempt in range(EPISODE_RETRIES + 1):
                        marks = self._mark_episode(recorder, runs_log, health_reports,
                                                   event_reports, lifecycles, trace)
                        record = self._run_one(flow, recorder, seed,
                                               runs_log, health_reports,
                                               event_reports, lifecycles, trace)
                        if record["status"] != "infrastructure_timeout":
                            break
                        if attempt >= EPISODE_RETRIES:
                            break
                        retries += 1
                        print(json.dumps({"seed": seed, "episode_retry": retries,
                                          "reason": record["status"]},
                                         ensure_ascii=False), flush=True)
                        self._rollback_episode(marks, recorder, runs_log, health_reports,
                                               event_reports, lifecycles, trace)
                    # 重试次数如实记进本 episode（诊断用，不进分）
                    if retries:
                        runs_log[-1]["episode_retries"] = retries
                    print(json.dumps({k: v for k, v in runs_log[-1].items()
                                      if k not in ("tool_calls", "verification")},
                                     ensure_ascii=False), flush=True)
            # ★ Q14：整轮跑完取一次动作耗时分布（**只读**，不影响判定）。
            #   放在 `stop()` **之前** —— 停了就够不着控制口了。
            #   取不到不算失败（它是诊断数据，不是评测产物）。
            try:
                self._timings = flow.timings()
            except Exception as exc:                            # noqa: BLE001
                self._timings = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        finally:
            flow.stop()

        # ── 落盘（与 `candidate_runner.py` 同一批文件名，判分侧不用改）
        (output / "suite.json").write_text(
            json.dumps(suite, sort_keys=True, allow_nan=False), encoding="utf8")
        recorder.finalize(suite_path=output / "suite.json",
                          output_path=output / "runs.json")
        for name in ("evidence.json", "runner_manifest.json"):
            path = output / "artifacts" / name
            data = json.loads(path.read_text(encoding="utf8"))
            data.update(source="resident_runner", runner_version="resident-v1",
                        simulator_version="PAOS persistent flow")
            for entry in data.get("runs", []):
                entry.update(source="resident_runner", runner_version="resident-v1",
                             simulator_version="PAOS persistent flow")
            _write_json(path, data)
        _write_json(output / "physics_health_diagnostic.json", health_reports)
        _write_json(output / "physical_trace.json", trace)
        # Q14：动作耗时分布（`run_suite` 会汇总成 `timings_summary.json`）
        if getattr(self, "_timings", None) is not None:
            _write_json(output / "timings.json", self._timings)
        _write_json(output / "supplemental_verification.json", [])
        _write_json(output / "trusted_event_evidence.json", event_reports)
        _write_json(output / "skill_lifecycle.json", lifecycles)

        diagnosis = {
            "candidate_status": "NOT_READY", "official_score": None,
            "episodes": runs_log, "physics_health": health_reports,
            "physics_health_policy": "independent diagnostics; no automatic scoring or clamp",
            "source": "ACTUAL selected TTG candidate, source collision assets; PAOS persistent flow",
            "physical_events": "PHYSX SUBSTEP EVIDENCE WHEN CONFIGURED; semantics pending official approval",
            "isolation": "namespace_chroot (contestant side, PAOS sandbox)",
            "exception_policy": ("Frozen V3 has no execution-status field; raw diagnostic may retain "
                                 "auxiliary points or pose success on a timeout. Operational readiness "
                                 "always fails on error/timeout."),
            "execution_chain": "PAOS persistent flow (resident_runner)",
        }
        expected = len(self.seeds)
        complete_repeats = len(self.seeds) == int(self.config["repeats"])
        # ★ 2026-10-18：**四种可评分终态** —— success / failure / budget_timeout / skill_error。
        #   `budget_timeout` 的物理终态是有效的（判据见 `_run_one`），照常评分；
        #   `infrastructure_timeout`（链路故障）**不**在内，它会走门禁阻断。
        SCORABLE = ("success", "failure", "budget_timeout", "skill_error")
        # ① 兜底：任一 episode 的 provider 侧动作账本不完整 → 不算有效。
        #   它与上面那两个条件**独立** —— 专抓「选手吞异常、status 看着正常」
        #   那一类（加速方案的异常分类够不着，见 §3.1）。
        incomplete = [r for r in runs_log
                      if r.get("execution_integrity") == "INCOMPLETE"]
        valid = (len(runs_log) == expected
                 and not any(r.get("status") not in SCORABLE for r in runs_log)
                 and not incomplete)
        if incomplete:
            diagnosis["error_category"] = "ACTION_EXECUTION_INCOMPLETE"
        diagnosis["gt_access_status"] = "PASS" if valid else "FAILED"
        # ⚠️ **少跑几个 seed 就不是评测，只是冒烟。**
        #
        #    `mesa_eval.validate_config` 硬要求 `repeats == 5`（sim），而
        #    `TrustedContractExtension.evaluate` 会 `len(runs) != tasks*R*2 → ValueError`。
        #    所以「只跑 seed 0」这种诊断**不能**也产出一个分数 —— 那会是一条
        #    看起来合法、其实少了一半配对的成绩。这里明确标成诊断并**不评分**。
        if valid and not complete_repeats:
            diagnosis.update(
                scoring_status="DIAGNOSTIC_PARTIAL_SEEDS",
                candidate_status="DIAGNOSTIC_ONLY",
                diagnostic_note=(f"只跑了 seed {list(self.seeds)}（repeats="
                                 f"{self.config['repeats']}）—— 只用于冒烟，不评分"),
                episodes_run=len(runs_log))
        score_name = "extension_candidate_score.json"
        replay_ok = False
        if valid and complete_repeats:
            try:
                report = self.engine.evaluate(
                    suite, {"schema_version": 1, "runs": recorder.runs}, output)
                _write_json(output / score_name, report)
                replay_ok = True
                diagnosis["extension_candidate_score_100"] = report["candidate_score_100"]
                # ★ 2026-10-18：新分已产出（`extension_candidate_score_100`）——
                #   这就是正式分。标 `CONTINUOUS_TERMINAL_STATE`（旧词
                #   `FROZEN_AUTHOR_PDF_RULES` 对应的是已停用的冻结讲解算，见 §3.6）。
                diagnosis.update(
                    scoring_status="CONTINUOUS_TERMINAL_STATE",
                    candidate_status=(
                        "EVALUATION_COMPLETED"
                        if all(r["status"] in SCORABLE for r in runs_log)
                        else "EXECUTION_FAILURE"))
            except Exception as replay_error:                   # noqa: BLE001
                diagnosis.update(
                    scoring_status="REFUSED: verifier/scorer replay failed",
                    candidate_status="EXECUTION_FAILURE",
                    error_category="VERIFIER_OR_SCORER_" + type(replay_error).__name__)
        else:
            diagnosis["scoring_status"] = (
                diagnosis.get("scoring_status", "REFUSED: missing trustworthy state"))

        _write_json(output / "diagnosis.json", diagnosis)
        if replay_ok:
            write_receipt(output, score_name, expected_runs=expected,
                          verifier="trusted_contract_extension")
            gate = ResultGate().check(output)
        elif valid and not complete_repeats:
            # 诊断跑：产物齐全但没有分数 —— 如实标成诊断，**不要**伪装成
            # `BLOCKED`（那看起来像「评测器坏了」），也不要跳过 gate 文件。
            gate = block(output, "DIAGNOSTIC_PARTIAL_SEEDS_NO_SCORE")
        else:
            gate = block(output, "VERIFIER_OR_STATE_INCOMPLETE")
        diagnosis["result_gate"] = gate
        return diagnosis

    # ── 一个 episode

    def _mark_episode(self, recorder, runs_log, health_reports,
                      event_reports, lifecycles, trace) -> dict:
        """记录**本轮 episode 之前的**状态，供 `_rollback_episode` 回退。

        `_run_one` 会往 6 个地方写东西：

        ```text
        runs_log / health_reports / event_reports / lifecycles / trace   ← 都是 list
        recorder.runs / recorder.evidence                                ← recorder 内部
        self.engine.packets                                              ← 判定包的 dict
        ```

        前 5 个是 append-only 的 list —— 记下长度即可 `del [n:]`。
        后两个是 dict/list，用**快照 key 集合**回退（它们按
        `(task_id, seed)` / 顺序键写入，所以按 key 删是精确的）。
        """
        return {
            "runs_log": len(runs_log),
            "health_reports": len(health_reports),
            "event_reports": len(event_reports),
            "lifecycles": len(lifecycles),
            "trace": len(trace),
            "recorder_runs": len(recorder.runs),
            "recorder_evidence": len(recorder.evidence),
            "packet_keys": set(self.engine.packets),
            # `_timings_seen` 是「上一轮看到的累计 unfinished 数」的基线。
            # 重跑要从**同一个基线**比，否则会把被丢弃那次 attempt 的残缺
            # 算到重跑头上（→ 误判 INCOMPLETE）。
            "timings_seen": getattr(self, "_timings_seen", None),
        }

    def _rollback_episode(self, marks, recorder, runs_log, health_reports,
                          event_reports, lifecycles, trace) -> None:
        """把一个失败的 episode 留下的产物**彻底清掉**，好让重跑从干净状态开始。

        ⚠️ 不清的话 `recorder.runs` 会有两条同 seed 记录 →
        `extension.evaluate()` 的 `len(runs) != len(tasks)*R` 直接 `ValueError`
        → 整个 case 反而更糟。

        ⚠️ `recorder.states_dir` 下那对 `<stem>_initial/final.json` **不删** ——
        重跑会**用同一个 stem 覆盖**它们（stem 只含 task_index/seed，不含 attempt），
        所以不会残留第二份；删反而多一次 IO 且可能与正在写的文件打架。
        """
        del runs_log[marks["runs_log"]:]
        del health_reports[marks["health_reports"]:]
        del event_reports[marks["event_reports"]:]
        del lifecycles[marks["lifecycles"]:]
        del trace[marks["trace"]:]
        del recorder.runs[marks["recorder_runs"]:]
        del recorder.evidence[marks["recorder_evidence"]:]
        # `engine.packets` 是 dict —— 按 key 回退到快照那一刻
        stale = set(self.engine.packets) - marks["packet_keys"]
        for key in stale:
            self.engine.packets.pop(key, None)
        # 累计计数器的基线也要回退（见 `_mark_episode` 的说明）
        self._timings_seen = marks["timings_seen"]

    def _run_one(self, flow, recorder, seed, runs_log, health_reports,
                 event_reports, lifecycles, trace) -> dict:
        started = time.monotonic()
        initial_seen: dict = {}

        def verdict(attempt, snapshot):
            """**裁判的判定** —— 已不再用于追加执行（单模式），保留签名兼容。"""
            result = self._verify(snapshot)
            initial_seen.setdefault("value", snapshot.get("initial_layout"))
            if attempt == 1:
                event_reports.append({"task_index": 0, "seed": seed,
                                      **self._event_block(snapshot)})
            return bool(result["success"])

        # ★ 单模式：每 seed 只调 `run()` **1** 次（不再有 final 的追加执行）。
        out = flow.attempt_episode(self.case_id, seed, max_attempts=1)
        snapshot = out["snapshot"]
        elapsed = max(time.monotonic() - started, 1e-6)

        # ★ ① 兜底可见性（2026-10-18）：**provider 侧的动作账本**。
        #
        #   加速方案的 `infrastructure_timeout` 走的是**异常分类**，挡不住
        #   「选手把动作异常 try/except 吞掉、然后正常返回」这一种 —— 那时
        #   player status 是 `ok`、episode 是 `success`/`failure`，门禁放行。
        #   但 provider 侧「有 start 无 finish」的痕迹**仍然在**。
        #
        #   判据只取 `unfinished_or_rejected` 的**增量**（它在整轮 flow 上累积，
        #   因为 `_executions` 跨 episode 不清）—— 不需要知道「期望动作数」，
        #   而期望值本来就没有可信先验（弱 skill 1 次、oracle 几十次）。
        #   见 plan/Track2_丢包自锁修复方案.md §3.2。
        execution_integrity = "OK"
        timings_seen = getattr(self, "_timings_seen", None)
        try:
            timings_now = flow.timings()
        except Exception:                                       # noqa: BLE001
            timings_now = None       # 取不到就不判 —— 它是兜底，不是主判据
        if isinstance(timings_now, dict):
            unfinished = int(timings_now.get("unfinished_or_rejected") or 0)
            previous = 0 if timings_seen is None else int(
                timings_seen.get("unfinished_or_rejected") or 0)
            if unfinished > previous:
                execution_integrity = "INCOMPLETE"
            self._timings_seen = timings_now

        # ★ 判定用**最终那一轮的快照**（`final` 的中间轮次不留存 —— 原实现
        #   `capture()` 也是最后一个 attempt 覆盖前者，语义一致）。
        verification = self._verify(snapshot)
        status = "success" if verification["success"] else "failure"

        # ★ 2026-10-18：选手侧终态分类（见 `sandbox/rpc.py` 的 `BudgetTimeout`）。
        #
        #   三种**可评分终态** —— `success` / `failure` / `budget_timeout`：
        #     * `budget_timeout` = 正常预算耗尽（600 s 墙钟 / 控制步预算）。
        #       语义是「到时间停止 → 冻结当前物理 → 照常按 P1/P2/C1 判分」。
        #       它的 `s` 照样来自**物理终态**（`_verify` 已经算过），所以计分不变。
        #     * 唯一的例外是**链路故障** `infrastructure_timeout`
        #       （丢包 / 503 / 动作超时且物理状态不明）—— 它**不得伪装**成正常超时，
        #       直接用自己的名字，`ResultGate` 按执行异常阻断。
        player_status = _player_terminal_status(out["player_summary"])
        if player_status == "budget_timeout":
            status = "budget_timeout"
        elif player_status == "skill_error":
            # ★ F2：选手包自己抛异常 —— **可评分**（按冻结终态照常判分）。
            status = "skill_error"
        elif player_status in ("infrastructure_timeout", "timeout",
                               "security_not_ready"):
            status = "infrastructure_timeout"

        initial_layout = snapshot.get("initial_layout")
        final_layout = snapshot.get("layout")
        if initial_layout is None or final_layout is None:
            raise ResidentRunError(
                "snapshot 缺少 initial_layout/layout —— runtime 侧没采到初态")

        # 复算一次 `capture` + `bind` 的等价物（`extension.evaluate` 会逐字段重放比对）
        self.engine.capture(self.task, seed,
                            snapshot.get("initial_physical"),
                            snapshot.get("physical"), verification,
                            self._task_config, self._event_block(snapshot),
                            initial_layout, final_layout,
                            static_objects=self.static_objects)
        for attempt_entry in out["rounds"]:
            trace.append({"task_index": 0, "seed": seed,
                          "attempt": attempt_entry["attempt"],
                          "physical": (attempt_entry["snapshot"] or {}).get("physical")})

        recorded = recorder.record(
            task_index=0, task=self.task, seed=seed, environment="sim",
            initial_layout=initial_layout, final_layout=final_layout,
            elapsed_s=elapsed,
            event_violations=self._event_block(snapshot).get("event_counts", {}),
            verification=verification,
            # ⚠️ `tool_calls` **不来自选手** —— 它是「裁判看到的调用账本」。
            #    常驻结构下这条账在 gateway 侧，我们**如实给空列表**并标注，
            #    而不是伪造一份。`ResultGate` 不查它（`result_gate.py` 只看
            #    status/verification/elapsed/plan_revisions/state）。
            tool_calls=[{"note": "tool call ledger lives in the PAOS gateway; "
                                 "not mirrored here (resident chain)"}],
            status=status)
        self.engine.bind_recorded_run(self.task, seed, recorded)

        health_reports.append({"task_index": 0, "seed": seed,
                               "summary": snapshot.get("physics_health"),
                               "private_evidence": None,
                               "private_evidence_sha256": None,
                               "note": "substep ring buffer lives in the PAOS runtime"})
        lifecycles.append({"seed": seed,
                           "events": ((out["player_summary"] or {}).get("result") or {})
                                     .get("lifecycle")})
        runs_log.append({"seed": seed, "status": status,
                         "player_status": player_status,
                         # ① 兜底判据：`INCOMPLETE` = provider 侧有「有 start
                         # 无 finish」的动作（与选手是否吞异常无关）
                         "execution_integrity": execution_integrity,
                         "attempts": int(out["attempts"]),
                         "error_category": player_status
                         if status not in ("success", "failure") else None,
                         "skill_active_wall_s": elapsed,
                         "total_wall_s": elapsed,
                         "success_scope": "CONTINUOUS_TERMINAL_STATE"
                         if "scoring_profile" in self.task
                         else "TRUSTED_DIAGNOSTIC_EXTENSION"})
        return runs_log[-1]

    # ── 判定 / 证据

    def _verify(self, snapshot) -> dict:
        initial = snapshot.get("initial_layout")
        final = snapshot.get("layout")
        if initial is None or final is None:
            raise ResidentRunError("snapshot 缺 initial_layout/layout")
        try:
            return self.engine.verify(
                self.task, initial, final, self.config,
                event_evidence=self._event_block(snapshot),
                initial_state=snapshot.get("initial_physical"),
                final_state=snapshot.get("physical"),
                static_objects=self.static_objects)
        except Exception as exc:                                # noqa: BLE001
            # 判定抛错 = 这一次评测不可信。**如实记成 failure 并带上原因**，
            # 不要吞掉（吞掉的后果是「用上一次的结论」这种静默错分）。
            raise ResidentRunError(
                f"裁判判定失败（{type(exc).__name__}: {str(exc)[:300]}）") from exc

    @staticmethod
    def _event_block(snapshot) -> dict:
        """事件证据的形状归一。

        `event_evidence` 可能带 `error`（runtime 侧采不到）—— 归成
        「未配置」而不是让 verify 崩在 KeyError 上。
        """
        evidence = snapshot.get("event_evidence")
        if not isinstance(evidence, dict) or "error" in evidence:
            return {"event_counts": {}, "coverage": "NOT CONFIGURED",
                    "P5": {"status": "NOT CONFIGURED"}}
        return evidence
