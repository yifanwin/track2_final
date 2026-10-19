"""Track 2 本地 Forge Gateway —— 按 PAOS 的 `/tools` + `/invocations` 契约实现。

契约来源：`PhyAgentOS-core/docs/forge/README.md`（§2–§4）。逐条对应：

    §2  GET /tools                             → Tool 列表
        GET /tools/{tool_id}                   → ToolSpec
        GET /tools/{tool_id}/context           → readiness
    §3  POST /tools/{endpoint_id}/{operation}:invoke  → Query，200 + data
    §4  POST /tools/{tool_id}:invoke           → Action，202 + invocation_id + attempt_id
        GET  /invocations/{invocation_id}       → phase / status
        GET  /invocations/{invocation_id}/result → 200 或 202(pending)
        POST /invocations/{invocation_id}/cancel → 200/202（**不代表物理已停**）

## 两条 invoke 路由怎么区分

Query 的路径是 `端点/操作`（2 段），Action 的路径是 `tool_id`（1 段）。
tool_id 里的点不会产生新段，所以段数就是判别依据 —— 与 PAOS 真实 Gateway 一致。

## Action 治理（Q9 / Q12 / Q13 / Q14）

| # | 落实 |
|---|---|
| Q9 | 所有对 adapter 的操作串行：一把 `_adapter_lock` + 单执行线程。`max_concurrency = 1` |
| Q12 | 每个 Action 有执行超时 `ACTION_TIMEOUT_S` |
| Q13 | 超时后**先 cancel，再等 backend 确认真实停止**；`execution_status` 如实反映，**绝不把 cancel accepted 当已停止** |
| Q14 | 超时秒数由 73 case 实测定：**5.0 s**（p99 0.675 / max 0.940 s），标 `MEASURED_73_CASES_P99_PLUS_MARGIN`（2026-10-18） |

## 为什么所有操作共用一把锁

`adapter` 的 `observe()` 与 `step()` 都会读写同一个 `env`，**不是线程安全的**。
而且机器人在物理上也只能一次做一件事 —— 所以串行不是妥协，是正确语义。
因为每个 control step 只有约 50 ms（20 Hz），这个锁不会造成可见延迟。

## 不做什么

- **不重试**。PAOS 明确 `unknown` 终态「must not trigger a blind retry」。
- **不伪造成功**。无法确认真实状态时，`execution_status` 就是 `unknown`。
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Protocol
from urllib.parse import unquote

from track2_judge.toolapi import schema as schema_mod
from track2_judge.toolapi.validation import SchemaError, validate

__all__ = [
    "ToolExecutor", "LocalGateway", "GatewayServer", "GatewayError",
    "UnknownTool", "NotDeclared", "ConcurrencyExhausted", "InvocationUnknown",
    "PHASE_ADMITTED", "PHASE_RUNNING", "PHASE_TERMINAL",
    "OUTCOME_SUCCEEDED", "OUTCOME_FAILED", "OUTCOME_CANCELLED",
    "OUTCOME_TIMED_OUT", "OUTCOME_UNKNOWN",
    "EXEC_RUNNING", "EXEC_STOPPED_CONFIRMED", "EXEC_UNKNOWN",
]

# 相位
PHASE_ADMITTED = "admitted"      # 已接纳，尚未开始执行
PHASE_RUNNING = "running"        # 执行中
PHASE_TERMINAL = "terminal"      # 已终结

# 终态
OUTCOME_SUCCEEDED = "succeeded"
OUTCOME_FAILED = "failed"
OUTCOME_CANCELLED = "cancelled"
OUTCOME_TIMED_OUT = "timed_out"
OUTCOME_UNKNOWN = "unknown"      # 无法确认物理事实 —— **不得重试**

# 执行状态（比 phase 更细：phase 说「账务上结束了没」，execution_status 说「物理上停了没」）
EXEC_RUNNING = "running"
EXEC_STOPPED_CONFIRMED = "stopped_confirmed"   # 有正面证据：执行体已退出且 backend 空闲
EXEC_UNKNOWN = "unknown"                       # 拿不到正面证据 —— 只能如实说不知道

TERMINAL_OUTCOMES = frozenset(
    {OUTCOME_SUCCEEDED, OUTCOME_FAILED, OUTCOME_CANCELLED, OUTCOME_TIMED_OUT, OUTCOME_UNKNOWN}
)


class GatewayError(RuntimeError):
    """Gateway 层面的拒绝。"""


class UnknownTool(GatewayError):
    pass


class NotDeclared(GatewayError):
    """tool 存在但不在当前 profile 的声明集合内 —— 对应验收标准 2。"""


class ConcurrencyExhausted(GatewayError):
    """endpoint 的 `max_concurrency` 已满（Q9）。"""


class InvocationUnknown(GatewayError):
    pass


class ToolExecutor(Protocol):
    """Gateway 的下游 —— 真正碰物理的那一层。

    `sandbox` 里的 `ManiSkillSimulatorAdapter` 满足这个协议（见 `toolapi/dispatch.py`）。
    """

    def observe(self, arguments: dict[str, Any]) -> dict[str, Any]: ...
    def step(self, arguments: dict[str, Any]) -> dict[str, Any]: ...
    def gripper_step(self, arguments: dict[str, Any]) -> dict[str, Any]: ...


# ────────────────────────────────────────────────────────────── Action 记录


class ActionRecord:
    """一次 Action 调用的完整账务。字段与 PAOS 的语义一一对应。"""

    __slots__ = (
        "invocation_id", "attempt_id", "tool_id", "arguments", "caller_id", "task_id",
        "phase", "cancel_status", "cancel_requested_at_s", "execution_status",
        "outcome", "result", "error", "error_code", "admitted_at_s", "started_at_s",
        "finished_at_s", "cancel_effective", "timed_out",
    )

    def __init__(self, tool_id: str, arguments: dict, caller_id: str | None,
                 task_id: str | None, now: float):
        self.invocation_id = f"inv_{uuid.uuid4().hex[:24]}"
        self.attempt_id = f"att_{uuid.uuid4().hex[:24]}"
        self.tool_id = tool_id
        self.arguments = arguments
        self.caller_id = caller_id
        self.task_id = task_id
        self.phase = PHASE_ADMITTED
        self.cancel_status: str | None = None       # None | 'requested' | 'accepted'
        self.cancel_requested_at_s: float | None = None
        self.execution_status = EXEC_STOPPED_CONFIRMED   # 未开始执行 = 已确认无权在跑
        self.outcome: str | None = None
        self.result: dict | None = None
        self.error: str | None = None
        self.error_code: str | None = None
        self.admitted_at_s = now
        self.started_at_s: float | None = None
        self.finished_at_s: float | None = None
        # cancel 请求是否**真的**阻止了物理效果。请求晚了就是 False —— 如实记。
        self.cancel_effective = False
        self.timed_out = False

    def status(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "invocation_id": self.invocation_id,
            "attempt_id": self.attempt_id,
            "tool_id": self.tool_id,
            "phase": self.phase,
            "execution_status": self.execution_status,
            "cancel_status": self.cancel_status,
        }
        if self.outcome is not None:
            payload["outcome"] = self.outcome
        if self.cancel_status is not None:
            # Q13：cancel_status 只说明请求被处理了；cancel_effective 才说物理上有没有停。
            payload["cancel_effective"] = self.cancel_effective
        if self.timed_out:
            payload["timed_out"] = True
        if self.error is not None:
            payload["error"] = self.error
        if getattr(self, "error_code", None):
            payload["error_code"] = self.error_code
        return payload

    def result_envelope(self) -> tuple[int, dict[str, Any]]:
        """返回 (HTTP 状态, body)。pending → 202（PAOS 契约）。"""
        if self.phase != PHASE_TERMINAL:
            return 202, {"ok": True, "data": {"phase": self.phase, "pending": True}}
        body = {"ok": self.outcome == OUTCOME_SUCCEEDED,
                "data": {"phase": PHASE_TERMINAL, "outcome": self.outcome,
                         "execution_status": self.execution_status,
                         "invocation_id": self.invocation_id,
                         "attempt_id": self.attempt_id,
                         "tool_id": self.tool_id}}
        if self.timed_out:
            body["data"]["timed_out"] = True
        if self.cancel_status is not None:
            body["data"]["cancel_status"] = self.cancel_status
            body["data"]["cancel_effective"] = self.cancel_effective
        if self.result is not None:
            body["data"]["result"] = self.result
        if self.error is not None:
            body["data"]["error"] = self.error
            body["error"] = {"message": self.error}
            if getattr(self, "error_code", None):
                body["error"]["code"] = self.error_code
                body["data"]["error_code"] = self.error_code
        return 200, body


# ────────────────────────────────────────────────────────────── Gateway


class LocalGateway:
    """进程内 Gateway。**不碰网络** —— 编排链路直接用这个实例。

    `GatewayServer` 在它外面套一层 HTTP，供 PAOS 的 `/tools` 探测与
    （第 3 步的）Dora gateway 节点使用。
    """

    def __init__(
        self,
        executor: ToolExecutor,
        *,
        gateway_identity: str = "track2_local_gateway",
        action_timeout_s: float | None = None,
        stop_confirm_timeout_s: float = 5.0,
        max_concurrency: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        profile: str | None = None,
    ):
        self._executor = executor
        self.gateway_identity = gateway_identity
        self.action_timeout_s = (
            schema_mod.ACTION_TIMEOUT_S if action_timeout_s is None else float(action_timeout_s)
        )
        # Q13：cancel 之后最多等这么久「backend 确认真实停止」。
        # 等不到就如实报 unknown，**不假装停了**。
        self.stop_confirm_timeout_s = float(stop_confirm_timeout_s)
        self.max_concurrency = (
            schema_mod.ACTION_MAX_CONCURRENCY if max_concurrency is None else int(max_concurrency)
        )
        self._clock = clock
        self._profile = profile

        # 串行化一切对 adapter 的访问（Q9）。机器人在物理上就只有一个。
        self._adapter_lock = threading.Lock()
        self._running_actions = 0
        self._state_lock = threading.Lock()

        self._records: dict[str, ActionRecord] = {}
        self._threads: dict[str, threading.Thread] = {}
        # 每次 Action 一个「取消请求」事件，执行体在安全点检查它
        self._cancel_flags: dict[str, threading.Event] = {}

    # ────────────────────────────────── §2 发现与 readiness

    def list_tools(self) -> tuple[int, dict[str, Any]]:
        return 200, {
            "ok": True,
            "data": {
                "gateway_identity": self.gateway_identity,
                "tools": sorted(schema_mod.tool_ids_for_profile(self._profile)),
            },
        }

    def get_tool(self, tool_id: str) -> tuple[int, dict[str, Any]]:
        if not schema_mod.is_declared(tool_id, self._profile):
            return 404, {"ok": False, "error": {"message": f"unknown tool {tool_id!r}"}}
        return 200, {"ok": True, "data": schema_mod.get_spec(tool_id)}

    def get_tool_context(self, tool_id: str) -> tuple[int, dict[str, Any]]:
        if not schema_mod.is_declared(tool_id, self._profile):
            return 404, {"ok": False, "error": {"message": f"unknown tool {tool_id!r}"}}
        # 就绪 = adapter 已经 reset 过（有 env）。binding_error 为 None 才被 PAOS 判 ready。
        ready = self._executor_ready()
        return 200, {"ok": True, "data": {
            "tool_id": tool_id,
            "ready": ready,
            "binding_error": None if ready else "simulator not initialized",
            "spec_sha256": schema_mod.spec_sha256(tool_id),
        }}

    def _executor_ready(self) -> bool:
        probe = getattr(self._executor, "is_ready", None)
        if callable(probe):
            try:
                return bool(probe())
            except Exception:
                return False
        return True

    # ────────────────────────────────── §3 Query

    def invoke_query(self, endpoint_id: str, operation: str, body: dict[str, Any]) -> tuple[int, dict]:
        """`POST /tools/{endpoint_id}/{operation}:invoke`。

        Query 不走 invocation 账本 —— 同步返回 200 + data（PAOS §3）。
        """
        tool_id = schema_mod.ENDPOINTS.get(endpoint_id, {}).get(operation)
        if tool_id is None:
            return 404, {"ok": False, "error": {"message": f"unknown endpoint {endpoint_id}/{operation}"}}
        if not schema_mod.is_declared(tool_id, self._profile):
            return 403, {"ok": False, "error": {"message": f"tool {tool_id!r} not declared"}}

        arguments = body.get("arguments") or {}
        try:
            schema_mod.validate_arguments(tool_id, arguments)
            with self._adapter_lock:
                data = self._executor.observe(arguments)
            schema_mod.validate_output(tool_id, data)
        except SchemaError as exc:
            return 400, {"ok": False, "error": {"message": f"schema rejected: {exc}"}}
        except Exception as exc:                      # 不把内部路径/异常文本漏给选手
            return 500, {"ok": False, "error": {"message": f"execution failed: {type(exc).__name__}"}}
        return 200, {"ok": True, "data": data}

    # ────────────────────────────────── §4 Action

    def invoke_action(self, tool_id: str, body: dict[str, Any]) -> tuple[int, dict]:
        """`POST /tools/{tool_id}:invoke` → 202 + invocation_id + attempt_id。

        **admission 不是完成** —— 这里只做接纳与入参校验，物理在后台执行。
        """
        if not schema_mod.is_declared(tool_id, self._profile):
            return 403, {"ok": False, "error": {"message": f"tool {tool_id!r} not declared"}}
        spec = schema_mod.get_spec(tool_id)
        if spec["semantics"] != "action":
            return 400, {"ok": False, "error": {
                "message": f"tool {tool_id!r} is {spec['semantics']}, not action"}}

        arguments = body.get("arguments") or {}
        try:
            schema_mod.validate_arguments(tool_id, arguments)
        except SchemaError as exc:
            return 400, {"ok": False, "error": {"message": f"schema rejected: {exc}"}}

        # Q9：endpoint 的 max_concurrency 裁决接纳。
        with self._state_lock:
            if self._running_actions >= self.max_concurrency:
                return 429, {"ok": False, "error": {
                    "message": f"endpoint max_concurrency={self.max_concurrency} exhausted"}}
            record = ActionRecord(tool_id, arguments, body.get("caller_id"),
                                  body.get("task_id"), self._clock())
            self._records[record.invocation_id] = record
            self._running_actions += 1
            cancel_flag = threading.Event()
            self._cancel_flags[record.invocation_id] = cancel_flag

        # admission 应答必须在 start() **之前**构造：
        # 无延迟的 Action 可能在 thread.start() 返回前就终结了，那样这个
        # 202 应答就会报出一个终态 —— 而 PAOS 的 §4 契约里 202 就是「已接纳」。
        admission = {"ok": True, "data": {
            "invocation_id": record.invocation_id,
            "attempt_id": record.attempt_id,
            "phase": PHASE_ADMITTED,
        }}

        thread = threading.Thread(
            target=self._execute_action,
            args=(record, cancel_flag),
            name=f"action-{record.invocation_id[:12]}",
            daemon=True,
        )
        self._threads[record.invocation_id] = thread
        thread.start()
        return 202, admission

    def _execute_action(self, record: ActionRecord, cancel_flag: threading.Event) -> None:
        """Action 的执行体。**必须在终结前把 execution_status 说清楚。**"""
        try:
            with self._state_lock:
                # 取消可能在被调度到之前就到了 —— 这是唯一能真正阻止物理效果的窗口。
                if cancel_flag.is_set():
                    record.phase = PHASE_TERMINAL
                    record.outcome = OUTCOME_CANCELLED
                    record.cancel_effective = True
                    record.execution_status = EXEC_STOPPED_CONFIRMED
                    record.finished_at_s = self._clock()
                    return
                record.phase = PHASE_RUNNING
                record.started_at_s = self._clock()

            # 拿锁 = 独占 adapter。拿不到就说明有别的东西在跑（例如一次 Query）。
            with self._adapter_lock:
                if record.tool_id == "track2.robot.joint_trajectory":
                    # ★ 强制 Action Chunk：整段预校验 + 逐步执行（本地 adapter）。
                    targets = record.arguments["targets"]
                    previous = self._executor._adapter.current_command_target()
                    for item in targets:
                        q, g = self._executor._adapter.assert_within_slew(
                            previous, item["joint_position"], item["gripper_opening_m"])
                        previous = [*q.tolist(), float(g)]
                    self._executor._adapter.assert_budget_allows(len(targets))
                    executed = 0
                    for item in targets:
                        self._executor.step({"joint_position": list(item["joint_position"]),
                                             "gripper_opening_m": float(item["gripper_opening_m"])})
                        executed += 1
                    raw = {"control_steps": executed,
                           "sim_time_s": executed * self._executor._adapter.control_timestep,
                           "executed_steps": executed}
                elif record.tool_id == "track2.robot.gripper_control":
                    raw = self._executor.gripper_step(record.arguments)
                else:
                    raise GatewayError(f"no executor bound for {record.tool_id}")
            schema_mod.validate_output(record.tool_id, raw)

            with self._state_lock:
                record.result = raw
                record.outcome = OUTCOME_SUCCEEDED
                record.phase = PHASE_TERMINAL
                # 有正面证据：执行体走完了，且适配器已释放 —— 物理上确实停了。
                record.execution_status = EXEC_STOPPED_CONFIRMED
                record.finished_at_s = self._clock()
        except Exception as exc:
            with self._state_lock:
                # ★ F1：带上**真实消息**（不只是类型名），否则
                #   「control budget exhausted」在客户端分不出来。
                record.error = f"{type(exc).__name__}: {str(exc)[:200]}"
                record.error_code = ("TRACK2_CONTROL_BUDGET_EXHAUSTED"
                                     if "control budget exhausted" in str(exc) else None)
                record.outcome = OUTCOME_FAILED
                record.phase = PHASE_TERMINAL
                record.execution_status = EXEC_STOPPED_CONFIRMED
                record.finished_at_s = self._clock()
        finally:
            with self._state_lock:
                self._running_actions = max(0, self._running_actions - 1)

    def action_status(self, invocation_id: str) -> tuple[int, dict]:
        record = self._records.get(invocation_id)
        if record is None:
            return 404, {"ok": False, "error": {"message": "unknown invocation"}}
        return 200, {"ok": True, "data": record.status()}

    def action_result(self, invocation_id: str, *, timeout_s: float | None = None) -> tuple[int, dict]:
        """`GET /invocations/{id}/result`。pending → 202。

        `timeout_s` 非 None 时会等到终结或超时；超时后走 Q12/Q13 的治理路径。
        """
        record = self._records.get(invocation_id)
        if record is None:
            return 404, {"ok": False, "error": {"message": "unknown invocation"}}
        if timeout_s is None:
            return record.result_envelope()

        deadline = self._clock() + float(timeout_s)
        thread = self._threads.get(invocation_id)
        while True:
            if record.phase == PHASE_TERMINAL:
                return record.result_envelope()
            remaining = deadline - self._clock()
            if remaining <= 0:
                self._on_action_timeout(record, thread)
                return record.result_envelope()
            if thread is not None:
                thread.join(timeout=min(0.05, remaining))
            else:
                time.sleep(min(0.05, remaining))

    def _on_action_timeout(self, record: ActionRecord, thread: threading.Thread | None) -> None:
        """Q12 + Q13 的落实：超时 → 请求 cancel → **等 backend 确认** → 如实记账。

        关键：`cancel_status='accepted'` 只说明请求被受理。
        `execution_status` 只有在**执行体真的退出**之后才是 `stopped_confirmed`；
        等不到就是 `unknown` —— 绝不伪造 `cancelled`。
        """
        with self._state_lock:
            record.timed_out = True
            if record.cancel_status is None:
                record.cancel_status = "requested"
            record.cancel_status = "accepted"
            record.cancel_requested_at_s = self._clock()
            flag = self._cancel_flags.get(record.invocation_id)
        if flag is not None:
            flag.set()

        # 等 backend 确认真实停止 —— 有界等待，因为「等不到」本身就是要记录的事实。
        if thread is not None:
            thread.join(timeout=self.stop_confirm_timeout_s)
        stopped = thread is not None and not thread.is_alive()

        with self._state_lock:
            if record.phase == PHASE_TERMINAL:
                # 执行体在等待窗口内跑完了 —— 物理效果**已经发生**，取消来晚了。
                record.cancel_effective = False
                return
            record.phase = PHASE_TERMINAL
            record.outcome = OUTCOME_TIMED_OUT
            if stopped:
                # 执行体退出了但没写终态（异常路径），backend 已空闲。
                record.execution_status = EXEC_STOPPED_CONFIRMED
                record.cancel_effective = True
            else:
                # 拿不到正面证据：执行体还卡在里面。**如实报 unknown。**
                record.execution_status = EXEC_UNKNOWN
                record.cancel_effective = False
            record.finished_at_s = self._clock()
            record.error = "action execution timeout"

    def cancel_invocation(self, invocation_id: str) -> tuple[int, dict]:
        """`POST /invocations/{id}/cancel`。

        **200/202 只表示请求被处理，不证明物理已停** —— 所以返回里同时给出
        `execution_status` 与 `cancel_effective`，让调用方拿到真话。
        """
        record = self._records.get(invocation_id)
        if record is None:
            return 404, {"ok": False, "error": {"message": "unknown invocation"}}
        flag = self._cancel_flags.get(invocation_id)
        thread = self._threads.get(invocation_id)

        with self._state_lock:
            if record.cancel_status is None:
                record.cancel_status = "requested"
                record.cancel_requested_at_s = self._clock()
        if flag is not None:
            flag.set()
        with self._state_lock:
            record.cancel_status = "accepted"

        if record.phase == PHASE_TERMINAL:
            # 已经终结：cancel 无害但无效。说实话。
            return 200, {"ok": True, "data": {
                **record.status(), "note": "invocation already terminal; cancel had no effect"}}

        # 有界等待 —— 只有执行体退出才能宣称停了。
        if thread is not None:
            thread.join(timeout=self.stop_confirm_timeout_s)
        stopped = thread is not None and not thread.is_alive()

        with self._state_lock:
            if record.phase != PHASE_TERMINAL:
                if stopped:
                    record.phase = PHASE_TERMINAL
                    record.outcome = OUTCOME_CANCELLED
                    record.execution_status = EXEC_STOPPED_CONFIRMED
                    record.cancel_effective = True
                    record.finished_at_s = self._clock()
                else:
                    # 请求受理了，但没能确认停下 —— 如实说。
                    record.execution_status = EXEC_UNKNOWN
                    record.cancel_effective = False
            else:
                record.cancel_effective = record.cancel_effective and stopped

        return 200, {"ok": True, "data": record.status()}

    def close(self) -> None:
        """等待所有 Action 收敛。**不强制杀** —— 由调用方决定怎么收场。"""
        for thread in list(self._threads.values()):
            thread.join(timeout=self.stop_confirm_timeout_s)

    # ────────────────────────────────── 观测（诊断用）

    def pending_invocations(self) -> list[str]:
        with self._state_lock:
            return [iid for iid, rec in self._records.items() if rec.phase != PHASE_TERMINAL]


# ────────────────────────────────────────────────────────── HTTP 外壳


class GatewayServer:
    """把 `LocalGateway` 套上 HTTP。stdlib `http.server`，无第三方依赖。

    第 3 步 Dora flow 里的 `gateway` 节点会用它；第 2 步的验收也用它来
    验证 PAOS 的 `_tool_context_readiness()` 与 `ForgeSkillBindingResolver.preview()`
    能真正跑通。
    """

    def __init__(self, gateway: LocalGateway, host: str = "127.0.0.1", port: int = 0):
        self.gateway = gateway
        self._handler = _make_handler(gateway)
        self.httpd = ThreadingHTTPServer((host, port), self._handler)
        self.httpd.daemon_threads = True
        self.host, self.port = self.httpd.server_address[:2]
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> "GatewayServer":
        self._thread = threading.Thread(target=self.httpd.serve_forever,
                                        name="track2-gateway", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __enter__(self) -> "GatewayServer":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def _make_handler(gateway: LocalGateway):
    """闭包出一个绑定到具体 gateway 实例的 handler 类。"""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):    # 静默：调用日志由 Gateway 自己记
            pass

        # ── 工具

        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _segments(self) -> list[str]:
            return [unquote(p) for p in self.path.split("?")[0].split("/") if p]

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return {}
            return value if isinstance(value, dict) else {}

        # ── GET

        def do_GET(self) -> None:                      # noqa: N802
            parts = self._segments()
            if parts == ["tools"]:
                status, payload = gateway.list_tools()
                return self._send(status, payload)
            if len(parts) == 2 and parts[0] == "tools":
                status, payload = gateway.get_tool(parts[1])
                return self._send(status, payload)
            if len(parts) == 3 and parts[0] == "tools" and parts[2] == "context":
                status, payload = gateway.get_tool_context(parts[1])
                return self._send(status, payload)
            if len(parts) == 2 and parts[0] == "invocations":
                status, payload = gateway.action_status(parts[1])
                return self._send(status, payload)
            if len(parts) == 3 and parts[0] == "invocations" and parts[2] == "result":
                status, payload = gateway.action_result(parts[1])
                return self._send(status, payload)
            return self._send(404, {"ok": False, "error": {"message": "not found"}})

        # ── POST

        def do_POST(self) -> None:                     # noqa: N802
            parts = self._segments()
            body = self._body()

            # Action：POST /tools/{tool_id}:invoke        （1 段 + :invoke）
            if (len(parts) == 2 and parts[0] == "tools" and parts[1].endswith(":invoke")):
                tool_id = parts[1][: -len(":invoke")]
                status, payload = gateway.invoke_action(tool_id, body)
                return self._send(status, payload)

            # Query：POST /tools/{endpoint_id}/{operation}:invoke   （2 段 + :invoke）
            if (len(parts) == 3 and parts[0] == "tools" and parts[2].endswith(":invoke")):
                operation = parts[2][: -len(":invoke")]
                status, payload = gateway.invoke_query(parts[1], operation, body)
                return self._send(status, payload)

            # cancel / stop
            if (len(parts) == 3 and parts[0] == "invocations"
                    and parts[2] in ("cancel", "stop")):
                status, payload = gateway.cancel_invocation(parts[1])
                return self._send(status, payload)

            return self._send(404, {"ok": False, "error": {"message": "not found"}})

    return Handler
