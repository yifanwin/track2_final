"""选手侧 `tools` 对象 —— 对 PAOS 是 Action，对选手是「调用即返回」。

## 设计要点（§3.3.2）

| 视角 | 看到什么 |
|---|---|
| **PAOS** | 真正的 Action：202 + `invocation_id` + `attempt_id`，可轮询、可取消 |
| **选手** | `tools.joint_control(q, g)` 一次调用拿到结果 —— **形状与现状完全一致** |

异步生命周期封装在这个类内部：

```text
invoke → 202 + invocation_id
   ↓
轮询 GET /invocations/{id} 直到 phase 终结
   ↓
GET /invocations/{id}/result
   ↓
返回结果
```

## 两条通路的同一份实现（§4.2）

- **沙箱内（第 2 步现状）**：调用被序列化成 JSON-lines 交给评测器侧的 `SandboxBridge`。
  沙箱这时候是 `--net` 隔离的，**连不上 127.0.0.1**，所以还不能直连 Gateway。
- **去掉网络隔离后（第 4 步）**：直接 HTTP 打 Gateway。

两种通路共用这套方法名与超时/取消语义 —— 换的只是 `_transport`。

## 取消路径的硬约束（Q13）

`cancel_status='accepted'` **不等于物理已停**。所以超时后：

1. 发 cancel
2. 读 Gateway 回给的真实 `execution_status`
3. `stopped_confirmed` 才报 `cancelled`
4. 否则**抛异常**，把它归到 `SKILL_TIMEOUT` —— 因为物理事实不清

评审链路（`runner/candidate_runner.py`）就是这么区分「没跑完」和「跑到一半状态不明」的。
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Protocol

from track2_judge.toolapi import schema as schema_mod
from track2_judge.toolapi.gateway import (
    EXEC_STOPPED_CONFIRMED, OUTCOME_SUCCEEDED, PHASE_TERMINAL,
)

__all__ = ["ToolClient", "HttpTransport", "InProcessTransport", "SandboxBridge",
           "EpisodeTools", "build_episode_tools",
           "ToolCallError", "ActionTimeout", "ActionStoppedUnknown"]


class ToolCallError(RuntimeError):
    """工具调用失败（业务层面）：schema 不过、未声明、并发已满、Action 执行失败。"""


class ActionTimeout(ToolCallError):
    """Action 超时，且**已确认物理已停**（`stopped_confirmed`）。"""


class ActionStoppedUnknown(ActionTimeout):
    """Action 超时后**无法确认物理已停**（Q13）。

    刻意与普通超时分开：普通超时是「没跑完」，
    这个是「跑到一半、不知道物理上停没停」—— 后者更严重，
    调用方必须把它归到超时并按超时处置，**不能重试**。
    """


class Transport(Protocol):
    """通路的共同接口。返回值统一是 `(status_code, body)`。"""

    def invoke_query(self, endpoint_id: str, operation: str, body: dict) -> tuple[int, dict]: ...
    def invoke_action(self, tool_id: str, body: dict) -> tuple[int, dict]: ...
    def action_status(self, invocation_id: str) -> tuple[int, dict]: ...
    def action_result(self, invocation_id: str) -> tuple[int, dict]: ...
    def cancel_invocation(self, invocation_id: str) -> tuple[int, dict]: ...


# ─────────────────────────────────────────────────────── 通路 1：进程内

class InProcessTransport:
    """直接调 `LocalGateway` 的方法。**不碰网络** —— 编排链路用。

    沙箱（`--net` 隔离）里的选手用不了这个；它属于评测器进程。
    """

    def __init__(self, gateway):
        self.gateway = gateway

    def invoke_query(self, endpoint_id, operation, body):
        return self.gateway.invoke_query(endpoint_id, operation, body)

    def invoke_action(self, tool_id, body):
        return self.gateway.invoke_action(tool_id, body)

    def action_status(self, invocation_id):
        return self.gateway.action_status(invocation_id)

    def action_result(self, invocation_id):
        return self.gateway.action_result(invocation_id)

    def cancel_invocation(self, invocation_id):
        return self.gateway.cancel_invocation(invocation_id)


# ─────────────────────────────────────────────────────── 通路 2：HTTP

class HttpTransport:
    """HTTP 打 Gateway。**去掉沙箱网络隔离后**（第 4 步）用这条。"""

    def __init__(self, base_url: str, *, timeout_s: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _request(self, path: str, method: str, body: dict | None) -> tuple[int, dict]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path, data=data, method=method,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace")
            try:
                return exc.code, json.loads(raw)
            except json.JSONDecodeError:
                return exc.code, {"ok": False, "error": {"message": raw[:200]}}

    def invoke_query(self, endpoint_id, operation, body):
        return self._request(f"/tools/{endpoint_id}/{operation}:invoke", "POST", body)

    def invoke_action(self, tool_id, body):
        return self._request(f"/tools/{tool_id}:invoke", "POST", body)

    def action_status(self, invocation_id):
        return self._request(f"/invocations/{invocation_id}", "GET", None)

    def action_result(self, invocation_id):
        return self._request(f"/invocations/{invocation_id}/result", "GET", None)

    def cancel_invocation(self, invocation_id):
        return self._request(f"/invocations/{invocation_id}/cancel", "POST", {})


# ─────────────────────────────────────────────────────── 对外客户端

class ToolClient:
    """选手 `run(observation, tools)` 拿到的 `tools`。

    **方法名与现状完全一致** —— 选手代码不用改（除了新契约下关节控制的语义）。
    """

    def __init__(
        self,
        transport: Transport,
        *,
        action_timeout_s: float | None = None,
        poll_interval_s: float = 0.005,
        max_action_steps: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self._transport = transport
        self.action_timeout_s = (
            schema_mod.ACTION_TIMEOUT_S if action_timeout_s is None else float(action_timeout_s)
        )
        self.poll_interval_s = float(poll_interval_s)
        # 每 episode 的 Action 次数预算。None = 不限制（由 Gateway/runner 管）。
        self.max_action_steps = max_action_steps
        self._clock = clock
        self._sleep = sleeper
        self.action_steps = 0

    # ── Query

    def observe(self, **kwargs: Any) -> dict:
        """读当前观测。**同步**返回（PAOS 的 Query）。

        关键字参数直接进 `arguments`（例如 `include_rgb=False`）。
        """
        status, body = self._transport.invoke_query(
            "track2.env", "observe", {"arguments": dict(kwargs)})
        if status != 200 or not body.get("ok"):
            raise ToolCallError(_error_text(body, status))
        return body["data"]

    # ── Action（对外仍是「调用即返回」）

    def joint_control(self, joint_position, gripper_opening_m, **kwargs: Any) -> dict:
        """下发一步关节目标 + 夹爪开度。

        内部走完整 Action 生命周期；对选手就是一次调用。
        """
        arguments = {"joint_position": list(joint_position),
                     "gripper_opening_m": float(gripper_opening_m)}
        arguments.update(kwargs)
        return self._run_action("track2.robot.joint_control", arguments)

    def gripper_control(self, gripper_opening_m, **kwargs: Any) -> dict:
        """只动夹爪，关节目标保持在上一次。"""
        arguments = {"gripper_opening_m": float(gripper_opening_m)}
        arguments.update(kwargs)
        return self._run_action("track2.robot.gripper_control", arguments)

    # ── 兼容别名（现状选手代码可继续跑）

    def step(self, joint_position, gripper_opening_m, **kwargs: Any) -> dict:
        return self.joint_control(joint_position, gripper_opening_m, **kwargs)

    def piper_joint_control(self, joint_position, gripper_opening_m, **kwargs: Any) -> dict:
        return self.joint_control(joint_position, gripper_opening_m, **kwargs)

    def piper_gripper_control(self, gripper_opening_m, **kwargs: Any) -> dict:
        return self.gripper_control(gripper_opening_m, **kwargs)

    # ── 通用 call（旧契约的逃生口，保持存在但受声明门禁约束）

    def call(self, tool_id: str, arguments: dict | None = None) -> Any:
        """按 tool_id 直接调用。

        只接受**已声明**的 tool —— 未声明的（例如 `motion.*`、`tabletop.*`）
        会在 Gateway 侧被 403 拒掉。这一层不做额外的名字白名单。
        """
        if not schema_mod.is_declared(tool_id):
            raise ToolCallError(f"tool {tool_id!r} is not declared for this profile")
        spec = schema_mod.get_spec(tool_id)
        if spec["semantics"] == "query":
            endpoint_id, operation = spec["endpoint_id"], spec["operation"]
            status, body = self._transport.invoke_query(
                endpoint_id, operation, {"arguments": arguments or {}})
            if status != 200 or not body.get("ok"):
                raise ToolCallError(_error_text(body, status))
            return body["data"]
        return self._run_action(tool_id, arguments or {})

    # ── Action 生命周期（内部）

    def _run_action(self, tool_id: str, arguments: dict) -> dict:
        if self.max_action_steps is not None:
            if self.action_steps >= self.max_action_steps:
                raise ToolCallError("episode action budget exhausted")

        status, body = self._transport.invoke_action(
            tool_id, {"arguments": arguments})
        if status != 202 or not body.get("ok"):
            # 429（并发已满）/ 400（schema）/ 403（未声明）都在这里。
            raise ToolCallError(_error_text(body, status))
        invocation_id = body["data"]["invocation_id"]
        deadline = self._clock() + self.action_timeout_s

        while True:
            if self._clock() >= deadline:
                # Q12 + Q13：超时 → cancel → **等 backend 确认** → 如实分类。
                raise self._timeout(invocation_id)
            status, body = self._transport.action_status(invocation_id)
            if status != 200:
                raise ToolCallError(_error_text(body, status))
            data = body["data"]
            if data.get("phase") == PHASE_TERMINAL:
                break
            self._sleep(self.poll_interval_s)

        if data.get("outcome") != OUTCOME_SUCCEEDED:
            raise ToolCallError(
                f"{tool_id} {data.get('outcome')}: {data.get('error') or 'no detail'}")

        status, body = self._transport.action_result(invocation_id)
        if status != 200 or not body.get("ok"):
            raise ToolCallError(_error_text(body, status))
        self.action_steps += 1
        return body["data"].get("result")

    def _timeout(self, invocation_id: str) -> Exception:
        """超时处理。**只有拿到正面证据才报 cancelled。**"""
        try:
            status, body = self._transport.cancel_invocation(invocation_id)
        except Exception:
            return ActionStoppedUnknown(
                f"action {invocation_id} timed out; cancel request failed, "
                "physical stop unconfirmed")
        data = (body or {}).get("data") or {}
        execution_status = data.get("execution_status")
        if execution_status == EXEC_STOPPED_CONFIRMED:
            return ActionTimeout(
                f"action {invocation_id} timed out and was confirmed stopped")
        # Q13 的核心：cancel accepted ≠ 已停。拿不到证据就如实说不知道。
        return ActionStoppedUnknown(
            f"action {invocation_id} timed out; cancel_status="
            f"{data.get('cancel_status')!r} but physical stop is unconfirmed "
            f"(execution_status={execution_status!r})")


def _error_text(body: dict, status: int) -> str:
    """从 Gateway 应答里取出可读错误。**不泄漏裁判侧路径。**"""
    error = body.get("error")
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    if isinstance(error, str):
        return error
    return f"tool call failed with HTTP {status}"


# ─────────────────────────────────────────── 沙箱侧的调用桥

# 沙箱内选手用 JSON-lines 发过来的方法名 → `ToolClient` 的方法。
# 这是**方法名兼容层**：旧选手代码里的 `tools.step(...)` 仍然能用。
_METHOD_ALIASES = {
    "observe": "observe",
    "step": "joint_control",
    "joint_control": "joint_control",
    "piper_joint_control": "joint_control",
    "gripper_control": "gripper_control",
    "piper_gripper_control": "gripper_control",
}


class SandboxBridge:
    """把沙箱内选手的 JSON-lines 调用，接到真实的 `ToolClient` 上。

    `PackageExecutor.run(..., dispatch)` 期望的 `dispatch(method, payload)` 就是这里的
    `__call__`。原来它直接打 adapter（评测器内部），现在改为**经 Gateway**：

    ```text
    沙箱内 tools.joint_control(q, g)
        → JSON-lines  {"type":"tool_call","tool":"joint_control","arguments":{...}}
        → SandboxBridge.__call__("joint_control", {...})
        → ToolClient.joint_control(q, g)
        → Gateway /tools/{id}:invoke → 202 → 轮询 → result
        → 物理
    ```

    **为什么还需要这一层**：第 2 步沙箱仍是 `--net` 隔离的，连不上 127.0.0.1。
    所以选手不能直连 HTTP Gateway，必须借评测器进程的手。第 4 步去掉网络隔离后，
    这一层可以整体换成沙箱内直连 `HttpTransport` —— 而 `ToolClient` 的语义不变。
    """

    def __init__(self, client: ToolClient):
        self.client = client
        # 原 runner 的白名单与 profile 字符串门禁已在第 1.7 步删除（Q16 单一新轨）。
        # 现在的准入完全由 Gateway 的 `required_tools` 声明门禁裁决：
        # 未声明的 tool 在 `ToolClient.call` 或 Gateway 侧被拒。
        self.calls: list[dict[str, Any]] = []

    def __call__(self, method: str, payload: dict | None = None) -> Any:
        target = _METHOD_ALIASES.get(method)
        if target is None:
            # 走通用 call 通路 —— 由声明门禁决定能不能过。
            # 这样 `tools.call('track2.env.observe', {})` 这类写法仍然受支持。
            return self.client.call(method, payload or {})

        payload = payload or {}
        if target == "observe":
            extra = {k: v for k, v in payload.items()}
            result = self.client.observe(**extra)
        elif target == "joint_control":
            q = payload.get("joint_position")
            g = payload.get("gripper_opening_m")
            if q is None or g is None:
                raise ToolCallError(
                    "joint_control requires joint_position and gripper_opening_m")
            result = self.client.joint_control(q, g)
        else:                                  # gripper_control
            g = payload.get("gripper_opening_m")
            if g is None:
                raise ToolCallError("gripper_control requires gripper_opening_m")
            result = self.client.gripper_control(g)

        # 账本：给 runner 写 trace 用（原 `calls.append` 的等价物）。
        self.calls.append({"method": target, "requested_as": method,
                           "payload": dict(payload)})
        return result

# ─────────────────────────────────────────── 组装一个 episode 的工具面

class EpisodeTools:
    """一个 episode 的工具面。Runner 用它取代原来的 `adapter.dispatch_public`。

    `dispatch(method, payload)` 的签名与第 1 步的 `dispatch_public` 完全一致，
    所以编排代码的形状没变 —— 变的只是底下**经过 Gateway**。
    """

    def __init__(self, gateway, client, *, sandboxed: bool, max_steps: int | None = None,
                 clock=None):
        self.gateway = gateway
        self.client = client
        self.sandboxed = sandboxed
        self.max_steps = max_steps
        # 传给沙箱内执行体的方法名 —— 与 `_METHOD_ALIASES` 同一套。
        self.sandbox = SandboxBridge(client)
        self._clock = clock or time.monotonic

    # 编排侧入口（与第 1 步的 dispatch_public 同形）
    def dispatch(self, method: str, payload: dict | None = None):
        return self.sandbox(method, payload)

    @property
    def calls(self):
        return self.sandbox.calls

    def close(self):
        self.gateway.close()


# ── 两个方向的统一入口（§4.2）

_HTTP_BRIDGE_ENV = "TRACK2_TOOL_GATEWAY_URL"


def build_episode_tools(adapter, *, sandboxed: bool, action_timeout_s=None,
                        max_action_steps=None, adapter_clock=None, gateway=None):
    """按**当前**通路搭一个 episode 的工具面。

    两条通路共用 `ToolClient` 与同一套语义，换的只是 `transport`：

    | 通路 | 什么时候用 | 为什么 |
    |---|---|---|
    | 进程内（`InProcessTransport`） | 第 2 步的现状 | 沙箱是 `--net` 隔离的，**连不上 127.0.0.1** |
    | HTTP（`HttpTransport`） | 第 3/4 步（Dora gateway 节点，去掉网络隔离后） | 选手直连 Gateway，与 PAOS 完全同构 |

    `adapter_clock` 用来把「仿真 + 等待 Action 完成」的时间算进 episode 预算
    （`runner.tool_service_s` 在此之前就累计，所以这里传同一个 `time.monotonic`）。
    """
    from track2_judge.toolapi.dispatch import GatewayExecutor
    from track2_judge.toolapi.gateway import LocalGateway

    executor = GatewayExecutor(adapter)
    gw = gateway or LocalGateway(executor)
    transport = InProcessTransport(gw)
    client = ToolClient(transport, action_timeout_s=action_timeout_s,
                        max_action_steps=max_action_steps,
                        clock=adapter_clock or time.monotonic)
    return EpisodeTools(gw, client, sandboxed=sandboxed, max_steps=max_action_steps,
                        clock=adapter_clock)
