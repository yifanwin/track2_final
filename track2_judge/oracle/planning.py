"""裁判侧 Oracle 的 **GT 规划**（第 6 步）。

## 它取代了什么

第 6 步之前，Oracle 是这样绕开唯一入口的
（`experiments/.../oracle_driver.py`）：

```python
class PrivateOracleInput(ObservationProvider):
    def capture_initial(self, env):
        self.plan = make_plan(env, task, target, model)     # ← 用私有 env 算 GT
    def observation(self):
        # ← 把 plan 塞进 text_instruction，直接交给沙箱里的 oracle skill
        return {'task_inputs': {'text_instruction': json.dumps(executor)}}}
```

问题有两个：

1. **绕过了 `release_entry.py`** —— 它的证据只是 harness 级
2. **它自己造了一个 `env`**，不是入口造的那个 —— 所以「Oracle 成功但入口会拒绝
   该 case」这种矛盾出现时看不出来（case38 正是入口会拒的 33 个之一）

## 现在的形态

GT 规划由**入口**在正确的时机做（拿到的是**入口自己造的那个 env**），
结果作为一个**显式字段**进公开输入：

```text
release_entry.py --oracle-plan <file>
  → select_case.run(..., oracle_plan=...)
    → 公开输入里多一个 `oracle_plan` 字段（`ObservationProvider` 白名单放行）
      → oracle skill 从 `observation['task_inputs']['oracle_plan']` 读
```

**为什么不做成 `track2.admin.oracle_plan` 那个 Tool**（计划原稿的设想）：
选手 manifest 的 `tool_profile` 只能声明**官方三个** Tool，`track2.admin.*`
会被第 4 步实现的准入检查拒掉。而 Oracle **既是选手又是裁判** ——
这双重身份没法用单一 allowlist 表达。所以走「入口的显式参数」。

## 边界

* 这个模块**只在裁判侧**跑，从不进选手的沙箱（沙箱里只有那把 `actions`）
* `oracle_plan` 这个公开字段**只在 `oracle_plan=` 显式传参时**才出现 ——
  普通评测的选手包拿不到它（`select_case.run()` 不传就没有）
"""
from __future__ import annotations

import json
from typing import Any

from track2_judge.scenario.kinematics import Kinematics
from track2_judge.scenario.prototype_env import prepare_model

__all__ = ["OraclePlanningError", "build_oracle_plan", "oracle_executor_payload",
           "model"]

#: 规划器用的运动学模型。**按需构造**（解析 URDF 有点慢）。
_MODEL: Kinematics | None = None


class OraclePlanningError(RuntimeError):
    """GT 规划失败。**不降级、不伪造** —— 规划不出来就是 Oracle 在这例上做不到。"""


def model() -> Kinematics:
    global _MODEL
    if _MODEL is None:
        _MODEL = Kinematics(prepare_model())
    return _MODEL


def build_oracle_plan(env, task: dict, target: str) -> dict:
    """用**入口造的那个 env** 做 GT 规划，返回 plan（含 `actions` / `stages`）。

    ⚠️ 它读 env 的物理状态（这是 GT 的来源），但**不写** env ——
    规划是只读的，真正的执行由沙箱里的 oracle skill 通过 Tool 做。
    """
    from track2_judge.oracle.oracle_planner import make_plan

    plan = make_plan(env, task, target, model())
    if not isinstance(plan, dict) or not isinstance(plan.get("actions"), list):
        raise OraclePlanningError("planner returned no usable action list")
    if not plan["actions"]:
        raise OraclePlanningError(
            f"planner produced zero actions (failure_reason="
            f"{plan.get('failure_reason')!r})")
    return plan


def oracle_executor_payload(plan: dict) -> dict:
    """把 plan 收窄成**跨进沙箱的那一份**。

    只带 `actions` / `stages` / `failure_reason` —— 与老实现
    （`PrivateOracleInput.observation()`）刻意保持一致的形状，
    这样 oracle skill 的执行路径不用动。

    **不带** `diagnostic`（里面有碰撞半径、AABB 距离这类私有诊断量）——
    沙箱里的 skill 不需要它们，给了就是白送信息。
    """
    return {
        "schema": "judge-private-oracle-executor-v1",
        "oracle_only": True,
        "actions": plan["actions"],
        "stages": plan.get("stages", []),
        "failure_reason": plan.get("failure_reason", "TRAJECTORY_COMPLETED"),
    }


def find_target(task: dict) -> str:
    """要搬的那个物体（P2 谓词的对象）。"""
    for pred in task.get("predicates", []):
        if pred.get("type") == "P2":
            return pred["object"]
    raise OraclePlanningError("task has no P2 target object")


def serializable(value: Any) -> Any:
    """把 numpy 值转成纯 JSON（规划结果里混着 numpy 标量）。"""
    return json.loads(json.dumps(value, default=_plain))


def _plain(value: Any) -> Any:
    try:
        return value.tolist()
    except AttributeError:
        raise TypeError(f"not JSON serializable: {type(value).__name__}") from None
