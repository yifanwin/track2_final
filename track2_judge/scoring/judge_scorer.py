"""终态连续评分器（纯函数）—— 新公式的实现。

见 `report/plan/Track2_终态连续评分_改造方案.md` §1。

```
S_case = max(0, S_target − P_disturb)                    （0–100）
S_target  = Σ_i w_i · T_i                                （任务完成度，0 或 [30,100]）
P_disturb = 70 · clip(Σ_j v_j · D_j, 0, 1)               （环境扰动处罚，[0,70]）
```

* **`P1` 是硬门**：不通过 → 目标项 0；通过 → 30 起。
* 位姿越准越接近 100（`target_base + target_span`）。
* **非目标只扣分、不给分**（这是 2026-10-19 的口径变更）：

  | 情况 | 目标分 | 扰动扣分 | 最终 |
  |---|---:|---:|---:|
  | 什么都不做 | 0 | 0 | **0** |
  | 只保护非目标物体 | 0 | 0 | **0** |
  | 目标刚进区域 | 30 | 0 | **30** |
  | 目标完美 | 100 | 0 | **100** |
  | 目标完美 + 严重扰动环境 | 100 | −70 | **30** |
  | 只做到基础 place + 严重破坏 | 30 | −70 | **0** |

  **语义**：先看任务完成得怎么样，再看为了完成任务造成了多少不必要的破坏。
  「保护环境」是**约束**（不做不奖励、做了不惩罚），不是值得单独奖励的目标 ——
  这解决了旧口径「没碰任何东西就白拿 30 分」的问题。

**为什么不改冻结件**：`mesa_eval.py` 是组委会下发的冻结 V3（`scorer_content_modified: false`）。
本模块**只吃它已经算出来的几何量**（`position_error_cm` / `rotation_error_deg` /
非目标逐项位移），不复制几何代码，也不碰冻结件。
"""
from __future__ import annotations

import math

__all__ = ["score_case", "score_target", "penalty_disturb", "clip", "constants_for_tier"]


def clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """把 `value` 夹到 `[low, high]`。"""
    if value < low:
        return low
    if value > high:
        return high
    return float(value)


# ── 目标物体（0–100）

def _q_pose(d_pos_cm: float, d_rot_deg: float, tau_pos_cm: float, tau_rot_deg: float,
            alpha: float = 0.5) -> float:
    """位姿质量 `Q ∈ [0,1]`：目标位姿误差的相对占比，越小越接近 1。

    `Q = 1 − [α·clip(d_pos/τ_pos) + (1−α)·clip(d_rot/τ_rot)]`。
    `α` 是位置/姿态的相对权重（默认各半）。
    """
    if tau_pos_cm <= 0 or tau_rot_deg <= 0:
        raise ValueError("tau constants must be positive")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    return 1.0 - (alpha * clip(d_pos_cm / tau_pos_cm)
                  + (1.0 - alpha) * clip(d_rot_deg / tau_rot_deg))


def score_target(*, p1_passed: bool, weight: float, d_pos_cm: float, d_rot_deg: float,
                 tau_pos_cm: float, tau_rot_deg: float, alpha: float = 0.5,
                 target_base: float = 30.0, target_span: float = 70.0) -> float:
    """单个目标物体的得分。

    `p1_passed` 为假 ⇒ 0（硬门）；为真 ⇒ `target_base + target_span · Q`。
    """
    if not p1_passed:
        return 0.0
    q = _q_pose(d_pos_cm, d_rot_deg, tau_pos_cm, tau_rot_deg, alpha)
    return weight * (target_base + target_span * q)


# ── 非目标物体（只扣分，0–70）

def _d_disturb(d_pos_cm: float, d_rot_deg: float,
               tau_keep_pos_cm: float, tau_keep_rot_deg: float,
               tau_safe: float, tau_fail: float, beta: float = 0.5) -> float:
    """单个非目标物体的扰动程度 `D ∈ [0,1]`（三段 ramp）。

    ```
    Δ = β·clip(d_pos/τ_keep_pos) + (1−β)·clip(d_rot/τ_keep_rot)      ∈ [0,1]
    D = 0                  Δ ≤ τ_safe
        (Δ−τ_safe)/(τ_fail−τ_safe)      τ_safe < Δ < τ_fail
        1                  Δ ≥ τ_fail
    ```
    """
    if tau_keep_pos_cm <= 0 or tau_keep_rot_deg <= 0:
        raise ValueError("keep tau constants must be positive")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0,1]")
    if tau_fail <= tau_safe:
        raise ValueError("tau_fail must be greater than tau_safe")
    delta = (beta * clip(d_pos_cm / tau_keep_pos_cm)
             + (1.0 - beta) * clip(d_rot_deg / tau_keep_rot_deg))
    if delta <= tau_safe:
        return 0.0
    if delta >= tau_fail:
        return 1.0
    return (delta - tau_safe) / (tau_fail - tau_safe)


def penalty_disturb(*, displacements, constants: dict) -> float:
    """环境扰动处罚 `P_disturb ∈ [0,70]`。

    `displacements` 是 `[(d_pos_cm, d_rot_deg), ...]`（每个非目标一项）。
    `v_j = 1/M`（均分）；**`M = 0` ⇒ 处罚 0** —— 无物可碰 = 没有扰动。
    """
    count = len(displacements)
    if count == 0:
        return 0.0
    total = sum(_d_disturb(p, r,
                           float(constants["tau_keep_pos_cm"]),
                           float(constants["tau_keep_rot_deg"]),
                           float(constants["tau_safe"]),
                           float(constants["tau_fail"]),
                           float(constants.get("beta", 0.5)))
                for p, r in displacements)
    mean = clip(total / count)
    return float(constants.get("disturb_total", 70.0)) * mean


def score_case(*, targets, displacements, constants: dict) -> dict:
    """一个 case 的总分（**0–100**）。

    * `targets`: `[{"p1_passed", "weight", "d_pos_cm", "d_rot_deg"}, ...]`
      （50 例里恒 1 项，`weight=1`）
    * `displacements`: `[(d_pos_cm, d_rot_deg), ...]`（非目标，可空）
    * `constants`: 由 `constants_for_tier()` 给

    返回 `{"score_target","penalty_disturb","score_100"}`；
    **`score_100 = max(0, score_target − penalty_disturb)`**。
    """
    c = dict(constants)
    t = sum(score_target(
        p1_passed=bool(ti["p1_passed"]),
        weight=float(ti.get("weight", 1.0)),
        d_pos_cm=float(ti["d_pos_cm"]), d_rot_deg=float(ti["d_rot_deg"]),
        tau_pos_cm=float(c["tau_pos_cm"]), tau_rot_deg=float(c["tau_rot_deg"]),
        alpha=float(c.get("alpha", 0.5)),
        target_base=float(c.get("target_base", 30.0)),
        target_span=float(c.get("target_span", 70.0)),
    ) for ti in targets)
    p = penalty_disturb(displacements=displacements, constants=c)
    return {"score_target": t, "penalty_disturb": p,
            "score_100": max(0.0, t - p)}


def constants_for_tier(tier: str) -> dict:
    """按 tier 给出常数（§1.5 表 + 2026-10-19 的扰动归一化）。

    * `τ_pos = 2×eps_pos`、`τ_rot = 2×eps_rot`（目标容差的 2 倍）
    * `τ_keep_pos/rot = τ_pos/rot`（非目标的「位姿变化」用同一把尺子）
    * `τ_safe` / `τ_fail`：**归一化 Δ 空间**里的安全/超限阈值。
      取法保证**物理阈值不变**：`d_pos = delta_max, d_rot = 0`（β=0.5）
      恰好落在 `τ_safe`；`d_pos = 3×delta_max` 落在 `τ_fail`。
      即 `τ_safe = 0.5·delta_max/τ_pos`、`τ_fail = 0.5·3·delta_max/τ_pos`。

    ⚠️ 这些常数**不能塞进 `contract/scoring_config.json`** ——
    那份文件被 `contract/profile.py:9` 校验为「逐字等于 `v3.default_config()`」，
    而 `v3.validate_config` 用白名单，多一个键就报错。所以它们独立存在。
    """
    from track2_judge.scoring.scorer_bridge import v3
    if tier not in v3.TIERS:
        raise ValueError(f"unknown tier: {tier}")
    # ⚠️ TIERS 的顺序是 `(eps_pos_sim, eps_pos_real, eps_rot_deg, delta_max_cm)`
    #    —— 与 `v3.default_config()` 的 `sim, real, rotation, delta = TIERS[tier]` 一致。
    eps_pos, _eps_pos_real, eps_rot, delta_max = v3.TIERS[tier]
    beta = 0.5
    tau_pos = 2.0 * float(eps_pos)
    tau_rot = 2.0 * float(eps_rot)
    d_safe_cm = float(delta_max)
    d_fail_cm = 3.0 * float(delta_max)
    return {
        "tau_pos_cm": tau_pos,
        "tau_rot_deg": tau_rot,
        "tau_keep_pos_cm": tau_pos,
        "tau_keep_rot_deg": tau_rot,
        "tau_safe": beta * d_safe_cm / tau_pos,
        "tau_fail": beta * d_fail_cm / tau_pos,
        "alpha": 0.5,
        "beta": beta,
        "target_base": 30.0,
        "target_span": 70.0,
        "disturb_total": 70.0,
    }
