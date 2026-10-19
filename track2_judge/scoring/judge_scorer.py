"""终态连续评分器（纯函数）—— 新公式的实现。

见 `report/plan/Track2_终态连续评分_改造方案.md` §1。

```
Score_case = Score_target + Score_preserve         （上限 100）
Score_target  = Σ_i w_i · T_i                       （目标物体，0 或 [30,70]）
Score_preserve = 30 · Σ_j v_j · Q_keep,j            （非目标保持，[0,30]）
```

* **`P1` 是硬门**：不通过 → 目标项 0；通过 → 30 起（`target_base`）。
* 位姿越准越接近 70（`target_base + target_span`）。
* 非目标位移用三段 ramp（安全区不扣 / 中间线性 / 超限归零）。

**为什么不改冻结件**：`mesa_eval.py` 是组委会下发的冻结 V3（`scorer_content_modified: false`）。
本模块**只吃它已经算出来的几何量**（`position_error_cm` / `rotation_error_deg` /
`displacement_cm`），不复制几何代码，也不碰冻结件。
"""
from __future__ import annotations

import math

__all__ = ["score_case", "score_target", "score_preserve", "clip"]


def clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """把 `value` 夹到 `[low, high]`。"""
    if value < low:
        return low
    if value > high:
        return high
    return float(value)


def _q_pose(d_pos_cm: float, d_rot_deg: float, tau_pos_cm: float, tau_rot_deg: float) -> float:
    """位姿质量 `Q ∈ [0,1]`：目标位姿误差的相对占比，越小越接近 1。"""
    if tau_pos_cm <= 0 or tau_rot_deg <= 0:
        raise ValueError("tau constants must be positive")
    return 1.0 - 0.5 * (clip(d_pos_cm / tau_pos_cm) + clip(d_rot_deg / tau_rot_deg))


def score_target(*, p1_passed: bool, weight: float, d_pos_cm: float, d_rot_deg: float,
                 tau_pos_cm: float, tau_rot_deg: float,
                 target_base: float = 30.0, target_span: float = 40.0) -> float:
    """单个目标物体的得分。

    `p1_passed` 为假 ⇒ 0（硬门）；为真 ⇒ `target_base + target_span · Q`。
    """
    if not p1_passed:
        return 0.0
    q = _q_pose(d_pos_cm, d_rot_deg, tau_pos_cm, tau_rot_deg)
    return weight * (target_base + target_span * q)


def _q_keep(d_pos_cm: float, d_rot_deg: float,
            d_safe_cm: float, d_fail_cm: float) -> float:
    """单个非目标物体的保持质量 `Q_keep ∈ [0,1]`（三段 ramp）。

    `D = 0.5·d_pos + 0.5·d_rot`；
    `D ≤ D_safe` → 1；`D_safe < D < D_fail` → 线性；`D ≥ D_fail` → 0。
    """
    if d_fail_cm <= d_safe_cm:
        raise ValueError("d_fail must be greater than d_safe")
    d = 0.5 * d_pos_cm + 0.5 * d_rot_deg
    if d <= d_safe_cm:
        return 1.0
    if d >= d_fail_cm:
        return 0.0
    return (d_fail_cm - d) / (d_fail_cm - d_safe_cm)


def score_preserve(*, displacements, d_safe_cm: float, d_fail_cm: float,
                   preserve_total: float = 30.0) -> float:
    """非目标物体保持分。

    `displacements` 是 `[(d_pos_cm, d_rot_deg), ...]`（每个非目标一项）。
    **`M = 0`（无可移动非目标）⇒ 满分** —— 无物可碰 = 完美保持。
    `v_j = 1/M`（均分）。
    """
    count = len(displacements)
    if count == 0:
        return float(preserve_total)
    total = sum(_q_keep(p, r, d_safe_cm, d_fail_cm) for p, r in displacements)
    return float(preserve_total) * total / count


def score_case(*, targets, displacements, constants) -> dict:
    """一个 case 的总分（内部 100 分）。

    * `targets`: `[{"p1_passed", "weight", "d_pos_cm", "d_rot_deg"}, ...]`
      （50 例里恒 1 项，`weight=1`）
    * `displacements`: `[(d_pos_cm, d_rot_deg), ...]`（非目标，可空）
    * `constants`: `tau_pos_cm` / `tau_rot_deg` / `d_safe_cm` / `d_fail_cm`
      / 可选 `target_base` / `target_span` / `preserve_total`

    返回 `{"score_target","score_preserve","score_100"}`。
    """
    c = dict(constants)
    t = sum(score_target(
        p1_passed=bool(ti["p1_passed"]),
        weight=float(ti.get("weight", 1.0)),
        d_pos_cm=float(ti["d_pos_cm"]), d_rot_deg=float(ti["d_rot_deg"]),
        tau_pos_cm=float(c["tau_pos_cm"]), tau_rot_deg=float(c["tau_rot_deg"]),
        target_base=float(c.get("target_base", 30.0)),
        target_span=float(c.get("target_span", 40.0)),
    ) for ti in targets)
    p = score_preserve(displacements=displacements,
                       d_safe_cm=float(c["d_safe_cm"]), d_fail_cm=float(c["d_fail_cm"]),
                       preserve_total=float(c.get("preserve_total", 30.0)))
    return {"score_target": t, "score_preserve": p, "score_100": t + p}


def constants_for_tier(tier: str) -> dict:
    """按 tier 给出常数（§1.5 表）：`tau=2×eps`、`D_safe=delta_max`、`D_fail=3×delta_max`。

    ⚠️ 这些常数**不能塞进 `contract/scoring_config.json`** ——
    那份文件被 `contract/profile.py:9` 校验为「逐字等于 `v3.default_config()`」，
    而 `v3.validate_config` 用白名单，多一个键就报错。所以它们独立存在。
    """
    from track2_judge.scoring.scorer_bridge import v3
    if tier not in v3.TIERS:
        raise ValueError(f"unknown tier: {tier}")
    # ⚠️ TIERS 的顺序是 `(eps_pos_sim, eps_pos_real, eps_rot_deg, delta_max_cm)`
    #    —— 与 `v3.default_config()` 的 `sim, real, rotation, delta = TIERS[tier]` 一致。
    #    别把第 2 项当成 eps_rot（我第一版就这么写错了）。
    eps_pos, _eps_pos_real, eps_rot, delta_max = v3.TIERS[tier]
    return {
        "tau_pos_cm": 2.0 * float(eps_pos),
        "tau_rot_deg": 2.0 * float(eps_rot),
        "d_safe_cm": float(delta_max),
        "d_fail_cm": 3.0 * float(delta_max),
        "target_base": 30.0,
        "target_span": 40.0,
        "preserve_total": 30.0,
    }
