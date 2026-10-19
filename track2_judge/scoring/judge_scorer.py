"""终态连续评分器（纯函数）—— 新公式的实现。

见 `report/plan/Track2_终态连续评分_改造方案.md` §1。

```
S_case = max(0, S_target − P_disturb)                    （0–100）
S_target  = Σ_i w_i · T_i
  T_i = 30 + 70·Q              若 P1 通过（进有效放置区）
      = approach_max · φ       若 P1 未通过（★ 2026-10-19 接近度项，φ ∈ [0,1]）
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

__all__ = ["score_case", "score_target", "penalty_disturb", "clip", "constants_for_tier",
           "aabb_gap_m", "aabb_inside_gap_m", "approach_phi"]


def clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """把 `value` 夹到 `[low, high]`。"""
    if value < low:
        return low
    if value > high:
        return high
    return float(value)


# ── 目标物体（0–100）

def aabb_gap_m(point, region_aabb_m) -> float:
    """点（米）到轴对齐区域（米）的**轴对齐最短间距**：区内为 0，区外为欧氏距离。

    与 P1 的 `whole_aabb` / `point` 成员判定**同一套几何**，
    所以「`gap == 0`」与「P1 通过」是同一件事。
    """
    if point is None or region_aabb_m is None:
        raise ValueError("point and region_aabb_m are required")
    low, high = region_aabb_m
    d2 = 0.0
    for p, lo, hi in zip(point, low, high):
        if p < lo:
            d2 += (lo - p) ** 2
        elif p > hi:
            d2 += (p - hi) ** 2
    return math.sqrt(d2)


def aabb_inside_gap_m(object_aabb_m, region_aabb_m) -> float:
    """物体 AABB（米）**完全落入**区域所需的最小位移（米）；已落入为 0。

    与 P1 的 `whole_aabb` 成员判定同源：P1 通过 ⟺ 本函数返回 0。
    比区域还大的轴无法满足 —— 按该轴的必要位移计（max），保守。
    """
    if object_aabb_m is None or region_aabb_m is None:
        raise ValueError("object_aabb_m and region_aabb_m are required")
    (b0, b1), (lo, hi) = object_aabb_m, region_aabb_m
    d2 = 0.0
    for a0, a1, l, h in zip(b0, b1, lo, hi):
        need = 0.0
        if a0 < l:
            need = l - a0
        if a1 > h:
            need = max(need, a1 - h)
        d2 += need ** 2
    return math.sqrt(d2)


def approach_phi(d_init_m: float, d_final_m: float, ref_m: float = 0.0) -> float:
    """接近度 `φ ∈ [0,1]`：把目标物朝有效放置区推近的**相对比例**。

    `φ = clip((d_init − d_final) / denom, 0, 1)`

    * **`ref_m <= 0`（旧口径）**：`denom = d_init` —— 逐字兼容旧行为。
    * **`ref_m > 0`（★ 2026-10-19 ②）**：`denom = min(d_init, ref_m)`。

    **为什么要 `min(d_init, ref_m)`**：纯相对口径下，`d_init` 越大，同样
    一段真实进展被稀释得越狠。实测 `d_init = 21.6 cm` 时推进 **6.4 cm**
    只值 `φ = 0.30`（≈3 分），而同尺度的小案例里 6.4 cm 已接近满分 ——
    对「起始距离远」的 case 系统性偏严。加固定尺度上限后，超过 `ref_m`
    的 `d_init` 一律按 `ref_m` 归一，**真实进展的分不再随起始距离衰减**。

    红线不变（见下），且**衔接不变**：推到区域边界 `d_final = 0 ⇒ φ = 1`
    （与 `ref_m` 无关），正好接上 30 分入门档。

    * `d_init == 0`（一开局就在区内）⇒ 返回 1
    * 没动（`d_final == d_init`）⇒ 0 ⇒ **不做 = 0 分**（保住原设计红线）
    * 推得更远（`d_final > d_init`）⇒ clip 到 0（不惩罚、不给分）
    """
    if d_init_m <= 0.0:
        return 1.0
    denom = d_init_m if ref_m <= 0.0 else min(d_init_m, float(ref_m))
    if denom <= 0.0:
        return 1.0
    return clip((d_init_m - d_final_m) / denom)


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
                 target_base: float = 30.0, target_span: float = 70.0,
                 phi: float = 0.0, approach_max: float = 10.0) -> float:
    """单个目标物体的得分。

    * `p1_passed` 为真 ⇒ `target_base + target_span · Q`（30–100，原口径不变）。
    * `p1_passed` 为假 ⇒ **接近度分** `approach_max · φ`（0–`approach_max`，默认 10）。

    `φ` 是「把目标物往有效放置区推近了多少」的相对比例（见 `approach_phi`）。
    `φ` 缺省 0 ⇒ 与旧口径逐字一致（`p1_passed=False` ⇒ 0），
    所以**只在调用方显式传入 φ 时**才启用接近度分。
    """
    if not p1_passed:
        # ★ 2026-10-19 接近度项：给「接近但没进区域」的弱方法一点分。
        #   红线不变：什么都没做 ⇒ φ=0 ⇒ 0 分（不会白拿分）。
        return weight * float(approach_max) * clip(phi)
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
        # ★ 接近度项：`ti` 里没给 `phi` 就是 0（与旧口径逐字一致）。
        phi=float(ti.get("phi", 0.0)),
        approach_max=float(c.get("approach_max", 10.0)),
    ) for ti in targets)
    p = penalty_disturb(displacements=displacements, constants=c)
    # ★ 2026-10-19 免赔额：`P_disturb` 的前 `disturb_franchise` 分不计。
    #   这是给弱方法的出口 —— 摸索过程里的小磕碰不把接近分吃光。
    #   `disturb_franchise=0` ⇒ 与旧口径逐字一致。
    p_net = max(0.0, p - float(c.get("disturb_franchise", 0.0)))
    return {"score_target": t, "penalty_disturb": p, "penalty_disturb_net": p_net,
            "score_100": max(0.0, t - p_net)}


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
        # ★ 接近度分上限（P1 未过时）。**远小于** 30 的入门档，
        #   保证「接近」永远拿不到「进区域」的分。
        #   2026-10-19 ②：8 → 10（用户口径：弱方法上限 10 分）。
        "approach_max": 10.0,
        # ★ 2026-10-19 ② 接近度归一尺度上限（米）。
        #   `approach_phi` 的分母取 `min(d_init, approach_ref_m)`：
        #   起始距离超过 10 cm 的 case 一律按 10 cm 归一，
        #   避免「起始远」的 case 把同样的真实进展稀释掉（实测 21.6 cm
        #   起始时推进 6.4 cm 只值 φ=0.30）。`0` ⇒ 退回纯相对口径。
        "approach_ref_m": 0.10,
        # ★ 2026-10-19 ③ 扰动免赔额：`P_disturb` 的前几分不计。
        #   给弱方法的出路 —— 「摸索过程里的小磕碰」不再把接近分吃光。
        #   在 0–70 的扰动尺度上取值；`0` ⇒ 与旧口径逐字一致。
        "disturb_franchise": 5.0,
    }
