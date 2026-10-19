"""参考图像：用 `target_layout` 现渲染（第 5 步 / 模式 B）。

## 为什么要有这个模块

模式 B（Q4）下「目标参考图像」是选手**唯一的感知来源之一**，
但 **casebank 里根本没有图像** ——

```bash
$ find judge_private/casebank -name "*.png" -o -name "*.jpg"     # 空
```

只有 `.obj` 网格与 `case.json`。所以参考图必须**现造**。

## 为什么可以「现造」得很简单

已核实（第 5 步勘察）：

```text
case.json:
  contract.target_layout = {coordinate_frame, scene_settings, objects:[{instance,position,rotation}]}
  initial_layouts        = [同样形状的 5 个 layout]
  objects                = 网格 + 质量/惯量（**与布局无关**）

⇒ 目标场景 = 同一个 bundle，只把 initial_layouts 换成 [target_layout]
```

`create_case_env` 的 `_load_scene` 就是从 `bundle['initial_layouts'][seed]['objects']`
取位姿的（`env_factory.py`），**不需要新的场景构建代码**。

## 四个设计决定（第 5 步已拍板）

| # | 决定 | 理由 |
|---|---|---|
| 1 | 参考图 = **目标摆位渲染图** | 赛题目录名里的 `real_image`/`scene_image` 资产不存在；现渲染至少给「目标摆置长什么样」 |
| 2 | 机器人**入镜但保持初始位** | 目标场景必须有机器人（`_initialize_episode` 会 `agent.reset`），但不摆成目标姿态 |
| 3 | 机位与实时帧**完全一致** | 选手才能把参考图与当前画面做对比；给参考图挑更好的角度等于泄漏 |
| 4 | **不改 casebank** | 它是 `2casefix` 的只读证据源（红线） |

## 与判定的隔离

参考图渲染**不参与任何判定**：渲染完就 `close()`，用的是一次性的 bundle 副本，
**从不写回 casebank**，也不进 `ResultGate` 的任何账本。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

__all__ = ["ReferenceImageError", "render_reference_image", "goal_bundle",
           "load_reference_image", "reference_image_path"]

_MAX_REFERENCE_BYTES = 1024 ** 2


class ReferenceImageError(RuntimeError):
    """参考图渲染失败。**不降级、不伪造** —— 模式 B 下没有参考图这条链路不成立。"""


def goal_bundle(bundle: dict, *, seed: int = 0) -> dict:
    """把 bundle 改造成「目标摆位」场景。

    只动 `initial_layouts` —— 其余（`objects` 的网格、质量、惯量、
    `robot_base_candidate_*`、`static_objects`）**原样保留**，
    因为那些与布局无关。

    `initial_robot_qpos` 也保留 —— 这就是「机器人入镜但保持初始位」。
    """
    target = bundle.get("contract", {}).get("target_layout")
    if not isinstance(target, dict) or not isinstance(target.get("objects"), list):
        raise ReferenceImageError("case has no usable contract.target_layout")
    if not target["objects"]:
        raise ReferenceImageError("contract.target_layout.objects is empty")

    out = copy.deepcopy(bundle)
    # `_load_scene` 读 `bundle['initial_layouts'][seed]['objects']`
    # —— 所以把目标布局放进 index 0，渲染时 seed 传 0。
    out["initial_layouts"] = [copy.deepcopy(target)]
    # 目标布局可能带自己的 scene_settings；`_load_scene` 不读它，
    # 但留着能让人看出这副本是从哪来的。
    return out


def reference_image_path(reference_root, case_id: str) -> Path:
    """预渲染参考图的位置。

    ⚠️ **放在 `judge_private/` 下，不放进 `casebank/`** —— casebank 是
    `2casefix` 的只读证据源（红线：不在它内部改任何文件）。
    """
    return Path(reference_root) / f"{case_id}.json"


def load_reference_image(reference_root, case_id: str) -> dict | None:
    """读**预渲染**的参考图。没有就返回 `None`（调用方决定怎么办）。

    评测链路**只走这条** —— 不在这里现渲染：渲染要 GPU，而评测（尤其
    回归/CPU 档）会把 `CUDA_VISIBLE_DEVICES` 清空，现渲染会直接崩
    （实测：`RuntimeError: failed to find device "cuda:0"`）。
    """
    path = reference_image_path(reference_root, case_id)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) and value.get("data_base64") else None


def render_reference_image(bundle: dict, casebank, *, seed: int = 0) -> dict:
    """**离线生成**一张参考图（需要 GPU / SAPIEN 渲染）。

    ⚠️ 这个函数**不在评测链路里调用**。评测只读 `load_reference_image`。
    它由 `work/scripts/prerender_reference_images.py` 批量调用，
    把 73 张图一次生成到 `judge_private/reference_images/`。
    """
    # 惰性 import：把 mani_skill/sapien/torch 拉起来很慢，不该在模块导入时做
    import numpy as np

    from track2_judge.observation.provider import rgb_payload
    from track2_judge.scenario.env_factory import (
        create_case_env, fixed_camera_from_bundle,
    )

    planned = goal_bundle(bundle, seed=seed)
    # ★ 2026-10-19 修复：机位必须与实时帧**同源**（都用**初始布局**）。
    #
    #   之前这里传的是 `planned` —— `render_reference_image` 把
    #   `initial_layouts` 换成了 `target_layout` 再算机位，而实时帧走
    #   `build_case_session()` → `fixed_camera_from_bundle(bundle, 0)`
    #   用的是**初始布局**。`fixed_camera_from_bundle` 从布局点云现算
    #   `aim`/`distance`，两种布局的点云不同 ⇒ **参考图与实时帧机位不同**。
    #   实测 50 例：相机中心偏移中位 3.1 cm / 最大 8.7 cm，摄距差最大
    #   11.7 cm（004）。而 `fixed_camera_from_bundle` 的 docstring 明写
    #   「参考图与实时帧用同一个规格，选手才能把两者对比」—— 实现违反了它。
    #
    #   ⚠️ 「画面内容」与「机位」必须**分开取**：
    #     * 内容 = `planned`（目标布局）—— 必须，否则参考图就不是「目标摆位」；
    #     * 机位 = `bundle`（初始布局）—— 必须，否则等于把答案编码进观测
    #       （docstring 硬约束 1），且与实时帧不同机位（选手没法对比）。
    camera_spec = fixed_camera_from_bundle(bundle, 0)

    env = None
    try:
        env = create_case_env(planned, casebank, 0,
                              allow_unapproved_diagnostics=True,
                              diagnostic_camera=camera_spec)
        env.reset(seed=0)
        env.scene.update_render()
        cam = env._human_render_cameras["diagnostic_public_rgb"]
        cam.capture()
        rgb = cam.camera._render_cameras[0].get_picture("Color")
        if rgb.dtype != np.uint8:
            rgb = np.clip(np.asarray(rgb)[:, :, :3] * 255, 0, 255).astype(np.uint8)
        frame = np.asarray(rgb)[:, :, :3]
        if frame.size == 0 or frame.std() <= 0.5:
            raise ReferenceImageError(
                f"reference render looks blank (std={float(frame.std()):.3f})")
        payload = rgb_payload(frame, source="target_layout_render")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:                                  # noqa: BLE001
                pass

    if len(payload["data_base64"]) > 1_500_000:
        raise ReferenceImageError("reference image exceeds budget")
    return payload


def load_case_bundle(casebank, case_id: str) -> dict:
    """读一例的 `case.json`。与 runtime 节点用的是同一个路径规则。"""
    path = Path(casebank) / "cases" / case_id / "case.json"
    if not path.is_file():
        raise ReferenceImageError(f"case not found: {case_id}")
    return json.loads(path.read_text(encoding="utf8"))
