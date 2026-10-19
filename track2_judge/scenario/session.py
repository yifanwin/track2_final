"""**唯一的**场景/观测构造点（C1 / 2026-10-17）。

## 为什么要有这个模块

之前「造一个 case 的 env + 观测」这件事被**写了两遍**：

```text
判官侧  select_case.py:173-190      （release_entry / run_suite 那条路）
PAOS 侧 node.py:203-227            （runtime 节点那条路）
```

两边共享组件是同一套（`ObservationProvider` / `create_case_env` /
`ManiSkillSimulatorAdapter`），但**接线各写一遍**，于是实测漂了四个字段：

| 构造点 | `live=` | `camera=` | `max_steps=` | `max_sim_time_s=` |
|---|---|---|---|---|
| `select_case.py` | **缺（默认 False）** | **None** | 缺（2400） | 缺（120 s） |
| `node.py` | **True** | 真回调 | 显式 | 显式 |

**而且漂了不报错** —— `live=False` 只是「不重拍」，不抛异常。

## 怎么用

```python
session = build_case_session(bundle, bank, seed,
                             camera=True,                       # 正式评测开；CPU 回归关
                             public_fields=tuple(public),       # 选手可见的字段集合
                             live=True)                         # 每步重拍（模式 B）
adapter = session.adapter
session.reset(seed, public_task_inputs=public)
```

## 一条硬规则

**`camera` 由「正式评测 vs CPU 回归」决定，不由「走哪条路」决定。**
加新字段时**只改这一处** —— 这是消灭漂移的唯一手段。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = ["CaseSession", "build_case_session", "camera_closure"]

#: 默认预算。**必须显式传** —— 留默认值是漂移的温床（见模块说明）。
DEFAULT_MAX_STEPS = 400
DEFAULT_MAX_SIM_TIME_S = 20.0


def camera_closure(spec: dict, *, gpu_env: str = "TRACK2_RENDER_GPU"):
    """按 `fixed_camera_from_bundle()` 的规格造一个「取一帧」的回调。

    它被 `ObservationProvider` 每次 `observe()` 调用（`live=True` 时），
    返回 RGB uint8 `(H, W, 3)`。

    ⚠️ 渲染 GPU 的选择在这里**不写死**：`prototype_env` 在 import 时会把
    `CUDA_VISIBLE_DEVICES` 清空（CPU 档约定），所以调用方要在建立 session
    **之前**把它设好。这里只负责取图。
    """
    import numpy as np

    def grab(env) -> Any:
        env.scene.update_render()
        cam = env._human_render_cameras["diagnostic_public_rgb"]
        cam.capture()
        rgb = cam.camera._render_cameras[0].get_picture("Color")
        if rgb.dtype != np.uint8:
            rgb = np.clip(np.asarray(rgb)[:, :, :3] * 255, 0, 255).astype(np.uint8)
        return np.asarray(rgb)[:, :, :3]

    return grab


@dataclass
class CaseSession:
    """一个 case 的场景 + 观测 + 适配器的**唯一**封装。"""

    case_id: str
    seed: int
    bundle: dict
    camera_spec: dict | None
    adapter: Any
    public_fields: tuple[str, ...] = ()
    _closed: bool = field(default=False, repr=False)

    # ── 给 Judge 侧的便利方法（私有真值，**不经 Tool**）

    def reset(self, seed: int | None = None, *, public_task_inputs: dict) -> None:
        """重建场景并采初始证据。**会重建 env**（`adapter.reset` 的语义）。"""
        self.adapter.reset(seed=self.seed if seed is None else seed,
                           public_task_inputs=public_task_inputs)

    def judge_state(self) -> dict:
        """**judge-only** 物理快照。

        ⚠️ 这必须是**实际跑过的那个 env** 的状态 —— 不许另建同 seed 场景再读
        （`same seed` 恢复不了选手执行产生的物体终态 / 碰撞 / 滑落 / 接触 / 动态误差）。
        见 `requirement/Track2_边界决策.md` §3。
        """
        return self.adapter.get_final_state()

    def layout(self) -> dict:
        return self.adapter.get_layout()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self.adapter.close()
            except Exception:                               # noqa: BLE001
                pass


def build_case_session(bundle: dict, casebank, seed: int, *,
                       camera: bool,
                       public_fields: tuple[str, ...] = (),
                       live: bool = True,
                       checkpoints: list | None = None,
                       recovery: bool = False,
                       max_steps: int = DEFAULT_MAX_STEPS,
                       max_sim_time_s: float = DEFAULT_MAX_SIM_TIME_S,
                       camera_spec: dict | None = None,
                       render_device: str | int | None = None,
                       allow_unapproved_diagnostics: bool = True,
                       case_id: str | None = None) -> CaseSession:
    """造一个 case 的完整会话。**两侧（判官 / PAOS）都调它。**

    参数：
        camera         要不要建相机。**正式评测 True，CPU 回归 False。**
        public_fields  选手可见的字段集合（决定 `ObservationProvider` 的白名单）
        live           每步重拍（模式 B 下恒为 True；CPU 回归可为 False）
        camera_spec    省略时按 `fixed_camera_from_bundle(bundle, seed)` 算
        seed           初始布局索引
    """
    from track2_judge.contract.trusted_state import spatial_snapshot
    from track2_judge.observation.provider import ObservationProvider
    from track2_judge.scenario.env_factory import (
        create_case_env, fixed_camera_from_bundle)
    from track2_judge.scenario.initial_gate import validate_initial_scene
    from track2_judge.scenario.prototype_backend import physical_snapshot
    from track2_judge.scenario.prototype_env import SETTINGS
    from track2_judge.toolapi.dispatch import ManiSkillSimulatorAdapter

    if camera and camera_spec is None:
        camera_spec = fixed_camera_from_bundle(bundle, seed if False else 0)
        # ⚠️ 固定用 seed 0：机位只依赖初始布局，`_fit_distance` 对 5 个 seed
        #    给出几乎相同的结果 —— 而且**参考图是用 seed 0 预渲染的**，
        #    两者必须一致，否则参考图与实时帧不同机位。

    return CaseSession(
        case_id=case_id or bundle.get("source_case_id") or "",
        seed=seed, bundle=bundle, camera_spec=camera_spec,
        public_fields=tuple(public_fields),
        adapter=ManiSkillSimulatorAdapter(
            lambda: create_case_env(bundle, casebank, seed,
                                    allow_unapproved_diagnostics=allow_unapproved_diagnostics,
                                    diagnostic_camera=camera_spec,
                                    render_device=render_device),
            snapshot=physical_snapshot,
            spatial_provider=spatial_snapshot,
            metadata={n: {"pivot_m": [0, 0, 0]} for n in bundle["objects"]},
            scene_settings=SETTINGS,
            public_fields=tuple(public_fields),
            observation_provider=ObservationProvider(
                approved_fields=tuple(public_fields),
                camera=camera_closure(camera_spec) if camera_spec is not None else None,
                camera_required=bool(camera),
                live=bool(live) and camera_spec is not None),
            checkpoints=list(checkpoints or []),
            recovery=bool(recovery),
            initial_validation=validate_initial_scene,
            max_steps=int(max_steps),
            max_sim_time_s=float(max_sim_time_s),
        ))
