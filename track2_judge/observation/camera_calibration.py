"""相机标定：把「图像像素」映射回「世界坐标」的那一份契约。

## 为什么必须有这个模块

比赛题目要求「参考目标图像，自主完成目标识别 / 任务拆解」。选手从图里认出目标
之后，下一步必然是**把像素坐标换成世界坐标**才能发关节目标：

```text
像素 (u, v)  →  世界 (x, y)  ── 需要机位 + 视场角 + 画面尺寸
```

没有标定，选手只能拿到一张图却不知道它对应世界的哪里 —— 于是「从图里定位物体」
这件事根本做不了，闭环策略只剩开环。

## 它为什么不是答案

标定由 `fixed_camera_from_bundle(bundle, seed)` 从**初始布局** + 机器人基座算出：

* 只用**初始布局**（选手第一帧就能看到的东西），**不碰 `target_layout`**
* 与「哪个物体是目标」「目标在哪」无关

所以给它是给「怎么看」，不是给「答案是什么」。

## 两条路必须给同一个东西

⚠️ 这个模块存在的直接原因：标定原先只拼在**裁判侧**那条路的
`text_instruction` 里（`DIAGNOSTIC_RGB_CAMERA=` 前缀，历史遗留命名），
而 PAOS runtime 那条路**没有** —— 同一份 skill 在两条路上拿到的标定不一样。

现在两条路都调 `camera_calibration()`，产出放进 `task_inputs.camera_calibration`。

## 怎么用（**这是关键**）

不要自己假设「画面的 u 轴对应世界的 x」。这台相机是**正俯视且 `up` 取的是场景
主轴**（`fixed_camera_from_bundle` 会挑跨度更大的那个轴），所以 u/v 与世界 x/y
的对应关系**会随场景变**。

标定直接给出**两个基向量**，规则恒为：

```text
world = top_left_m + (u + 0.5) · u_axis_m + (v + 0.5) · v_axis_m
```

* `top_left_m` —— 像素 `(0, 0)` **中心**处对应的世界点
* `u_axis_m` —— u 每 +1 像素，世界坐标的位移（含正负号）
* `v_axis_m` —— v 每 +1 像素，世界坐标的位移

**这对任何机位都成立**，选手不需要知道 up 是哪个轴。

（实测校准：case 001 的 `up=(1,0,0)`，于是 `u_axis_m ∝ (0,-1,0)`、
`v_axis_m ∝ (-1,0,0)` —— 也就是 u 沿 **−y**、v 沿 **−x**。
直接假设「u→x、v→y」会把物体投到画面外，实测偏差 ~460 px。）
"""
from __future__ import annotations

__all__ = ["camera_calibration", "CALIBRATION_FIELD"]

#: 公开输入里放标定的字段名。**与参考图同一层**（都在 `task_inputs` 下）。
CALIBRATION_FIELD = "camera_calibration"

#: 对外暴露的标定字段。
_PUBLIC_KEYS = ("aim_m", "eye_m", "center_m", "distance_m", "fov_rad",
                "width", "height", "span_m",
                "top_left_m", "u_axis_m", "v_axis_m", "up_axis")


def _unit(values):
    import math
    norm = math.sqrt(sum(float(v) ** 2 for v in values))
    return [float(v) / norm for v in values] if norm > 1e-12 else [0.0, 0.0, 0.0]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def _up_axis(up) -> str:
    """把 `look_at` 的 up 向量收成轴名（`"+x"` / `"-z"` 这类）。**仅作提示** ——
    真正的换算请用 `u_axis_m` / `v_axis_m`。"""
    try:
        values = [float(v) for v in up]
    except (TypeError, ValueError):
        return "unknown"
    if len(values) != 3:
        return "unknown"
    index = max(range(3), key=lambda i: abs(values[i]))
    if abs(values[index]) < 1e-9:
        return "unknown"
    return ("+" if values[index] > 0 else "-") + "xyz"[index]


def camera_calibration(spec: dict, *, width: int = 256, height: int = 256) -> dict:
    """把 `fixed_camera_from_bundle()` 的输出整理成**给选手的标定**。

    参数：
        spec     `{'center_m','eye_m','fov_rad','aim_m','distance_m','up'}`
        width/height  画面尺寸（与 `env_factory` 的 `CameraConfig` 一致）

    返回的字典直接放进 `task_inputs["camera_calibration"]`。

    ⚠️ **两个基向量是核心**（`u_axis_m` / `v_axis_m`）—— 它们让选手不必猜
    「u 对应世界的哪个轴」。推导见 `_screen_axes()`。
    """
    import math

    aim = [float(v) for v in spec["aim_m"]]
    fov = float(spec["fov_rad"])
    distance = float(spec["distance_m"])
    span = 2.0 * distance * math.tan(fov / 2.0)     # 画面的世界跨度（米，两方向同）

    u_axis, v_axis = _screen_axes(spec)
    u_axis_m = [c * span / float(width) for c in u_axis]
    v_axis_m = [c * span / float(height) for c in v_axis]
    # 像素 (0,0) 的中心 = 画面中心 - 半个画面的位移
    top_left = [aim[i] - (u_axis_m[i] * width + v_axis_m[i] * height) / 2.0
                for i in range(3)]

    return {
        "aim_m": aim,
        "eye_m": [float(v) for v in spec.get("eye_m", [])],
        "center_m": [float(v) for v in spec.get("center_m", aim)],
        "distance_m": distance,
        "fov_rad": fov,
        "width": int(width),
        "height": int(height),
        "span_m": [span, span],
        "top_left_m": top_left,
        "u_axis_m": u_axis_m,
        "v_axis_m": v_axis_m,
        "up_axis": _up_axis(spec.get("up")),
    }


def _screen_axes(spec: dict):
    """求「屏幕 u / v 方向」在世界里的**单位**方向向量。

    相机是正俯视（`eye` 在 `aim` 正上方），`look_at(eye, aim, up=场景主轴)`。
    `look_at` 的基（ManiSkill `sapien_utils.look_at`）：

    ```text
    forward = normalize(aim - eye)          # 往下看
    left    = normalize(cross(up, forward))
    up'     = normalize(cross(forward, left))
    ```

    SAPIEN 相机约定 `(forward, right, up_cam) = (x, -y, z)`，所以：

    ```text
    right_world = -left          →  屏幕 u 沿它增大
    down_world  = -up'           →  屏幕 v 沿它增大
    ```

    ⚠️ **不能假设「u→+x、v→+y」**：`up` 是场景主轴（跨度更大的那个），
    所以映射随场景翻转。实测 case 001（`up=(1,0,0)`）：
    `u_axis ∝ (0,-1,0)`、`v_axis ∝ (-1,0,0)` —— 与直觉相反。
    """
    eye = [float(v) for v in spec["eye_m"]]
    aim = [float(v) for v in spec["aim_m"]]
    up_in = [float(v) for v in spec.get("up", (0.0, 0.0, 1.0))]

    forward = _unit([aim[i] - eye[i] for i in range(3)])
    left = _unit(_cross(_unit(up_in), forward))
    up_cam = _unit(_cross(forward, left))

    u_axis = [-c for c in left]        # right = -left
    v_axis = [-c for c in up_cam]      # down = -up'
    return u_axis, v_axis


def calibration_shape() -> tuple[str, ...]:
    """给测试用的字段清单（`camera_calibration()` 的键）。"""
    return _PUBLIC_KEYS
