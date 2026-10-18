"""Track 2 的 ToolSpec 声明 —— **唯一权威来源**。

一份声明，三处复用：

1. **本地校验**：Gateway 收到 invoke 时按 `input_schema` 校验
2. **Gateway `/tools` 响应**：PAOS 读它来 freeze `spec_sha256`
3. **PAOS `required_tools`**：Bundle 的 `skill.yaml` 从这里取 tool_id 列表

## 为什么必须 byte 级稳定

PAOS 的 `ForgeSkillBindingResolver.preview()` 对每个 ToolSpec 算
`canonical_sha256(spec)` 并冻结；之后每次绑定校验
（`forge/binding.py:223`）。只要 JSON 序列化结果变一个字节，就会报
`Forge ToolSpec 'x' changed after AgentTask binding`。

所以：**`SPECS` 里的字典字面量顺序无关紧要**（`canonical_sha256` 会 `sort_keys`），
但**不要**在 spec 里放任何随时间变化的值（时间戳、随机 ID、计数器）。

`canonical_sha256` 的实现与 `PhyAgentOS/forge/binding.py:67` **逐字一致**：
`json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)` → utf-8 → sha256。

## 三个 Tool 的语义

| tool_id | semantics | 说明 |
|---|---|---|
| `track2.env.observe` | **query** | 读当前观测（机器人本体 + 批准图像） |
| `track2.robot.joint_control` | **action** | 一步关节目标 + 夹爪开度 |
| `track2.robot.gripper_control` | **action** | 只动夹爪 |

依据：Q5（Action 语义）、Q8（关节控制）、Q11（路线 B，自造 `track2.*`）、Q4（模式 B 观测）。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from track2_judge.toolapi.validation import SchemaError, is_valid, validate

__all__ = [
    "SPECS", "TOOL_IDS", "REQUIRED_TOOLS", "ENDPOINTS",
    "canonical_sha256", "get_spec", "spec_sha256", "validate_arguments",
    "validate_output", "tool_ids_for_profile",
]

IMPLEMENTATION_ID = "track2.runtime"
# 批准图像的声明。引用处用 Python 层的名字共享 ——
# 序列化后是完全 inline 的 JSON（PAOS 不解析 `$ref`）。
#
# ⚠️ **里面没有 `seq`**（第 5 步的决定）：`seq` 是官方 gateway 推
# `/ws/images` 时盖的（`runtime_service.py:205`），不在 `observe` 的返回里。
# 选手判「这一帧是不是新的」靠 `sha256` 自比。
_PNG_IMAGE = {
    "type": "object",
    "additionalProperties": False,
    "required": ["mime_type", "width", "height", "data_base64", "sha256"],
    "properties": {
        "mime_type": {"const": "image/png"},
        "width": {"type": "integer", "minimum": 1, "maximum": 512},
        "height": {"type": "integer", "minimum": 1, "maximum": 512},
        "data_base64": {"type": "string"},
        "sha256": {"type": "string"},
        "source": {"type": "string"},
    },
}

ROBOT_FRAME_PROFILE = {
    "robot_id": "piper",
    "base_frame": "arm_base",
    "tool_frame": "tcp",
    "frames": {},
    "directions": {
        "forward": {"frame": "arm_base", "axis": "x", "sign": 1},
    },
}

# ─────────────────────────────────────────────────────────── observe（Query）

_OBSERVE = {
    "tool_id": "track2.env.observe",
    "implementation_id": IMPLEMENTATION_ID,
    "endpoint_id": "track2.env",
    "operation": "observe",
    "semantics": "query",
    "description": "Read the approved current observation: robot proprioception, control clock, and "
                   "the policy camera frame when RGB is requested.",
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            # 模式 B（Q4）：不给任何目标位姿/容差。目标参考图由 task_inputs 里的
            # reference_image 承载，而不是把 grounded 数值目标直接喂给选手。
            "include_rgb": {"type": "boolean", "default": True},
        },
    },
    "output_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["observation_id", "captured_at_s", "sim_time_s", "control_steps", "robot"],
        "properties": {
            "observation_id": {"type": "string"},
            "captured_at_s": {"type": "number"},
            "sim_time_s": {"type": "number"},
            "control_steps": {"type": "integer", "minimum": 0},
            "robot": {
                "type": "object",
                "additionalProperties": False,
                "required": ["joint_position", "joint_velocity"],
                "properties": {
                    "joint_position": {"type": "array", "items": {"type": "number"},
                                       "minItems": 8, "maxItems": 8},
                    "joint_velocity": {"type": "array", "items": {"type": "number"},
                                       "minItems": 8, "maxItems": 8},
                    "joint_limits_rad": {"type": "array"},
                    "tcp_position_m": {"type": "array", "items": {"type": "number"},
                                       "minItems": 3, "maxItems": 3},
                    "tcp_quaternion_wxyz": {"type": "array", "items": {"type": "number"},
                                            "minItems": 4, "maxItems": 4},
                },
            },
            # task_inputs 是「公开输入」的容器。模式 B 收窄的是**内容**
            # （去掉 target_pose / eps_pos / eps_rot_deg / region_aabb_m），
            # 不是这个字段本身 —— 所以 ToolSpec 这里对内容不做约束。
            "task_inputs": {"type": "object"},
            # 图像载体。模式 B（Q4）下**这是选手唯一的感知来源**。
            #
            # ⚠️ 键名从 `first_frame` 改成 `camera`（第 5 步）：模式 B 下它
            # **每控制步都更新**，`first_frame` 这个名字会误导选手只调一次
            # `observe` 就以为拿到了全部感知。
            #
            # `camera_status` 是运行事实（拍到了没），不是目标答案。
            "images": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "camera": _PNG_IMAGE,
                    "camera_status": {"type": "string"},
                },
            },
        },
    },
    "readiness": [],
    "robot_frame_profile": ROBOT_FRAME_PROFILE,
}

# ──────────────────────────────────────── joint_control / gripper_control（Action）

# Q13：超时后必须先 cancel，再等 backend 确认真实停止。
# Q14（2026-10-18 定）：全量 73 例实测 p50 0.097 / p99 0.675 / max 0.940 s
#   → 取 5.0 s（≈ max 的 5.3 倍）。**客户端先到点**（< gateway 的
#   `invoke_timeout_ms=30000`），避免两者同刻到点的竞态。
ACTION_TIMEOUT_S = 5.0
ACTION_TIMEOUT_SOURCE = "MEASURED_73_CASES_P99_PLUS_MARGIN"

# Q9：关节控制天然串行，两个 Action 均为 1。
ACTION_MAX_CONCURRENCY = 1

_JOINT_CONTROL = {
    "tool_id": "track2.robot.joint_control",
    "implementation_id": IMPLEMENTATION_ID,
    "endpoint_id": "track2.robot",
    "operation": "joint_control",
    "semantics": "action",
    "description": "Command one joint position target plus gripper opening. The physical effect "
                   "completes when the Action terminates.",
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["joint_position", "gripper_opening_m"],
        "properties": {
            # 6 个臂关节（deg 无关，单位 rad）。夹爪开度单列，不混在这里。
            "joint_position": {"type": "array", "items": {"type": "number"},
                               "minItems": 6, "maxItems": 6},
            "gripper_opening_m": {"type": "number", "minimum": 0, "maximum": 0.1},
        },
    },
    "output_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["control_steps", "sim_time_s"],
        "properties": {
            "control_steps": {"type": "integer", "minimum": 1},
            "sim_time_s": {"type": "number"},
        },
    },
    "readiness": [],
    "robot_frame_profile": ROBOT_FRAME_PROFILE,
}

_GRIPPER_CONTROL = {
    "tool_id": "track2.robot.gripper_control",
    "implementation_id": IMPLEMENTATION_ID,
    "endpoint_id": "track2.robot",
    "operation": "gripper_control",
    "semantics": "action",
    "description": "Command the gripper opening only, holding the arm at its current joint target.",
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["gripper_opening_m"],
        "properties": {
            "gripper_opening_m": {"type": "number", "minimum": 0, "maximum": 0.1},
        },
    },
    "output_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["control_steps", "sim_time_s"],
        "properties": {
            "control_steps": {"type": "integer", "minimum": 1},
            "sim_time_s": {"type": "number"},
        },
    },
    "readiness": [],
    "robot_frame_profile": ROBOT_FRAME_PROFILE,
}

# ─────────────────────────────────────────────────────────────────────────────

SPECS: dict[str, dict[str, Any]] = {
    _OBSERVE["tool_id"]: _OBSERVE,
    _JOINT_CONTROL["tool_id"]: _JOINT_CONTROL,
    _GRIPPER_CONTROL["tool_id"]: _GRIPPER_CONTROL,
}

TOOL_IDS: tuple[str, ...] = tuple(SPECS)

# Bundle 的 skill.yaml 直接用这个列表作为 required_tools。
REQUIRED_TOOLS: tuple[str, ...] = TOOL_IDS

# endpoint_id -> operation -> tool_id（Gateway 的 Query 路由用）
ENDPOINTS: dict[str, dict[str, str]] = {}
for _spec in SPECS.values():
    ENDPOINTS.setdefault(_spec["endpoint_id"], {})[_spec["operation"]] = _spec["tool_id"]


def canonical_sha256(value: Any) -> str:
    """与 `PhyAgentOS/forge/binding.py:67` 逐字一致的规范化哈希。"""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def get_spec(tool_id: str) -> dict[str, Any]:
    """取一个 ToolSpec。未知 tool_id 抛 `KeyError`。"""
    return SPECS[tool_id]


def spec_sha256(tool_id: str) -> str:
    """单个 ToolSpec 的冻结哈希 —— 与 PAOS 的 `spec_sha256` 同算法。"""
    return canonical_sha256(SPECS[tool_id])


def validate_arguments(tool_id: str, arguments: Any) -> None:
    """按 tool 的 `input_schema` 校验入参。不通过抛 `SchemaError`。"""
    if tool_id not in SPECS:
        raise SchemaError("", f"unknown tool {tool_id!r}")
    validate(arguments, SPECS[tool_id]["input_schema"], "")


def validate_output(tool_id: str, value: Any) -> None:
    """按 tool 的 `output_schema` 校验出参。不通过抛 `SchemaError`。

    在 Gateway 里对每次调用做一次，保证**交给 PAOS/选手的东西**符合已冻结的声明。
    """
    validate(value, SPECS[tool_id]["output_schema"], "")


def tool_ids_for_profile(profile: str | None = None) -> tuple[str, ...]:
    """某 profile 下可用的 tool 集合。

    现状只有一个 profile 面，所以直接返回全部 —— 但接口留在这里，
    第 3 步 Bundle 的 `profiles.<name>.required_tools` 会覆盖它。
    """
    return REQUIRED_TOOLS


def is_declared(tool_id: str, profile: str | None = None) -> bool:
    """该 tool 是否在指定 profile 的声明集合内。

    第 1 步删掉了原来的 `tool_profile` 字符串门禁，准入由这里接管：
    **未声明的 tool 一律拒绝**（对应验收标准 2）。
    """
    return tool_id in tool_ids_for_profile(profile)


__all__ += ["is_declared", "ACTION_TIMEOUT_S", "ACTION_TIMEOUT_SOURCE",
            "ACTION_MAX_CONCURRENCY", "SchemaError", "is_valid"]
