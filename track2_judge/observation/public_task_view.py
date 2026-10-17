"""模式 B 的公开输入投影（Q4 + Q10）。

## 这一版删掉了什么

模式 B = **选手拿原始图文自己理解目标**，不由 OS 代为感知。所以公开输入里
**不能有任何数值形态的目标**。上一版投影了三样，第 5 步全部删除：

| 删掉的 | 为什么 |
|---|---|
| `entry['target_pose']`（P2 精确目标位姿） | 直接给出目标位姿 = 模式 A 的做法，与 Q4 冲突 |
| `entry['eps_pos']` / `entry['eps_rot_deg']`（容差） | 容差告诉选手「离多近算成功」，同属答案 |
| `entry['region']`（P1 的 `region_aabb_m`） | **Q10 已定不公开** —— 模式 B 下给区域等于把答案告诉选手 |

## 现在投影什么

| 字段 | 内容 |
|---|---|
| `text_instruction` | 自然语言指令（「把 X 放进目标区域」） |
| `reference_image` | **目标参考图像** —— 由 `reference_image.py` 用 `target_layout`
**现渲染**（casebank 里没有图像资产），机位与实时帧**完全一致** |

**注意谓词本身仍然列出**（`P1`/`P2` 的类型与目标物体名）：
它们是**任务语义**（「哪个物体、要满足哪条谓词」），不是数值答案 ——
选手本来就能从自然语言指令里得到同样信息。
被删掉的是**数值**（位姿/容差/区域），不是「要做什么」。

## 与判定的关系

评分仍然用完整的 `task['predicates']`（裁判侧，含全部数值）——
这里改的只是**选手能看到什么**。
"""
import json
import math

from track2_judge.observation.provider import ObservationProvider
from track2_judge.scoring.scorer_bridge import v3

PROFILE = 'track2-public-goals-candidate-1'

#: 允许出现在公开输入里的谓词字段。**数值字段一个都不在里面**。
_PUBLIC_PREDICATE_FIELDS = ('type', 'object', 'weight', 'reference', 'relation')
#: 允许的约束字段（同样不含数值）。
_PUBLIC_CONSTRAINT_FIELDS = ('type',)


def _instruction(obj: str) -> str:
    """自然语言指令。**不含数值**。"""
    return (
        f"Move the object {obj} to the position shown in the reference image; "
        "preserve the other objects."
    )


def project(task, *, reference_image=None):
    """把一例的私有 contract 投影成**选手可见的公开输入**。

    `reference_image` 是 `observation.reference_image.render_reference_image`
    的产物（`rgb_payload` 形状）。**必须给** —— 模式 B 下它是选手唯一的
    目标信息来源；不给就只剩一句自然语言指令，信息量不足以完成任务。
    """
    if task.get('task_type') != 'pick_place':
        raise ValueError('public profile supports only the confirmed 73 pick/place tasks')

    public = []
    for pred in task['predicates']:
        entry = {k: v for k, v in pred.items() if k in _PUBLIC_PREDICATE_FIELDS}
        if 'type' not in entry or 'object' not in entry:
            raise ValueError('unapproved predicate projection')
        # 数值目标一个都不许漏出去 —— 白名单之外还做一次**数值扫描**，
        # 防止将来有人往谓词里塞新的数值字段（`region` / `target_pose` 就是这么来的）。
        _reject_numbers(entry, where=f"predicate {entry['type']}")
        public.append(entry)

    constraints = []
    for con in task.get('constraints', []):
        if con['kind'] != 'non_target_displacement':
            raise ValueError('unapproved constraint projection')
        entry = {k: v for k, v in con.items() if k in _PUBLIC_CONSTRAINT_FIELDS}
        _reject_numbers(entry, where="constraint")
        constraints.append(entry)

    obj = next((p['object'] for p in task['predicates'] if p['type'] == 'P2'), None)
    if obj is None:
        raise ValueError('pick_place task without a P2 target object')

    raw = {
        'text_instruction': _instruction(obj),
        'predicates': public,
        'constraints': constraints,
    }
    if reference_image is not None:
        raw['reference_image'] = reference_image

    encoded = json.dumps(raw, allow_nan=False)
    if len(encoded) > 16384 * 8:
        raise ValueError('public task budget')
    return ObservationProvider(
        approved_fields=('text_instruction', 'reference_image', 'predicates', 'constraints')
    ).prepare(raw)


def _reject_numbers(value, *, where: str) -> None:
    """公开输入里**不许有数值**（除了权重这种与目标无关的）。

    `weight` 是谓词权重（评分用），不泄漏目标，所以放行。
    """
    def scan(node, path):
        if isinstance(node, dict):
            for key, child in node.items():
                if key == 'weight':
                    continue
                scan(child, f"{path}.{key}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                scan(child, f"{path}[{index}]")
        elif isinstance(node, bool):
            return
        elif isinstance(node, (int, float)):
            raise ValueError(
                f"public input must not contain numeric targets: {where}{path}")
    scan(value, "")
