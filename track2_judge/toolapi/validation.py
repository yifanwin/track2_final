"""最小 JSON Schema 校验器（stdlib-only）。

为什么自己写：本机评测环境（`env_ready`，Python 3.10）**没有 `jsonschema` 包**，
而评测运行时不允许联网装包（`check_install.py` 明确「never install packages or use network」）。

支持的关键字（刚好覆盖三个 `track2.*` ToolSpec 所需）：

    type / required / additionalProperties / properties
    items / minItems / maxItems
    minimum / maximum / exclusiveMinimum / exclusiveMaximum
    const / enum / oneOf

**不支持**（有意）：`$ref` / `allOf` / `anyOf` / `not` / `patternProperties` / `format`。
遇到不认识的关键字**不报错**（宽松跳过），因为 schema 是裁判自己写的，
真正的职责是「拦住选手里程碑式的错误输入」，不是做完整规范实现。

与前缀 `_` 的函数是内部实现；对外只有 `validate(instance, schema)` 与 `SchemaError`。
"""
from __future__ import annotations

from typing import Any

__all__ = ["SchemaError", "validate", "is_valid"]

# JSON Schema 的 type 名 -> 本实现对 Python 类型的判定
_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    # bool 是 int 的子类，必须显式排掉，否则 True 会被当成 number
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}

# 数值比较前要排掉的类型（bool 参与比较会得到意外结果）
_NON_NUMERIC = (bool,)


class SchemaError(ValueError):
    """输入不满足 schema。消息里带 JSON 路径，便于选手定位。"""

    def __init__(self, path: str, message: str):
        self.path = path
        self.message = message
        super().__init__(f"{path or '$'}: {message}")


def _type_names(schema_type: Any) -> list[str]:
    """把 `type` 归一成列表。支持 `"string"` 与 `["number","null"]` 两种写法。"""
    if isinstance(schema_type, str):
        return [schema_type]
    if isinstance(schema_type, list):
        return [item for item in schema_type if isinstance(item, str)]
    return []


def _check_type(value: Any, schema: dict, path: str) -> None:
    names = _type_names(schema.get("type"))
    if not names:
        return
    for name in names:
        check = _TYPE_CHECKS.get(name)
        if check is None:
            continue                      # 不认识的 type 名：宽松跳过
        if check(value):
            return
    raise SchemaError(path, f"expected {'|'.join(names)}, got {_kind(value)}")


def _kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _validate_number(value: Any, schema: dict, path: str) -> None:
    if isinstance(value, _NON_NUMERIC) or not isinstance(value, (int, float)):
        return
    if (bound := schema.get("minimum")) is not None and value < bound:
        raise SchemaError(path, f"must be >= {bound}")
    if (bound := schema.get("maximum")) is not None and value > bound:
        raise SchemaError(path, f"must be <= {bound}")
    if (bound := schema.get("exclusiveMinimum")) is not None and value <= bound:
        raise SchemaError(path, f"must be > {bound}")
    if (bound := schema.get("exclusiveMaximum")) is not None and value >= bound:
        raise SchemaError(path, f"must be < {bound}")


def _validate_enum(value: Any, schema: dict, path: str) -> None:
    if "const" in schema and value != schema["const"]:
        raise SchemaError(path, f"must be {schema['const']!r}")
    choices = schema.get("enum")
    if isinstance(choices, list) and value not in choices:
        raise SchemaError(path, f"must be one of {choices!r}")


def _validate_object(value: dict, schema: dict, path: str) -> None:
    for name in schema.get("required", ()) or ():
        if name not in value:
            raise SchemaError(path, f"missing required property {name!r}")

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    if schema.get("additionalProperties") is False:
        extra = sorted(set(value) - set(properties))
        if extra:
            raise SchemaError(path, f"unknown propert{'y' if len(extra) == 1 else 'ies'} {extra!r}")

    for name, item in value.items():
        sub = properties.get(name)
        if sub is None:
            continue
        child = f"{path}.{name}" if path else name
        validate(item, sub, child)


def _validate_array(value: list, schema: dict, path: str) -> None:
    if (low := schema.get("minItems")) is not None and len(value) < low:
        raise SchemaError(path, f"needs at least {low} item(s), got {len(value)}")
    if (high := schema.get("maxItems")) is not None and len(value) > high:
        raise SchemaError(path, f"needs at most {high} item(s), got {len(value)}")
    items = schema.get("items")
    if items is None:
        return
    if isinstance(items, list):           # 元组式 items（本项目的 schema 不用，宽松支持）
        for index, (item, sub) in enumerate(zip(value, items)):
            validate(item, sub, f"{path}[{index}]")
        return
    for index, item in enumerate(value):
        validate(item, items, f"{path}[{index}]")


def _validate_one_of(value: Any, schema: dict, path: str) -> None:
    branches = schema.get("oneOf")
    if not isinstance(branches, list) or not branches:
        return
    matched = 0
    for branch in branches:
        try:
            validate(value, branch, path)
        except SchemaError:
            continue
        matched += 1
    if matched != 1:
        raise SchemaError(path, f"must match exactly one of {len(branches)} variants, matched {matched}")


def validate(instance: Any, schema: Any, path: str = "") -> None:
    """按 schema 校验 instance。不通过抛 `SchemaError`；通过返回 None。

    校验顺序（先粗后细）：type → enum/const → oneOf → 分类型细化。
    """
    if not isinstance(schema, dict):
        return

    _check_type(instance, schema, path)
    _validate_enum(instance, schema, path)

    if "oneOf" in schema:
        _validate_one_of(instance, schema, path)

    if isinstance(instance, dict):
        _validate_object(instance, schema, path)
    elif isinstance(instance, list):
        _validate_array(instance, schema, path)

    _validate_number(instance, schema, path)


def is_valid(instance: Any, schema: Any) -> bool:
    """`validate` 的布尔版，供需要分支判断的场景使用。"""
    try:
        validate(instance, schema)
    except SchemaError:
        return False
    return True
