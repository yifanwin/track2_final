"""选手包 manifest 校验（第 4 步）。

## 选手契约

```json
{
  "skill_name": "my-grasp-skill",
  "version": "1.0.0",
  "entrypoint": "entrypoint.py",
  "api_version": "ctrc-skill-v1",
  "tool_profile": ["track2.env.observe", "track2.robot.joint_control"]
}
```

相对旧契约只多了一件事：

| 字段 | 含义 | 谁看 |
|---|---|---|
| `tool_profile` | 这个包要用哪些 Tool（**Tool ID 列表**） | 准入校验 |

## `requires_gpu` 已删除（2026-10-17）

旧契约里还有一个 `requires_gpu` 布尔，用来在两个沙箱档之间选。**现在只有一档**
—— 所有选手包都在官方 runtime 里跑（Python 3.12.14 + 官方预装 ML 依赖 + 一张 GPU），
所以这个字段没有意义了。

它现在**明确拒绝**（不静默忽略），理由与 `tool_profile` 的旧字符串形态一样：
静默接受会让选手以为自己选的档生效了，实际跑的是另一套环境。

**取消两档的代价说清**：放弃了「不带模型的选手享受更强隔离」这条设计。
隔离本身没有退化（独立 mount/pid/user namespace + seccomp + 只读挂载 +
看不到宿主与裁判目录 + 网络隔离），只是**不再对选手区分强弱**。

## 为什么 `tool_profile` 是列表而不是字符串

第 1.7 步删掉了旧的 `tool_profile` **字符串门禁**（`"ctrc-skill-v1"` 之类），
改为「**Tool 集合 ⊆ 官方 `required_tools`**」。字符串没有表达能力，判不了准入。
因此旧字符串形态**明确拒绝**，并给出可操作的提示 —— 不做静默兼容（Q16）。

## 这个模块放在裁判侧（`track2_judge`）的原因

它与 `toolapi/schema.py` 是同一类东西：**契约的唯一权威**。
`start.sh`（注入时）与 player 节点（运行时）都 import 它，不各写一份。
所以它依赖 `TRACK2_PHYSICS_ROOT`（与 Bundle 里 `endpoints.py` 依赖
`track2_judge.toolapi.schema` 是同一模式）。
"""
from __future__ import annotations

import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from track2_judge.runner.version import SKILL_API_VERSION
from track2_judge.toolapi import schema as schema_mod

__all__ = [
    "PlayerManifestError", "PlayerManifest", "validate_player_dir",
    "validate_manifest_dict", "REQUIRED_TOOL_IDS", "ADMIN_TOOL_PREFIX",
    "REMOVED_FIELDS",
]

# 选手**只能**声明这三个。权威来源是 schema.py —— 不在这里抄一遍。
REQUIRED_TOOL_IDS: tuple[str, ...] = tuple(schema_mod.REQUIRED_TOOLS)

# 裁判侧 Tool 的前缀。选手 manifest 里出现任何一个都直接拒。
ADMIN_TOOL_PREFIX = "track2.admin."

#: 已从契约里删掉的字段 → 给选手的提示。出现即拒，**不静默忽略** ——
#: 静默接受会让选手以为自己的声明生效了。
REMOVED_FIELDS: dict[str, str] = {
    "requires_gpu": (
        "all contestant packages now run in the same official sandbox "
        "(Python 3.12.14, the official ML dependencies and one GPU are always "
        "available). Delete this line — there is no replacement field."
    ),
}

_MANIFEST_NAME = "manifest.json"
_MANIFEST_MAX_BYTES = 64 * 1024
_ENTRYPOINT_MAX_BYTES = 128 * 1024
_REQUIRED_FIELDS = ("skill_name", "version", "entrypoint", "api_version")
_REQUIRED_FUNCTIONS = ("initialize", "run", "shutdown")


class PlayerManifestError(ValueError):
    """选手包不满足契约。消息**面向选手** —— 要能直接行动。"""


@dataclass(frozen=True)
class PlayerManifest:
    """校验通过的选手 manifest。"""

    root: Path
    skill_name: str
    version: str
    api_version: str
    entrypoint: Path
    tool_profile: tuple[str, ...]

    @property
    def relative_entrypoint(self) -> str:
        return self.entrypoint.relative_to(self.root).as_posix()


# ─────────────────────────────────────────────── 单字段校验


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlayerManifestError(f"manifest.{field} must be a non-empty string")
    return value.strip()


def _validate_tool_profile(value: object, *, where: str) -> tuple[str, ...]:
    """校验 `tool_profile`。

    **旧形态（字符串）明确拒绝** —— Q16 不做兼容。给可操作提示，不静默接受，
    否则选手会以为声明生效了，实际拿不到任何准入。
    """
    if isinstance(value, str):
        raise PlayerManifestError(
            f"{where}: tool_profile must be a LIST of Tool IDs, got the string "
            f"{value!r}. The legacy string form is not supported (Q16 single track). "
            f"Use e.g. [\"{REQUIRED_TOOL_IDS[0]}\", \"{REQUIRED_TOOL_IDS[1]}\"]."
        )
    if not isinstance(value, list):
        raise PlayerManifestError(
            f"{where}: tool_profile must be a list of Tool IDs")
    if not value:
        raise PlayerManifestError(
            f"{where}: tool_profile is empty — a skill that calls no Tool "
            "cannot complete any task")

    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise PlayerManifestError(
                f"{where}: every entry of tool_profile must be a non-empty string")
        tool_id = item.strip()
        if tool_id in seen:
            raise PlayerManifestError(
                f"{where}: tool_profile lists {tool_id!r} twice")
        seen.add(tool_id)
        if tool_id.startswith(ADMIN_TOOL_PREFIX):
            raise PlayerManifestError(
                f"{where}: {tool_id!r} is a judge-only Tool and can never be "
                "declared by a contestant skill")
        if tool_id not in REQUIRED_TOOL_IDS:
            raise PlayerManifestError(
                f"{where}: {tool_id!r} is not an official Tool. "
                f"Available: {', '.join(REQUIRED_TOOL_IDS)}")
    # 按官方顺序规范化 —— 让集合比较与证据稳定
    return tuple(t for t in REQUIRED_TOOL_IDS if t in seen)


def _reject_removed_fields(data: dict, *, where: str) -> None:
    """已删掉的字段出现即拒 —— 不做静默忽略。

    `requires_gpu` 曾经决定沙箱档。现在只有一档，再收这个字段意味着选手
    以为自己选的档生效了。**明确失败**比「接受了但没用」好。
    """
    for field, hint in REMOVED_FIELDS.items():
        if field in data:
            raise PlayerManifestError(f"{where}: manifest.{field} was removed. {hint}")


def _contained_entrypoint(root: Path, entry: object, *, where: str) -> Path:
    """词法先行：**先判包含性，再碰文件系统**。

    与 `sandbox/package.py` 同一个理由 —— 恶意 manifest 不能借
    `is_file()` 的成功/失败当作探测裁判路径是否存在的信道。
    """
    text = _text(entry, "entrypoint")
    if "\\" in text or ":" in text:
        raise PlayerManifestError(f"{where}: manifest.entrypoint must use POSIX paths")
    posix = PurePosixPath(text)
    if posix.is_absolute() or ".." in posix.parts or not posix.parts:
        raise PlayerManifestError(
            f"{where}: manifest.entrypoint must stay inside the skill package")
    candidate = root.joinpath(*posix.parts)
    if candidate.is_symlink():
        raise PlayerManifestError(f"{where}: manifest.entrypoint must not be a symlink")
    if not candidate.is_file():
        raise PlayerManifestError(f"{where}: manifest.entrypoint not found: {text}")
    if candidate.stat().st_size > _ENTRYPOINT_MAX_BYTES:
        raise PlayerManifestError(
            f"{where}: entrypoint exceeds {_ENTRYPOINT_MAX_BYTES} bytes")
    return candidate


def _validate_entrypoint_source(path: Path, *, where: str) -> None:
    """entrypoint 必须能解析，且定义三个生命周期函数。

    与 `runner/skill_manifest.py` 的 `_validate_entrypoint` 同一套要求 ——
    失败要在**注入时**就暴露，而不是跑起来才炸。
    """
    try:
        source = path.read_text(encoding="utf-8-sig").lstrip("﻿")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise PlayerManifestError(f"{where}: cannot parse entrypoint: {exc}") from exc
    defined = {node.name for node in tree.body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    missing = sorted(set(_REQUIRED_FUNCTIONS) - defined)
    if missing:
        raise PlayerManifestError(
            f"{where}: entrypoint must define top-level functions: "
            f"{', '.join(_REQUIRED_FUNCTIONS)} (missing: {', '.join(missing)})")


# ─────────────────────────────────────────────── 入口


def validate_manifest_dict(data: object, root: Path, *, where: str = "player") -> PlayerManifest:
    """校验一份已解析的 manifest，`root` 是选手包根目录。"""
    root = Path(root).resolve()
    if not isinstance(data, dict):
        raise PlayerManifestError(f"{where}: manifest.json must contain an object")

    missing = sorted(set(_REQUIRED_FIELDS) - set(data))
    if missing:
        raise PlayerManifestError(
            f"{where}: manifest missing fields: {', '.join(missing)}")

    # 已删字段要在**其余校验之前**拒 —— 否则选手会先看到别的错误，
    # 改完才发现 `requires_gpu` 也留不得。
    _reject_removed_fields(data, where=where)

    api_version = _text(data["api_version"], "api_version")
    if api_version != SKILL_API_VERSION:
        raise PlayerManifestError(
            f"{where}: unsupported api_version {api_version!r}; "
            f"expected {SKILL_API_VERSION!r}")

    entrypoint = _contained_entrypoint(root, data["entrypoint"], where=where)
    _validate_entrypoint_source(entrypoint, where=where)

    return PlayerManifest(
        root=root,
        skill_name=_text(data["skill_name"], "skill_name"),
        version=_text(data["version"], "version"),
        api_version=api_version,
        entrypoint=entrypoint,
        tool_profile=_validate_tool_profile(data.get("tool_profile"), where=where),
    )


def validate_player_dir(root: str | Path, *, where: str | None = None) -> PlayerManifest:
    """校验一个**已解包**的选手目录。

    会先做两条结构性检查（单一 manifest、无符号链接），再校验 manifest ——
    与 `sandbox/package.py:unpack_skill` 的检查同向，只是这里针对已在磁盘上的目录。
    """
    root = Path(root).resolve()
    where = where or f"player {root}"
    if not root.is_dir():
        raise PlayerManifestError(f"{where}: not a directory: {root}")

    if any(p.is_symlink() for p in root.rglob("*")):
        raise PlayerManifestError(f"{where}: symlinks are not allowed in a skill package")

    manifests = sorted(root.rglob(_MANIFEST_NAME))
    if len(manifests) != 1:
        raise PlayerManifestError(
            f"{where}: exactly one {_MANIFEST_NAME} required, found {len(manifests)}")
    manifest_path = manifests[0]
    if manifest_path.stat().st_size > _MANIFEST_MAX_BYTES:
        raise PlayerManifestError(
            f"{where}: manifest.json exceeds {_MANIFEST_MAX_BYTES} bytes")

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, RecursionError) as exc:
        raise PlayerManifestError(f"{where}: manifest.json is not valid JSON: {exc}") from exc

    # ⚠️ 根目录**以 manifest 所在目录为准**，不是传进来的目录。
    # 允许选手把包打成 `pkg/<skill>/manifest.json` 这种多一层的形式，
    # 同时也让「manifest 之外还有文件」这件事无处藏身（下面这条检查兜住）。
    package_root = manifest_path.parent
    if package_root != root:
        stray = [p for p in root.rglob("*") if p.is_file()
                 and package_root not in p.parents]
        if stray:
            raise PlayerManifestError(
                f"{where}: {len(stray)} file(s) outside the skill root "
                f"({stray[0].relative_to(root)}) — one skill per package")

    return validate_manifest_dict(data, package_root, where=where)


def main(argv: list[str] | None = None) -> int:
    """给 `start.sh` 用的命令行入口：成功打一行摘要，失败非零退出。"""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: python -m track2_judge.runner.player_manifest <player_dir>",
              file=sys.stderr)
        return 2
    try:
        manifest = validate_player_dir(args[0])
    except PlayerManifestError as exc:
        print(f"[player-manifest] REJECTED: {exc}", file=sys.stderr)
        return 1
    print(f"[player-manifest] OK skill={manifest.skill_name} "
          f"version={manifest.version} entrypoint={manifest.relative_entrypoint} "
          f"tools={','.join(manifest.tool_profile)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
