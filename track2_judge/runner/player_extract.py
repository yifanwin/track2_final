"""把选手包安全解包进 runtime（第 4 步 / Q18）。

## 为什么不能直接用 `unzip`

沙箱的敌方模型是「选手提交的 ZIP **故意构造**来打到裁判侧」。所以沿用
`sandbox/package.py:unpack_skill` 那套结构性检查：不信任 ZIP 里的任何路径、
模式位、声明尺寸或压缩比，并且在**校验通过前**不往最终目录写任何东西。

## 与 `unpack_skill` 的差别：**两段预算**（Q18）

Q18 定的是「官方镜像提供依赖，选手**只带权重**」。所以 `model/` 必须有独立预算 ——
权重天然比代码大两三个数量级，用同一个 20 MiB 限会让「允许携带模型」变成空话：

| 区 | 条目数上限 | 总字节上限 | 单文件上限 | 压缩比检查 |
|---|---|---|---|---|
| 代码区（`model/` 之外） | 200 | 20 MiB | 5 MiB | ≤ 200:1 |
| `model/` 区 | 2000 | 8 GiB | 8 GiB | **不查**（权重本就不可压） |

`model/` 里放 Python 依赖**是可以的**（不禁止），但官方规则建议只放权重；
依赖版本一致性由官方镜像负责（Q18）。这里不做「必须只有权重」的强判 ——
那是规则问题，不是安全问题。

## 目录名

`model/` 指的是**包根下**的 `model/`（大小写敏感）。放在子目录里的
`foo/model/` 不享受放宽 —— 否则「包装一层躲过预算」就成了免费通道。
"""
from __future__ import annotations

import os
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

__all__ = ["PlayerPackageError", "ExtractedPlayer", "extract_player_package",
           "stage_player_package", "MODEL_DIR_NAME", "CODE_MAX_TOTAL_BYTES",
           "MODEL_MAX_TOTAL_BYTES"]

MODEL_DIR_NAME = "model"

CODE_MAX_ENTRIES = 200
CODE_MAX_TOTAL_BYTES = 20 * 1024 ** 2
CODE_SINGLE_FILE_MAX_BYTES = 5 * 1024 ** 2
CODE_MAX_RATIO = 200

MODEL_MAX_ENTRIES = 2000
MODEL_MAX_TOTAL_BYTES = 8 * 1024 ** 3
MODEL_SINGLE_FILE_MAX_BYTES = 8 * 1024 ** 3

_MANIFEST_NAME = "manifest.json"
_READ_BLOCK = 1 << 20


class PlayerPackageError(ValueError):
    """选手包不能被安全解包。消息面向选手。"""


@dataclass(frozen=True)
class ExtractedPlayer:
    """解包结果。`root` 是 manifest 所在目录。"""

    root: Path
    staging: Path
    code_bytes: int
    model_bytes: int
    entries: int

    @property
    def has_model(self) -> bool:
        return self.model_bytes > 0


def _zone(parts: tuple[str, ...], root_parts: tuple[str, ...]) -> str:
    """`model` **必须是 skill 根目录下的直接子目录** 才享受权重预算。

    ```text
    pkg/skill/model/w.bin   root=pkg/skill  → model 区  ✅
    skill/model/w.bin       root=skill      → model 区  ✅
    skill/modelx/w.bin      root=skill      → 代码区（名字相似不算）
    skill/sub/model/w.bin   root=skill      → 代码区（**不在根下**）
    skill/model             root=skill      → 目录条目本身按 model 区记账
    ```

    判据是「根下第一段 == `model`」而不是「路径里出现过 model」——
    后者会让 `skill/sub/model/` 变成绕过预算的通道。
    """
    if len(parts) <= len(root_parts):
        return "code"
    if parts[:len(root_parts)] != root_parts:
        return "code"
    return "model" if parts[len(root_parts)] == MODEL_DIR_NAME else "code"


def _root_prefix(names: list[tuple[str, ...]]) -> tuple[str, ...]:
    """从**成员名**（不读内容）推出 skill 根的路径前缀。

    取 manifest.json 所在目录。有多个候选时取**最浅**的那个 ——
    多 manifest 的包最终会被 `_locate_root` 拒掉，这里只是让预算记账有个确定口径。
    """
    roots = [parts[:-1] for parts in names if parts and parts[-1] == _MANIFEST_NAME]
    if not roots:
        return ()
    return min(roots, key=len)


def _safe_parts(name: str, *, where: str) -> tuple[str, ...]:
    """把 ZIP 成员名规约成安全的分段路径。不合格直接拒。"""
    if not name:
        raise PlayerPackageError(f"{where}: empty ZIP entry name")
    if "\\" in name:
        raise PlayerPackageError(f"{where}: backslash in ZIP entry {name!r}")
    if ":" in name:
        raise PlayerPackageError(f"{where}: colon in ZIP entry {name!r}")
    posix = PurePosixPath(name)
    if posix.is_absolute():
        raise PlayerPackageError(f"{where}: absolute ZIP entry {name!r}")
    parts = tuple(p for p in posix.parts if p not in ("", "."))
    if not parts:
        raise PlayerPackageError(f"{where}: empty ZIP entry path {name!r}")
    if ".." in parts:
        raise PlayerPackageError(f"{where}: ZIP entry escapes the package: {name!r}")
    return parts


def stage_player_package(source: str | Path, staging: str | Path) -> ExtractedPlayer:
    """把选手 ZIP 解到**新**目录 `staging` 并做结构检查。调用方负责改名/清理。

    失败时不留下部分结果：临时目录建在 `staging` 的同级，出错即整棵删掉。
    `staging` 本身**不得已存在** —— 防止「旧包残留」伪装成「新包解出来了」。
    """
    source = Path(source).resolve()
    staging = Path(staging).resolve()
    if staging.exists():
        raise PlayerPackageError(f"player staging already exists: {staging}")
    if not source.is_file():
        raise PlayerPackageError(f"player package not found: {source}")
    if source.suffix.lower() != ".zip":
        raise PlayerPackageError(
            f"player package must be a .zip, got {source.suffix!r}")

    staging.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".player-", dir=staging.parent))
    try:
        stats = _extract_into(source, temporary)
        root = _locate_root(temporary)
        return ExtractedPlayer(root=root, staging=temporary, **stats)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def extract_player_package(source: str | Path, destination: str | Path) -> ExtractedPlayer:
    """解包并**就地落盘**到 `destination`（不存在时必须新建）。

    先 `stage_player_package` 到同级临时目录，最后一步 `os.replace` ——
    所以任何失败都不会在 `destination` 留下半个包。
    """
    destination = Path(destination).resolve()
    staged = stage_player_package(source, destination)
    root = staged.root
    try:
        os.replace(staged.staging, destination)
    except BaseException:
        shutil.rmtree(staged.staging, ignore_errors=True)
        raise
    return ExtractedPlayer(root=destination / root.relative_to(staged.staging),
                           staging=destination, code_bytes=staged.code_bytes,
                           model_bytes=staged.model_bytes, entries=staged.entries)


def _extract_into(source: Path, staging: Path) -> dict[str, int]:
    counts = {"code": 0, "model": 0}
    totals = {"code": 0, "model": 0}
    entries = 0
    seen: set[str] = set()

    try:
        archive = zipfile.ZipFile(source)
    except (OSError, zipfile.BadZipFile) as exc:
        raise PlayerPackageError(f"not a readable ZIP: {exc}") from exc

    with archive:
        # Sizing pass: 预算判定要**先知道 skill 根在哪** —— 没这一步，
        # `pkg/skill/model/` 与 `skill/model/` 的判据会不一致。
        # 只读中央目录（文件名 + 声明尺寸），不读任何内容。
        sizing = [_safe_parts(m.filename.rstrip("/"), where="player package")
                  for m in archive.infolist() if m.filename.rstrip("/")]
        root_parts = _root_prefix(sizing)

        for member in archive.infolist():
            name = member.filename
            if name.endswith("/"):
                name = name.rstrip("/")
                if not name:
                    continue
            parts = _safe_parts(name, where="player package")
            key = "/".join(parts)
            if key in seen:
                raise PlayerPackageError(f"duplicate ZIP entry {key!r}")
            seen.add(key)

            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise PlayerPackageError(f"symlink in ZIP rejected: {key!r}")
            if not member.is_dir() and stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                raise PlayerPackageError(f"special file in ZIP rejected: {key!r}")

            zone = _zone(parts, root_parts)
            if member.is_dir():
                (staging.joinpath(*parts)).mkdir(parents=True, exist_ok=True)
                continue

            entries += 1
            counts[zone] += 1
            if counts[zone] > (MODEL_MAX_ENTRIES if zone == "model" else CODE_MAX_ENTRIES):
                raise PlayerPackageError(
                    f"{zone} zone exceeds "
                    f"{MODEL_MAX_ENTRIES if zone == 'model' else CODE_MAX_ENTRIES} entries")

            single_max = (MODEL_SINGLE_FILE_MAX_BYTES if zone == "model"
                          else CODE_SINGLE_FILE_MAX_BYTES)
            if member.file_size > single_max:
                raise PlayerPackageError(
                    f"{key!r} is {member.file_size} bytes; {zone} zone single-file "
                    f"limit is {single_max}")
            if zone == "code" and member.compress_size:
                if member.file_size / member.compress_size > CODE_MAX_RATIO:
                    raise PlayerPackageError(
                        f"{key!r} compression ratio rejected "
                        f"({member.file_size}/{member.compress_size})")

            totals[zone] += member.file_size
            zone_max = (MODEL_MAX_TOTAL_BYTES if zone == "model"
                        else CODE_MAX_TOTAL_BYTES)
            if totals[zone] > zone_max:
                raise PlayerPackageError(
                    f"{zone} zone exceeds {zone_max} bytes "
                    f"(running total {totals[zone]})")

            target = staging.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            try:
                with archive.open(member) as src, target.open("xb") as out:
                    while True:
                        block = src.read(_READ_BLOCK)
                        if not block:
                            break
                        written += len(block)
                        # 声明尺寸可以撒谎 —— 以实际写出的字节为准
                        if written > single_max:
                            raise PlayerPackageError(
                                f"{key!r} exceeds its declared size "
                                f"(actual > {single_max})")
                        out.write(block)
            except PlayerPackageError:
                raise
            except OSError as exc:
                raise PlayerPackageError(f"cannot write {key!r}: {exc}") from exc
            if written != member.file_size:
                raise PlayerPackageError(
                    f"{key!r}: declared {member.file_size} bytes, got {written}")

    return {"code_bytes": totals["code"], "model_bytes": totals["model"],
            "entries": entries}


def _locate_root(staging: Path) -> Path:
    """找到**唯一**的 skill 根：manifest 所在目录，其余文件必须都在它下面。

    与 `sandbox/package.py` 同向 —— 防「包里藏第二个 skill」。
    """
    manifests = sorted(staging.rglob(_MANIFEST_NAME))
    if len(manifests) != 1:
        raise PlayerPackageError(
            f"exactly one {_MANIFEST_NAME} required at the package root, "
            f"found {len(manifests)}")
    root = manifests[0].parent
    stray = [p for p in staging.rglob("*")
             if p.is_file() and root not in p.parents]
    if stray:
        raise PlayerPackageError(
            f"{len(stray)} file(s) outside the skill root "
            f"({stray[0].relative_to(staging)}) — one skill per package")
    return root
