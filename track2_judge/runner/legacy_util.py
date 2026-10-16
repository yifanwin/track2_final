"""Small standard-library utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class InputError(ValueError):
    """Invalid runner input."""


def read_json(path: str | Path) -> Any:
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise InputError(f"cannot read JSON {path}: {exc}") from exc


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def sha256_tree(root: str | Path, *, exclude: set[str] | None = None) -> str:
    root = Path(root)
    excluded = exclude or set()
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel in excluded:
            continue
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def default_v3_root() -> Path:
    root = Path(__file__).resolve().parents[1] / "scoring" / "frozen_scorer"
    if (root / "mesa_eval.py").is_file():
        return root
    raise InputError("Frozen V3 scorer missing from self-contained package")


def ensure_empty_directory(path: str | Path, *, force: bool = False) -> Path:
    path = Path(path).resolve()
    if path.exists():
        if not path.is_dir():
            raise InputError(f"output path exists and is not a directory: {path}")
        if any(path.iterdir()) and not force:
            raise InputError(f"output directory must be new or empty: {path}")
        if force:
            for child in sorted(path.iterdir(), key=lambda p: p.name, reverse=True):
                if child.is_dir():
                    import shutil

                    shutil.rmtree(child)
                else:
                    child.unlink()
    else:
        path.mkdir(parents=True, exist_ok=True)
    return path


def canonical_json_hash(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(data)
