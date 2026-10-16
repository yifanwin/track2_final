"""Skill package validation and process-isolated execution."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable

from .legacy_util import InputError, read_json, sha256_tree
from .version import SKILL_API_VERSION


class SkillValidationError(InputError):
    """The supplied skill package does not satisfy the runner contract."""


class SkillExecutionError(RuntimeError):
    """The skill process reported an execution error."""


class SkillTimeout(TimeoutError):
    """The skill process exceeded its execution deadline."""


class ToolBudgetExceeded(RuntimeError):
    """The skill exceeded the per-episode tool-call budget."""


class SkillPackage:
    """Validated, immutable view of a skill directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise SkillValidationError(f"skill package is not a directory: {self.root}")
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise SkillValidationError(f"skill manifest not found: {manifest_path}")
        self.manifest_path = manifest_path
        self.manifest = read_json(manifest_path)
        self._validate_manifest()
        self.entrypoint = (self.root / str(self.manifest["entrypoint"])).resolve()
        if not self.entrypoint.is_file():
            raise SkillValidationError(f"skill entrypoint not found: {self.entrypoint}")
        try:
            self.entrypoint.relative_to(self.root)
        except ValueError as exc:
            raise SkillValidationError("entrypoint must be inside the skill package") from exc
        self._validate_entrypoint()
        self.package_hash = sha256_tree(self.root)

    def _validate_manifest(self) -> None:
        if not isinstance(self.manifest, dict):
            raise SkillValidationError("manifest.json must contain an object")
        required = {"skill_name", "version", "entrypoint", "api_version"}
        missing = sorted(required - set(self.manifest))
        if missing:
            raise SkillValidationError(f"manifest missing fields: {missing}")
        if self.manifest.get("api_version") != SKILL_API_VERSION:
            raise SkillValidationError(
                f"unsupported skill api_version {self.manifest.get('api_version')!r}; expected {SKILL_API_VERSION!r}"
            )
        for key in ("skill_name", "version", "entrypoint"):
            if not isinstance(self.manifest.get(key), str) or not self.manifest[key].strip():
                raise SkillValidationError(f"manifest.{key} must be a nonempty string")

    def _validate_entrypoint(self) -> None:
        try:
            source = self.entrypoint.read_text(encoding="utf-8-sig").lstrip("\ufeff")
            tree = ast.parse(source, filename=str(self.entrypoint))
        except (OSError, SyntaxError, UnicodeError) as exc:
            raise SkillValidationError(f"cannot parse skill entrypoint: {exc}") from exc
        functions = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        required = {"initialize", "run", "shutdown"}
        missing = sorted(required - functions)
        if missing:
            raise SkillValidationError(f"entrypoint missing top-level functions: {missing}")


class SkillExecutor:
    """Run a skill in a child process and route tool calls to the runner."""

    def __init__(
        self,
        package: SkillPackage,
        *,
        call_timeout_s: float = 10.0,
        workdir: str | Path | None = None,
    ):
        self.package = package
        self.call_timeout_s = float(call_timeout_s)
        self.workdir = Path(workdir).expanduser().resolve() if workdir else Path(tempfile.mkdtemp(prefix="ctrc_skill_"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.process: subprocess.Popen[str] | None = None
        self._messages: queue.Queue[str | None] = queue.Queue()
        self._stderr: list[str] = []

    @property
    def stderr_text(self) -> str:
        return "".join(self._stderr)

    def __enter__(self) -> "SkillExecutor":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        if self.process is not None:
            return
        self._messages = queue.Queue()
        self._stderr = []
        host = Path(__file__).with_name("skill_host.py")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(host), "--entrypoint", str(self.package.entrypoint)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            cwd=str(self.workdir),
            env=env,
            bufsize=1,
        )
        threading.Thread(target=self._read_stdout, name="ctrc-skill-stdout", daemon=True).start()
        threading.Thread(target=self._read_stderr, name="ctrc-skill-stderr", daemon=True).start()
        ready = self._recv(self.call_timeout_s)
        if ready.get("type") != "ready":
            raise SkillExecutionError(f"skill process did not become ready: {ready}")

    def initialize(self, context: dict[str, Any]) -> dict[str, Any]:
        return self._request({"type": "init", "context": context})

    def run(self, observation: dict[str, Any], tool_handler: Callable[[str, dict[str, Any]], Any]) -> dict[str, Any]:
        if self.process is None or self.process.stdin is None:
            raise SkillExecutionError("skill process is not running")
        self._send({"type": "run", "observation": observation})
        deadline = time.monotonic() + self.call_timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SkillTimeout(f"skill run exceeded {self.call_timeout_s}s")
            message = self._recv(remaining)
            kind = message.get("type")
            if kind == "tool_call":
                name = message.get("tool")
                arguments = message.get("arguments")
                if not isinstance(name, str) or not isinstance(arguments, dict):
                    raise SkillExecutionError("malformed tool call from skill")
                try:
                    result = tool_handler(name, arguments)
                    self._send({"type": "tool_result", "ok": True, "result": result})
                except Exception as exc:
                    self._send({"type": "tool_result", "ok": False, "error": f"{type(exc).__name__}: {exc}"})
                    raise
                continue
            if kind == "run_result":
                return message.get("result") or {}
            if kind == "error":
                raise SkillExecutionError(str(message.get("error") or "skill error"))
            raise SkillExecutionError(f"unexpected message from skill: {message}")

    def shutdown(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                self._request({"type": "shutdown"}, timeout_s=2.0)
            except Exception:
                pass
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def close(self) -> None:
        self.shutdown()
        if self.process is not None:
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        self.process = None

    def terminate(self) -> None:
        """Terminate the current skill process; the next request starts a fresh one."""
        self._terminate()

    def _request(self, payload: dict[str, Any], *, timeout_s: float | None = None) -> dict[str, Any]:
        if self.process is None:
            self.start()
        self._send(payload)
        message = self._recv(self.call_timeout_s if timeout_s is None else timeout_s)
        if message.get("type") == "error":
            raise SkillExecutionError(str(message.get("error") or "skill error"))
        return message

    def _send(self, payload: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise SkillExecutionError("skill process is not running")
        try:
            self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
        except OSError as exc:
            raise SkillExecutionError(f"cannot write to skill process: {exc}") from exc

    def _recv(self, timeout_s: float) -> dict[str, Any]:
        try:
            line = self._messages.get(timeout=max(0.0, timeout_s))
        except queue.Empty as exc:
            self._terminate()
            raise SkillTimeout(f"skill process did not respond within {timeout_s}s") from exc
        if line is None:
            raise SkillExecutionError(f"skill process exited unexpectedly. stderr: {self.stderr_text[-2000:]}")
        try:
            message = json.loads(line)
        except ValueError as exc:
            raise SkillExecutionError(f"skill process sent invalid JSON: {line!r}") from exc
        if not isinstance(message, dict):
            raise SkillExecutionError("skill process sent a non-object message")
        return message

    def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            self._messages.put(line)
        self._messages.put(None)

    def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        for line in self.process.stderr:
            self._stderr.append(line)

    def _terminate(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
