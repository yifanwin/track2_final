"""Child-process host for the ctrc-skill-v1 contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
import types
from typing import Any


def _java_object(value: Any) -> Any:
    return value


class ToolClient:
    def __init__(self, send, receive):
        self._send = send
        self._receive = receive

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        self._send({"type": "tool_call", "tool": name, "arguments": arguments or {}})
        message = self._receive()
        if message.get("type") != "tool_result":
            raise RuntimeError(f"tool broker returned unexpected message: {message}")
        if not message.get("ok"):
            raise RuntimeError(str(message.get("error") or "tool call failed"))
        return message.get("result")

    def observe(self, **kwargs) -> dict[str, Any]:
        return self.call("observe", dict(kwargs))

    # 第 2 步新契约：`joint_trajectory` / `gripper_control` 是**方法名**。
    # ⚠️ 2026-10-18 强制 Action Chunk：`joint_control` / `step` /
    #    `piper_joint_control` **已真删** —— 沙箱内调它们会 `AttributeError`。
    #    只保留夹爪的旧名（单步语义未变）。
    def joint_trajectory(self, targets):
        return self.call('joint_trajectory', {'targets': targets})
    def gripper_control(self, gripper_opening_m):
        return self.call('gripper_control', {'gripper_opening_m': gripper_opening_m})
    def piper_gripper_control(self, gripper_opening_m):
        return self.call('piper_gripper_control', {'gripper_opening_m': gripper_opening_m})


def _send(payload: dict[str, Any]) -> None:
    protocol_out.write(json.dumps(payload, ensure_ascii=False,allow_nan=False) + "\n")
    protocol_out.flush()


def _receive() -> dict[str, Any]:
    line = protocol_in.readline()
    if not line:
        raise EOFError("runner closed skill protocol")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("runner sent non-object protocol message")
    return value


def _load_module(entrypoint: Path):
    # Strip any UTF-8 BOM(s) so Windows text editors (which may write a BOM)
    # do not break the skill entrypoint at parse time.
    source = entrypoint.read_text(encoding="utf-8-sig").lstrip("\ufeff")
    module = types.ModuleType("ctrc_player_skill")
    module.__file__ = str(entrypoint)
    sys.modules["ctrc_player_skill"] = module
    code = compile(source, str(entrypoint), "exec")
    exec(code, module.__dict__)
    return module


def main() -> int:
    global protocol_in, protocol_out
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", required=True)
    args = parser.parse_args()
    entrypoint = Path(args.entrypoint).resolve()
    sys.path.insert(0, str(entrypoint.parent))

    protocol_fd = os.dup(sys.stdout.fileno())
    protocol_out = os.fdopen(protocol_fd, "w", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr
    protocol_in = sys.stdin

    try:
        module = _load_module(entrypoint)
        if not all(callable(getattr(module, name, None)) for name in ("initialize", "run", "shutdown")):
            raise RuntimeError("entrypoint must define initialize, run and shutdown")
        _send({"type": "ready", "pid": os.getpid()})
        while True:
            message = _receive()
            kind = message.get("type")
            if kind == "init":
                context = message.get("context") or {}
                module.initialize(context)
                _send({"type": "init_result", "result": {"ok": True}})
            elif kind == "run":
                tools = ToolClient(_send, _receive)
                result = module.run(message.get("observation") or {}, tools)
                _send({"type": "run_result", "result": result if isinstance(result, dict) else {"value": result}})
            elif kind == "shutdown":
                module.shutdown({})
                _send({"type": "shutdown_result", "result": {"ok": True}})
                return 0
            else:
                _send({"type": "error", "error": f"unknown protocol message: {kind}"})
    except Exception as exc:
        # Contestant exception text/tracebacks can contain arbitrary secrets or
        # noisy file paths. The judge records only a stable failure category.
        _send({"type": "error", "error": "skill execution failure", "category": type(exc).__name__})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

