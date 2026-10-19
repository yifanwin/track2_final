"""Trusted run recorder for the v3 scorer-compatible runs schema."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .legacy_util import canonical_json_hash, sha256_tree, write_json
from .version import RUNNER_VERSION, SIMULATOR_VERSION


class TrustedRecorder:
    """Write run records and separate provenance evidence.

    The recorder receives state only from the simulator and timing/event data
    only from the runner. It never reads result fields from the skill process.
    """

    def __init__(self, output_dir: str | Path, *, skill_hash: str, suite_hash: str, runs_dir: str | Path | None = None):
        self.output_dir = Path(output_dir).resolve()
        self.runs_dir = Path(runs_dir).resolve() if runs_dir else self.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.states_dir = self.output_dir / "states"
        self.states_dir.mkdir(parents=True, exist_ok=True)
        self.skill_hash = skill_hash
        self.suite_hash = suite_hash
        self.runs: list[dict[str, Any]] = []
        self.evidence: list[dict[str, Any]] = []

    def record(
        self,
        *,
        task_index: int,
        task: dict[str, Any],
        seed: int,
        environment: str,
        initial_layout: dict[str, Any],
        final_layout: dict[str, Any],
        elapsed_s: float,
        event_violations: dict[str, int],
        verification: dict[str, Any],
        tool_calls: list[dict[str, Any]],
        status: str,
        error: str | None = None,
    ) -> dict[str, Any]:
        task_key = "".join(char if char.isalnum() else "_" for char in task["task_id"]).strip("_")[:80]
        stem = f"{task_index:03d}_{task_key}_seed{seed}"
        initial_path = self.states_dir / f"{stem}_initial.json"
        final_path = self.states_dir / f"{stem}_final.json"
        write_json(initial_path, initial_layout)
        write_json(final_path, final_layout)
        run = {
            "run_id": stem,
            "task_id": task["task_id"],
            "seed": int(seed),
            "environment": environment,
            "initial_layout": initial_path.relative_to(self.runs_dir).as_posix(),
            "final_layout": final_path.relative_to(self.runs_dir).as_posix(),
            "elapsed_s": float(elapsed_s),
            "event_violations": {key: int(value) for key, value in event_violations.items()},
        }
        self.runs.append(run)
        self.evidence.append(
            {
                "run_id": stem,
                "source": "ctrc_runner",
                "runner_version": RUNNER_VERSION,
                "simulator_version": SIMULATOR_VERSION,
                "skill_hash": self.skill_hash,
                "suite_hash": self.suite_hash,
                "task_contract_hash": canonical_json_hash(task),
                "task_id": task["task_id"],
                "seed": int(seed),
                "environment": environment,
                "status": status,
                "error": error,
                "tool_calls": deepcopy(tool_calls),
                "verification": deepcopy(verification),
            }
        )
        return run

    def finalize(self, *, suite_path: str | Path, output_path: str | Path) -> Path:
        runs_path = Path(output_path).resolve()
        write_json(runs_path, {"schema_version": 1, "runs": self.runs})
        write_json(
            self.output_dir / "evidence.json",
            {
                "schema_version": 1,
                "source": "ctrc_runner",
                "runner_version": RUNNER_VERSION,
                "simulator_version": SIMULATOR_VERSION,
                "suite": str(Path(suite_path).resolve()),
                "skill_hash": self.skill_hash,
                "suite_hash": self.suite_hash,
                "runs": self.evidence,
            },
        )
        write_json(
            self.output_dir / "runner_manifest.json",
            {
                "schema_version": 1,
                "runner_version": RUNNER_VERSION,
                "simulator_version": SIMULATOR_VERSION,
                "suite": str(Path(suite_path).resolve()),
                "run_count": len(self.runs),
                "skill_hash": self.skill_hash,
                "suite_hash": self.suite_hash,
            },
        )
        return runs_path
