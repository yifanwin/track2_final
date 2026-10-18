"""Offline MesaTask CTRC evaluator. Python 3.10+, standard library only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from fractions import Fraction

VERSION = "3.0.0"
POLICY_ID = "mesa-ctrc-v3"
MOVED_THRESHOLD_CM = 1.0
BASELINE = {"completion_threshold": .6, "time_ref_s": 90.0, "violation_cap": 1.0}
REPEATS = {"sim": 5, "real": 3}
REPLAN_BUDGET = 2
TIERS = {"loose": (5, 5, 15, 3), "standard": (3, 4, 10, 2), "strict": (2, 3, 5, 1)}
# 2026-10-18 组委会口径变更：取消「执行效率 E」项，其 10% 权重并入 SR_final
#   （0.50 → 0.60）。时间不再参与任何计分，只作资源/异常保护（600 s run() 墙钟）。
#   见 report/requirement/Track2_评分口径变更_效率项改超时.md §1.2（读法 A）。
WEIGHTS = {"final": .60, "first": .20, "constraints": .15, "stability": .05}
EVENTS = {"collision", "forbidden_zone"}


class InputError(ValueError):
    """Invalid or incomplete evaluation inputs; no ranking should be issued."""


def require(condition, message):
    if not condition:
        raise InputError(message)


def report_meta():
    return {"schema_version": 1, "evaluator_version": VERSION,
            "policy_id": POLICY_ID, "policy_status": "project_frozen"}


def validate_meta(value):
    require(type(value.get("schema_version")) is int, "schema_version must be integer 1")
    require(all(value.get(k) == v for k, v in report_meta().items()),
            "expected v3.0.0 / mesa-ctrc-v3 project_frozen metadata; regenerate legacy contracts")


def moved_policy(threshold_cm=MOVED_THRESHOLD_CM):
    require(number(threshold_cm, "moved_threshold_cm", 0) > 0, "moved_threshold_cm must be positive")
    return {"mode": "position_distance_gte", "threshold_cm": threshold_cm,
            "rotation_selects_targets": False}


def validate_moved_policy(policy):
    require(isinstance(policy, dict), "missing moved_policy; regenerate suite with v3 prepare")
    expected = moved_policy(policy.get("threshold_cm"))
    require(policy == expected and type(policy.get("rotation_selects_targets")) is bool,
            "unsupported moved policy; v3 selects by position only")


def allowed_fields(value, fields, label):
    require(isinstance(value, dict), f"{label}: expected object")
    require(not (set(value) - set(fields)), f"{label}: unsupported fields {sorted(set(value) - set(fields))}")


def number(value, label, minimum=None, maximum=None):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    require(valid, f"{label}: expected finite number")
    require(minimum is None or value >= minimum, f"{label}: below minimum {minimum}")
    require(maximum is None or value <= maximum, f"{label}: above maximum {maximum}")
    return value


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, f"{label}: expected integer >= {minimum}")
    number(value, label)
    return value


def vector(value, length, label):
    require(isinstance(value, list) and len(value) == length, f"{label}: expected {length} numbers")
    return [number(x, label) for x in value]


def quaternion(value):
    q = vector(value, 4, "rotation xyzw")
    norm = math.hypot(*q)
    require(math.isfinite(norm) and norm > 1e-12, "rotation: zero or invalid quaternion")
    return [v / norm for v in q]


def rotation_error_deg(a, b):
    # Align signs, then use chord lengths to avoid acos cancellation near zero.
    a, b = quaternion(a), quaternion(b)
    if sum(x * y for x, y in zip(a, b)) < 0:
        b = [-value for value in b]
    return math.degrees(4 * math.atan2(math.hypot(*(x-y for x, y in zip(a, b))),
                                     math.hypot(*(x+y for x, y in zip(a, b)))))


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    try:
        with Path(path).open(encoding="utf-8-sig") as f:
            return json.load(f, object_pairs_hook=no_duplicates,
                             parse_constant=lambda x: (_ for _ in ()).throw(InputError(f"invalid JSON number: {x}")))
    except (OSError, UnicodeError, ValueError) as exc:
        raise InputError(f"cannot read valid JSON {path}: {exc}") from exc


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def object_map(layout, reference=None):
    """Return canonical poses in scene-local centimeters; never use bbox centers."""
    require(isinstance(layout, dict), "layout must be a JSON object")
    frame = layout.get("coordinate_frame", "scene_local_cm")
    require(frame in ("scene_local_cm", "world_m"), "unknown coordinate_frame")
    settings = layout.get("scene_settings")
    require(settings is not None or "coordinate_frame" in layout, "observation requires scene_settings or explicit coordinate_frame")
    if reference is None:
        require(frame == "scene_local_cm" and isinstance(settings, dict), "dataset layout requires scene_settings and local cm")
    if settings is not None:
        require(isinstance(settings, dict), "scene_settings must be an object")
        require(settings.get("units") == ("meters" if frame == "world_m" else "centimeters"), "incorrect units for coordinate_frame")
        require(settings.get("up_axis") == "Z", "only Z-up is supported")
        require(settings.get("rotation_convention") == "absolute_quaternion_xyzw", "only absolute xyzw quaternions are supported")
        offset = vector(settings.get("scene_offset", [0, 0, 0]), 3, "scene_offset (meters)")
        if reference is not None and "scene_offset" in settings and "scene_offset" in reference["scene_settings"]:
            ref_offset = reference["scene_settings"].get("scene_offset", [0, 0, 0])
            require(all(abs(a-b) <= 1e-9 for a, b in zip(offset, ref_offset)), "scene_offset differs from target; do not silently change frames")
    if frame == "world_m":
        require(reference is not None, "world coordinates need a target reference")
        require("scene_offset" in reference["scene_settings"], "world_m requires an explicit target scene_offset; cannot assume zero")
    objects = layout.get("objects")
    require(isinstance(objects, list), "objects must be a list")
    if reference is None:
        require(objects, "dataset objects must be nonempty")
    result = {}
    for obj in objects:
        require(isinstance(obj, dict), "object entry must be an object")
        key = obj.get("instance")
        require(isinstance(key, str) and key.strip(), "object instance must be a nonempty string")
        require(key not in result, f"duplicate instance: {key}")
        pos = vector(obj.get("position"), 3, f"{key}.position")
        if frame == "world_m":
            offset = reference["scene_settings"].get("scene_offset", [0, 0, 0])
            pos = [(x - off) * 100 for x, off in zip(pos, offset)]
            vector(pos, 3, f"{key}.converted position")
        entry = {"position": pos, "rotation": quaternion(obj.get("rotation"))}
        if "size" in obj:
            entry["size"] = vector(obj["size"], 3, f"{key}.size (cm)")
            require(all(x > 0 for x in entry["size"]), f"{key}: size must be positive")
        for field in ("selected_uid", "resolved_glb"):
            if field in obj:
                require(isinstance(obj[field], str), f"{key}.{field}: expected string")
                entry[field] = obj[field]
        result[key] = entry
    return result


def check_identity(actual, target, complete=False):
    require(not (actual.keys() - target.keys()), f"unknown instances: {sorted(actual.keys() - target.keys())}")
    if complete:
        require(actual.keys() == target.keys(), "initial and target object sets must match exactly")
    for key, obj in actual.items():
        for field in ("selected_uid", "resolved_glb", "size"):
            if field in obj and field in target[key]:
                require(obj[field] == target[key][field], f"{key}: changed {field}; only poses may change")


def make_task(task_id, target, initial, scope="moved", moved_threshold_cm=MOVED_THRESHOLD_CM):
    require(scope in ("moved", "all"), "unknown scope")
    moved_policy(moved_threshold_cm)
    goal = object_map(target)
    start = object_map(initial, target)
    check_identity(start, goal, complete=True)
    moved = []
    for key in sorted(goal):
        distance = number(math.dist(goal[key]["position"], start[key]["position"]), "target-initial distance", 0)
        if distance >= moved_threshold_cm:
            moved.append(key)
    selected = sorted(goal) if scope == "all" else moved
    require(selected, f"{task_id}: no changed objects; cannot create an empty contract")
    return {"task_id": task_id, "target_layout": target, "initial_layout": initial,
            "predicates": [{"type": "P2", "object": key, "weight": 1.0} for key in selected],
            "moved_objects": moved}


def default_config(environment="sim", tier="standard"):
    require(isinstance(environment, str) and environment in REPEATS
            and isinstance(tier, str) and tier in TIERS, "unknown environment/tier")
    sim, real, rotation, delta = TIERS[tier]
    return {"environment": environment, "tier": tier, "repeats": REPEATS[environment],
            "eps_pos_cm": sim if environment == "sim" else real, "eps_rot_deg": rotation,
            "delta_max_cm": delta, **BASELINE, "replan_budget": REPLAN_BUDGET,
            "non_target_displacement": True,
            "event_constraints": [], "weights": WEIGHTS.copy(),
            "auxiliary_mode": "final", "stdev": "population", "constraint_score_policy": "penalty_only"}


def validate_config(config):
    require(isinstance(config, dict), "config must be an object")
    allowed_fields(config, default_config(), "config")
    require(config.get("environment") in ("sim", "real"), "environment must be sim or real")
    require(isinstance(config.get("tier"), str) and config["tier"] in TIERS, "unknown tolerance tier")
    integer(config.get("repeats"), "repeats", 1)
    require(config["repeats"] == REPEATS[config["environment"]],
            "competition repeats must be 5 for sim or 3 for real")
    integer(config.get("replan_budget"), "replan_budget")
    require(config["replan_budget"] == REPLAN_BUDGET, "competition replan budget must be 2")
    for key in ("eps_pos_cm", "delta_max_cm"):
        number(config.get(key), key, 0)
    number(config.get("eps_rot_deg"), "eps_rot_deg", 0, 180)
    number(config.get("completion_threshold"), "completion_threshold", 0, 1)
    for key in ("time_ref_s", "violation_cap"):
        require(number(config.get(key), key, 0) > 0, f"{key} must be positive")
    require(type(config.get("non_target_displacement")) is bool, "non_target_displacement must be boolean")
    events = config.get("event_constraints")
    require(isinstance(events, list) and all(isinstance(x, str) for x in events), "event_constraints must be a list of strings")
    require(len(events) == len(set(events)) and set(events) <= EVENTS, "unsupported or duplicate event constraint")
    require(config.get("weights") == WEIGHTS, "competition weights must match source formula")
    require(config.get("auxiliary_mode") == "final" and config.get("stdev") == "population"
            and config.get("constraint_score_policy") == "penalty_only", "unsupported scoring policy")


def validate_task(task):
    require(isinstance(task, dict), "task must be an object")
    require(not (set(task) - {"task_id", "target_layout", "initial_layout", "predicates", "moved_objects"}),
            "unsupported task fields; per-task overrides/constraints are not implemented; use the documented suite config")
    require(isinstance(task.get("task_id"), str) and task["task_id"].strip(), "missing task_id")
    goal = object_map(task.get("target_layout"))
    start = object_map(task.get("initial_layout"), task["target_layout"])
    check_identity(start, goal, complete=True)
    predicates = task.get("predicates")
    require(isinstance(predicates, list) and predicates, "empty predicates")
    seen = set()
    for p in predicates:
        require(isinstance(p, dict), "predicate must be an object")
        require(not (set(p) - {"type", "object", "weight"}),
                "unsupported predicate fields; targets come from target_layout and tolerances from suite config")
        require(p.get("type") == "P2", "this CTRC evaluator supports P2 pose predicates only")
        require(isinstance(p.get("object"), str) and p["object"] in goal and p["object"] not in seen, "unknown or duplicate predicate object")
        seen.add(p["object"])
        require(number(p.get("weight"), "predicate weight", 0) > 0, "predicate weight must be positive")


def score_state(task, actual_layout, initial_layout, config, event_counts=None):
    validate_config(config)
    validate_task(task)
    reference = task["target_layout"]
    goal = object_map(reference)
    actual = object_map(actual_layout, reference)
    initial = object_map(initial_layout, reference)
    check_identity(initial, goal, complete=True)
    check_identity(actual, goal)
    selected = {p["object"] for p in task["predicates"]}
    details = []
    for p in task["predicates"]:
        key = p["object"]
        detail = {"type": "P2", "object": key, "weight": p["weight"]}
        if key not in actual:
            detail.update(status="inconclusive", passed=False, reason="object missing from final observation")
        else:
            dist = math.dist(actual[key]["position"], goal[key]["position"])
            number(dist, "position distance", 0)
            angle = rotation_error_deg(actual[key]["rotation"], goal[key]["rotation"])
            passed = dist <= config["eps_pos_cm"] and angle <= config["eps_rot_deg"]
            detail.update(status="success" if passed else "failure", passed=passed,
                          position_error_cm=dist, rotation_error_deg=angle)
        details.append(detail)
    # Exact sums of the supplied numeric weights; round the ratio only once.
    # Avoids overflow and the 3/(3+1+1) -> .6000000000000001 scaling artifact.
    completion = float(sum(Fraction(p["weight"]) for p in details if p["passed"])
                       / sum(Fraction(p["weight"]) for p in details))
    score = completion if completion >= config["completion_threshold"] else 0.0
    displacement_violations = []
    if config["non_target_displacement"]:
        for key in sorted(goal.keys() - selected):
            if key not in actual:
                displacement_violations.append({"object": key, "reason": "missing non-target object"})
            else:
                dist = math.dist(actual[key]["position"], initial[key]["position"])
                number(dist, "non-target displacement", 0)
                if dist > config["delta_max_cm"]:
                    displacement_violations.append({"object": key, "displacement_cm": dist})
    # A10: each run's initial state is the C1 reference, including randomized backgrounds.
    event_counts = {} if event_counts is None else event_counts
    require(isinstance(event_counts, dict), "event_violations must be an object")
    require(set(event_counts) == set(config["event_constraints"]), "event_violations must exactly cover enabled event_constraints")
    for event, count in event_counts.items():
        integer(count, f"event_violations.{event}")
    violations = len(displacement_violations) + sum(event_counts.values())
    strict = all(p["passed"] for p in details) and violations == 0
    return {"completion": completion, "s": score, "strict_success": strict,
            "status": "success" if strict else ("inconclusive" if any(p["status"] == "inconclusive" for p in details) else "failure"),
            "predicates": details, "violation_count": violations,
            "displacement_violations": displacement_violations, "event_violations": event_counts}


def discover(root):
    root = Path(root).resolve()
    require((root / "scenes").is_dir(), f"{root}: expected scenes/ directory")
    layouts = {p.parent.relative_to(root / "scenes").as_posix(): p
               for p in sorted((root / "scenes").rglob("layout.json"))}
    require(layouts, f"no layout.json found under {root}")
    return layouts


def prepare(args):
    targets, initials = discover(args.target_root), discover(args.shuffled_root)
    require(targets.keys() == initials.keys(), "target/shuffled scene paths do not match exactly")
    require(len(targets) == args.expected_scenes, f"expected {args.expected_scenes} scenes, found {len(targets)}")
    config = default_config(args.environment, args.tier)
    config.update(time_ref_s=args.time_ref_s, violation_cap=args.violation_cap,
                  completion_threshold=args.completion_threshold, event_constraints=args.event_constraints)
    for key in ("eps_pos_cm", "eps_rot_deg", "delta_max_cm"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    validate_config(config)
    tasks = [make_task(key, read_json(targets[key]), read_json(initials[key]), args.scope, args.moved_threshold_cm) for key in targets]
    suite = {**report_meta(), "kind": "mesa_ctrc_suite", "scope": args.scope,
             "moved_policy": moved_policy(args.moved_threshold_cm), "config": config, "tasks": tasks}
    write_json(args.output, suite)
    return {**report_meta(), "saved": str(Path(args.output).resolve()), "tasks": len(tasks),
            "moved_policy": suite["moved_policy"], "selected_objects": sum(len(t["predicates"]) for t in tasks),
            "note": "v3.0 contracts: position-only target selection; rotation remains part of P2 scoring."}


def layout_command(args):
    target = read_json(args.target)
    initial = read_json(args.initial) if args.initial else target
    require(args.initial or args.scope == "all", "--scope moved requires --initial")
    task = make_task("single_scene", target, initial, args.scope, args.moved_threshold_cm)
    config = default_config(args.environment, args.tier)
    report = {**report_meta(), "kind": "layout_diagnostic", "config": config,
              "scope": args.scope, "moved_policy": moved_policy(args.moved_threshold_cm),
              "result": score_state(task, read_json(args.actual), initial, config),
              "note": "Pose diagnostic only. No time, recovery, collision or full competition score inferred."}
    if args.output:
        write_json(args.output, report)
    return report


def evaluation_identity(suite):
    """Cross-environment identity excludes the deliberately different sim/real tolerances."""
    return fingerprint({"scope": suite["scope"], "moved_policy": suite["moved_policy"],
                        "tasks": sorted(suite["tasks"], key=lambda t: t["task_id"])})


def summarize(first, final, config):
    # Also protect direct API callers; evaluate validates these values upstream.
    # Trial counts are checked by evaluate; this helper permits small math fixtures.
    validate_config(config)
    for mode, runs in (("first", first), ("final", final)):
        require(isinstance(runs, (list, tuple)) and runs, "summary groups must be nonempty sequences")
        for run in runs:
            require(isinstance(run, dict), "summary run must be an object")
            number(run.get("s"), "s", 0, 1)
            integer(run.get("violation_count"), "violation_count")
            require(number(run.get("elapsed_s"), "elapsed_s", 0) > 0, "elapsed_s must be positive")
            revisions = integer(run.get("plan_revisions"), "plan_revisions")
            require(revisions <= (0 if mode == "first" else config["replan_budget"]), "replan budget exceeded")
            require(type(run.get("strict_success")) is bool, "strict_success must be boolean")
    sr_first = statistics.mean(r["s"] for r in first)
    sr_final = statistics.mean(r["s"] for r in final)
    violation = min(1.0, statistics.mean(r["violation_count"] for r in final) / config["violation_cap"])
    # 2026-10-18：E 项取消。`elapsed_s` **仍是** run 的必填结构字段（诊断用），
    # 但不再进任何 component —— 上面那条 `elapsed_s > 0` 的校验保留。
    sigma = statistics.pstdev(r["s"] for r in final)
    components = {"final": WEIGHTS["final"] * sr_final, "first": WEIGHTS["first"] * sr_first,
                  "constraints": WEIGHTS["constraints"] * (1-violation),
                  "stability": WEIGHTS["stability"] * (1-sigma)}
    return {"SR_first": sr_first, "SR_final": sr_final, "V": violation, "sigma": sigma,
            "N_rev": statistics.mean(r["plan_revisions"] for r in final),
            "strict_success_rate_first": statistics.mean(int(r["strict_success"]) for r in first),
            "strict_success_rate_final": statistics.mean(int(r["strict_success"]) for r in final),
            "components": components, "S": sum(components.values())}


def evaluate(suite, submission, base_dir):
    require(isinstance(suite, dict) and isinstance(submission, dict), "suite and submission must be objects")
    allowed_fields(suite, {*report_meta(), "kind", "scope", "config", "tasks", "synthetic_demo_only", "moved_policy"}, "suite")
    allowed_fields(submission, {"schema_version", "runs", "synthetic_demo_only", "baseline_diagnostic_only"}, "submission")
    validate_meta(suite)
    validate_moved_policy(suite.get("moved_policy"))
    for flag in ("synthetic_demo_only", "baseline_diagnostic_only"):
        for value in (suite, submission):
            require(flag not in value or type(value[flag]) is bool, f"{flag} must be boolean")
    require(suite.get("schema_version") == 1 and suite.get("kind") == "mesa_ctrc_suite", "unsupported suite")
    config = suite.get("config")
    validate_config(config)
    require(suite.get("scope") in ("all", "moved"), "invalid suite scope")
    require(isinstance(suite.get("tasks"), list) and suite["tasks"], "suite tasks must be nonempty")
    tasks = {}
    for task in suite["tasks"]:
        validate_task(task)
        generated = make_task(task["task_id"], task["target_layout"], task["initial_layout"],
                              suite["scope"], suite["moved_policy"]["threshold_cm"])
        require({p["object"] for p in task["predicates"]} == {p["object"] for p in generated["predicates"]},
                "predicate objects disagree with frozen scope/moved_policy; regenerate suite")
        require(task.get("moved_objects", generated["moved_objects"]) == generated["moved_objects"],
                "moved_objects disagrees with frozen moved_policy")
        require(task["task_id"] not in tasks, "duplicate suite task_id")
        tasks[task["task_id"]] = task
    require(submission.get("schema_version") == 1, "unsupported submission schema_version")
    require(type(submission.get("schema_version")) is int, "submission schema_version must be integer 1")
    runs = submission.get("runs")
    require(isinstance(runs, list), "submission.runs must be a list")
    require(len(runs) == len(tasks) * config["repeats"] * 2, "wrong run count: every task needs both independent modes and R seeds")
    groups = {key: {"first": {}, "final": {}} for key in tasks}
    initial_poses = {}
    run_ids = set()
    evidence = []
    for run in runs:
        require(isinstance(run, dict), "run must be an object")
        allowed_fields(run, {"run_id", "task_id", "mode", "seed", "environment", "initial_layout",
                             "final_layout", "elapsed_s", "plan_revisions", "event_violations"}, "run")
        task_id, mode, seed = run.get("task_id"), run.get("mode"), run.get("seed")
        require(isinstance(task_id, str) and task_id in tasks, "unknown run task_id")
        require(mode in ("first", "final"), "mode must be first or final")
        integer(seed, "seed")
        run_id = run.get("run_id")
        require(isinstance(run_id, str) and run_id.strip() and run_id not in run_ids, "run_id must be globally unique")
        run_ids.add(run_id)
        require(run.get("environment") == config["environment"], "run environment differs from suite")
        require(seed not in groups[task_id][mode], "duplicate task/mode/seed")
        revisions = integer(run.get("plan_revisions"), "plan_revisions")
        require(revisions <= (0 if mode == "first" else config["replan_budget"]), "replan budget exceeded; invalid trial")
        elapsed = number(run.get("elapsed_s"), "elapsed_s", 0)
        require(elapsed > 0, "elapsed_s must be positive")
        layouts = {}
        for field in ("initial_layout", "final_layout"):
            require(isinstance(run.get(field), str) and run[field], f"{field}: expected file path")
            path = (Path(base_dir) / run[field]).resolve()
            layouts[field] = read_json(path)
            evidence.append({"run_id": run_id, "role": field, "path": run[field], "json_sha256": fingerprint(layouts[field])})
        result = score_state(tasks[task_id], layouts["final_layout"], layouts["initial_layout"], config, run.get("event_violations"))
        # Canonical poses allow different JSON key/object order without breaking pairing.
        poses = object_map(layouts["initial_layout"], tasks[task_id]["target_layout"])
        initial_poses[(task_id, mode, seed)] = poses
        start_hash = fingerprint(poses)
        result.update(run_id=run_id, seed=seed, initial_pose_sha256=start_hash, elapsed_s=elapsed, plan_revisions=revisions)
        groups[task_id][mode][seed] = result
    per_task = []
    for task_id, modes in groups.items():
        require(len(modes["first"]) == len(modes["final"]) == config["repeats"], f"{task_id}: incomplete repeats")
        require(modes["first"].keys() == modes["final"].keys(), f"{task_id}: seeds differ between modes")
        for seed in modes["first"]:
            a = initial_poses[(task_id, "first", seed)]
            b = initial_poses[(task_id, "final", seed)]
            # Physical equality, not byte/hash equality: equivalent quaternion signs and
            # world/local conversion roundoff must not invalidate a correctly paired run.
            require(all(math.dist(a[key]["position"], b[key]["position"]) <= 1e-6
                        and rotation_error_deg(a[key]["rotation"], b[key]["rotation"]) <= 1e-5 for key in a),
                    f"{task_id}/{seed}: paired modes have different initial layouts")
        ordered = {mode: [values[seed] for seed in sorted(values)] for mode, values in modes.items()}
        per_task.append({"task_id": task_id, **summarize(ordered["first"], ordered["final"], config), "runs": ordered})
    total = statistics.mean(t["S"] for t in per_task)
    return {**report_meta(), "kind": "competition_report", "environment": config["environment"],
            "scope": suite["scope"], "moved_policy": suite["moved_policy"],
            "baseline_diagnostic_only": submission.get("baseline_diagnostic_only", False),
            "synthetic_demo_only": bool(suite.get("synthetic_demo_only") or submission.get("synthetic_demo_only")),
            "config": config, "suite_sha256": fingerprint(suite), "task_identity_sha256": evaluation_identity(suite),
            "submission_sha256": fingerprint(submission), "task_count": len(tasks), "run_count": len(runs),
            "score_mean": total, "score_100": total * 100, "tasks": per_task, "evidence": evidence,
            "limitations": ["Offline scoring assumes trusted simulator/verifier observations and telemetry.",
                            "Unique run IDs and paired initial states do not prove independent execution.",
                            "P2 only; no mesh collision, image recognition, P1/P3/P4/P5 or robot control."]}


def evaluate_command(args):
    report = evaluate(read_json(args.suite), read_json(args.runs), Path(args.runs).resolve().parent)
    write_json(args.output, report)
    return {**report_meta(), "saved": str(Path(args.output).resolve()), "task_count": report["task_count"],
            "run_count": report["run_count"], "score_mean": report["score_mean"], "score_100": report["score_100"]}


def stages(sim, real, evidence_score, defense_score=None, zero_sim_policy="zero"):
    require(zero_sim_policy == "zero", "v2.0 competition zero-sim policy is fixed to zero")
    number(evidence_score, "C_evi", 0, 1)
    if defense_score is not None:
        number(defense_score, "defense score", 0, 1)
    task_sets = []
    for report, environment in ((sim, "sim"), (real, "real")):
        require(isinstance(report, dict), "stage report must be an object")
        validate_meta(report)
        validate_moved_policy(report.get("moved_policy"))
        require(report.get("kind") == "competition_report" and report.get("environment") == environment, "wrong report kind/environment")
        require(isinstance(report.get("tasks"), list) and report["tasks"], "empty report tasks")
        validate_config(report.get("config"))
        require(report["config"]["environment"] == environment, "stage report config/environment mismatch")
        values = {}
        for task in report["tasks"]:
            require(isinstance(task, dict), "stage task must be an object")
            key = task.get("task_id")
            require(isinstance(key, str) and key not in values, "duplicate/invalid report task_id")
            values[key] = number(task.get("S"), "task S", 0, 1)
        task_sets.append(values)
    a, b = task_sets
    require(a.keys() == b.keys(), "sim/real task sets must match exactly")
    require(sim["moved_policy"] == real["moved_policy"], "sim/real moved policies differ")
    require(isinstance(sim.get("task_identity_sha256"), str) and sim["task_identity_sha256"] == real.get("task_identity_sha256"), "sim/real task contracts differ")
    for key in ("tier", "completion_threshold", "replan_budget", "violation_cap", "non_target_displacement",
                "event_constraints", "weights", "auxiliary_mode", "stdev", "constraint_score_policy"):
        require(sim["config"][key] == real["config"][key], f"incompatible sim/real policy: {key}")
    ratios = {}
    for key in a:
        if a[key] == 0:
            ratios[key] = 0.0
        else:
            ratios[key] = min(1.0, b[key] / a[key])
    semi = .80 * statistics.mean(b.values()) + .10 * evidence_score + .10 * statistics.mean(ratios.values())
    report = {**report_meta(), "kind": "stage_report",
              "moved_policy": sim["moved_policy"],
              "baseline_diagnostic_only": bool(sim.get("baseline_diagnostic_only") or real.get("baseline_diagnostic_only")),
              "synthetic_demo_only": bool(sim.get("synthetic_demo_only") or real.get("synthetic_demo_only")),
              "score_init": statistics.mean(a.values()), "real_score_mean": statistics.mean(b.values()),
              "C_evi": evidence_score, "rho_by_task": ratios, "rho_mean": statistics.mean(ratios.values()),
              "zero_sim_policy": zero_sim_policy, "score_semi": semi}
    if defense_score is not None:
        report.update(defense_score=defense_score, score_final=.70 * semi + .30 * defense_score)
    return report


def stages_command(args):
    report = stages(read_json(args.sim_report), read_json(args.real_report), args.evidence_score,
                    args.defense_score, args.zero_sim_policy)
    write_json(args.output, report)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", action="version", version=VERSION)
    commands = p.add_subparsers(dest="command", required=True)
    prepare_p = commands.add_parser("prepare", help="freeze paired dataset scenes as P2 contracts")
    prepare_p.add_argument("--target-root", required=True)
    prepare_p.add_argument("--shuffled-root", required=True)
    prepare_p.add_argument("--expected-scenes", type=int, default=100)
    prepare_p.add_argument("--time-ref-s", type=float, default=BASELINE["time_ref_s"])
    prepare_p.add_argument("--violation-cap", type=float, default=BASELINE["violation_cap"])
    prepare_p.add_argument("--completion-threshold", type=float, default=BASELINE["completion_threshold"])
    for option in ("--eps-pos-cm", "--eps-rot-deg", "--delta-max-cm"):
        prepare_p.add_argument(option, type=float)
    prepare_p.add_argument("--event-constraints", nargs="*", choices=sorted(EVENTS), default=[])
    prepare_p.add_argument("--output", required=True)
    prepare_p.set_defaults(func=prepare)
    layout_p = commands.add_parser("layout", help="diagnose a single final layout, without a competition score")
    layout_p.add_argument("--target", required=True)
    layout_p.add_argument("--actual", required=True)
    layout_p.add_argument("--initial")
    layout_p.add_argument("--output")
    layout_p.set_defaults(func=layout_command)
    for sub in (prepare_p, layout_p):
        sub.add_argument("--moved-threshold-cm", type=float, default=MOVED_THRESHOLD_CM,
                         help="select targets with position distance >= this cm value (default: 1.0; calibrated on real 100 scenes)")
        sub.add_argument("--scope", choices=("moved", "all"), default="moved")
        sub.add_argument("--tier", choices=sorted(TIERS), default="standard")
        sub.add_argument("--environment", choices=("sim", "real"), default="sim")
    evaluate_p = commands.add_parser("evaluate", help="score complete independent first/final trials")
    evaluate_p.add_argument("--suite", required=True)
    evaluate_p.add_argument("--runs", required=True)
    evaluate_p.add_argument("--output", required=True)
    evaluate_p.set_defaults(func=evaluate_command)
    stages_p = commands.add_parser("stages", help="combine sim/real reports and committee-assigned scores")
    stages_p.add_argument("--sim-report", required=True)
    stages_p.add_argument("--real-report", required=True)
    stages_p.add_argument("--evidence-score", type=float, required=True)
    stages_p.add_argument("--defense-score", type=float)
    stages_p.add_argument("--zero-sim-policy", choices=("zero",), default="zero")
    stages_p.add_argument("--output", required=True)
    stages_p.set_defaults(func=stages_command)
    return p


def main():
    # Keep redirected JSON output readable on Windows, including Chinese paths.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parser().parse_args()
    try:
        result = args.func(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (InputError, OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
