"""未声明的 Tool 必须在物理启动前被拒（验收标准 2）。

## 这个测试取代了什么

第 1 步之前，这里测的是 `runner.py:37` 的 `tool_profile` 字符串门禁 ——
选手包不带 `tool_profile: piper-joint-v2-candidate` 就不许跑。

第 1.7 步按 Q16（单一新轨）删掉了那个门禁：**不再按固定字符串判准入**。
准入改为「tool 是否在官方 `required_tools` 声明集合内」，由 Gateway 裁决。

所以本测试现在检查的是**真东西**：一个调用未声明 Tool 的选手包，
必须在 Skill 跑起来、物理动起来**之前**被拒。

## 顺带记一笔既有缺陷

源目录的同名测试（`legacy_profile_reject_test.py:18`）断言
`'Piper candidate tool profile required' in str(error)`，但实际抛出的 ValueError
文案不同 —— 所以它在源目录里**本来就是失败的**（rc=1）。
本测试是重写版，不是「把一个能过的测试改坏」。
"""
import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

from track2_judge.toolapi import schema as S
from track2_judge.toolapi.client import ToolCallError, ToolClient, InProcessTransport

fails = []


def check(label, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {label}{'  ' + str(detail) if detail else ''}")
    if not ok:
        fails.append(label)


# ── 1. 声明集合本身

print("=== 声明集合 ===")
check("官方声明恰好 3 个", len(S.REQUIRED_TOOLS) == 3, S.REQUIRED_TOOLS)
for tool_id in S.REQUIRED_TOOLS:
    check(f"  {tool_id} 已声明", S.is_declared(tool_id))

UNDECLARED = [
    ("motion.move_pose", "move-arm-by-ee 的任务级 Tool（路线 A，已否）"),
    ("motion.move_joints", "同上"),
    ("gripper.set_opening", "同上"),
    ("tabletop.set_pose", "生产 v4.2 的摆位语义（Q16 废弃）"),
    ("tabletop.translate", "同上"),
    ("tabletop.get_pose", "同上"),
    ("track2.admin.reset", "裁判专用，不进选手 required_tools"),
    ("track2.admin.finalize", "同上"),
    ("get_final_state", "裁判 API，不得经 RPC 逃逸"),
    ("get_layout", "同上"),
    ("piper_joint_control", "旧方法名不是 tool_id"),
    ("set_pose", "旧契约裸名"),
]
print("\n=== 未声明的必须被拒 ===")
for tool_id, why in UNDECLARED:
    check(f"  {tool_id:24s} 被拒", not S.is_declared(tool_id), why)


# ── 2. 真实的 SDK 级拒绝（不启动物理）

print("\n=== 客户端侧拒绝（无需物理）===")


class _NeverCalled:
    """任何调用都算失败 —— 拒必须在**碰到执行体之前**发生。"""

    def __getattr__(self, name):
        def boom(*a, **k):
            raise AssertionError(f"executor.{name} was reached; rejection must come first")
        return boom


from track2_judge.toolapi.gateway import LocalGateway  # noqa: E402

client = ToolClient(InProcessTransport(LocalGateway(_NeverCalled())))
for tool_id, _why in UNDECLARED:
    try:
        client.call(tool_id, {})
        check(f"  call({tool_id}) 被拒", False, "(未被拒)")
    except ToolCallError as exc:
        check(f"  call({tool_id:24s}) 被拒", True, str(exc)[:58])
    except AssertionError as exc:
        check(f"  call({tool_id}) 在碰到执行体前被拒", False, str(exc))


# ── 3. 沙箱形状：JSON-lines 里的旧方法名也要被挡住

print("\n=== 沙箱 JSON-lines 形状 ===")
from track2_judge.toolapi.client import SandboxBridge  # noqa: E402

bridge = SandboxBridge(ToolClient(InProcessTransport(LocalGateway(_NeverCalled()))))
for tool_id, _why in UNDECLARED:
    try:
        bridge(tool_id, {})
        check(f"  bridge({tool_id:20s}) 被拒", False, "(未被拒)")
    except ToolCallError as exc:
        check(f"  bridge({tool_id:20s}) 被拒", True, str(exc)[:50])
    except AssertionError as exc:
        check(f"  bridge({tool_id}) 在碰到执行体前被拒", False, str(exc))


# ── 4. 端到端：真跑一个调用未声明 Tool 的选手包

print("\n=== 端到端：选手包调用未声明 Tool ===")


def build_probe_zip(destination: Path, tool_name: str, arguments=None) -> Path:
    """造一个最小选手包，`run` 里调用指定的 tool。"""
    arguments = {} if arguments is None else arguments
    source = f'''import json

def initialize(context):
    if context.get('skill_api_version') != 'ctrc-skill-v1':
        raise ValueError('context boundary')

def run(observation, tools):
    tools.call({tool_name!r}, json.loads({json.dumps(arguments)!r}))
    return {{'status': 'completed'}}

def shutdown(context):
    pass
'''
    manifest = {
        "skill_name": "undeclared_tool_probe",
        "version": "2.0.0",
        "entrypoint": "main.py",
        "api_version": "ctrc-skill-v1",
    }
    # `unpack_skill` 要求恰好一个 manifest.json，且所有文件都在它的目录内 ——
    # 所以打成 `probe/` 顶层目录，而不是散在 zip 根。
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("probe/manifest.json", json.dumps(manifest))
        archive.writestr("probe/main.py", source)
    return destination


def main(output):
    root = Path(output).resolve()
    root.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.umask(0o077)
    result = {"status": "PASS", "checks": {}, "episodes": [],
              "scope": "synthetic undeclared-tool package through the real entry"}

    from track2_judge.cli.select_case import run

    probe = build_probe_zip(root / "probe.zip", "motion.move_pose")
    diagnosis = run("001_real_world_images__real_image_1", probe, root / "run")
    episodes = diagnosis.get("episodes", [])
    result["episodes"] = [{k: v for k, v in e.items()
                           if k in ("seed", "mode", "status", "error_category", "phase")}
                          for e in episodes]

    # `run()` **不抛异常** —— 一个 episode 被拒是「已记录的失败」，不是编排错误。
    # 这正是原来的行为（failures 列表），我们要检查的是失败被正确分类。
    rejected = [e for e in episodes
                if e.get("status") == "error" and e.get("error_category") == "ACTION_REJECTED"]
    result["checks"]["undeclared_tool_rejected_as_action_rejected"] = (
        "PASS" if len(rejected) == len(episodes) and episodes else "FAIL")
    result["checks"]["no_episode_succeeded"] = (
        "PASS" if not any(e.get("status") == "success" for e in episodes) else "FAIL")
    # 计分必须被挡住。
    # 注意：`scoring_status` 仍是 FROZEN_AUTHOR_PDF_RULES（冻结评分器没有执行状态字段，
    # 这是**既有**语义 —— 见 `diagnosis['exception_policy']`），
    # 真正兜底的是 `candidate_status=EXECUTION_FAILURE` + ResultGate BLOCKED。
    result["checks"]["candidate_status_execution_failure"] = (
        "PASS" if diagnosis.get("candidate_status") == "EXECUTION_FAILURE" else "FAIL")
    gate = diagnosis.get("result_gate", {})
    result["checks"]["result_gate_blocked"] = (
        "PASS" if gate.get("status") == "BLOCKED" and gate.get("publishable") is False else "FAIL")

    # 对照：同一个包改成调用**已声明**的 tool（空参数），应当能跑完 ——
    # 证明拒绝的是「未声明」，不是「所有 call」。
    good_zip = build_probe_zip(root / "good.zip", "track2.env.observe", {})
    good = run("001_real_world_images__real_image_1", good_zip, root / "run_good")
    good_eps = good.get("episodes", [])
    result["checks"]["declared_tool_not_rejected_as_action_rejected"] = (
        "PASS" if not any(e.get("error_category") == "ACTION_REJECTED" for e in good_eps) else "FAIL")
    result["checks"]["declared_tool_reaches_completion"] = (
        "PASS" if all(e.get("status") in ("success", "failure") for e in good_eps) and good_eps
        else "FAIL")

    # 错参数也要在 Gateway 层被拒（拿合法 tool 传非法参数）
    bad_args_zip = build_probe_zip(root / "badargs.zip", "track2.env.observe",
                                   {"target_pose": [1, 2, 3]})
    bad_args = run("001_real_world_images__real_image_1", bad_args_zip, root / "run_badargs")
    result["checks"]["bad_arguments_rejected"] = (
        "PASS" if all(e.get("error_category") == "ACTION_REJECTED"
                      for e in bad_args.get("episodes", [])) and bad_args.get("episodes")
        else "FAIL")

    result["status"] = "PASS" if all(v == "PASS" for v in result["checks"].values()) else "FAIL"
    (root / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(f"\n{'=' * 60}")
    print(f"{'静态部分全部通过' if not fails else str(len(fails)) + ' 项失败: ' + str(fails)}")
    if fails:
        sys.exit(1)
    sys.exit(main(args.output))
