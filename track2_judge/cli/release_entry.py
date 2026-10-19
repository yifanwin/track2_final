"""唯一离线发布入口；私有产物，无生产队列/回调。

单一权威 casebank（无 --casebank / --freeze-manifest 参数）。
入口只做「结构性检查 + 单例评测」，不再做发布期哈希门禁。
"""
import argparse
import os
import sys
from pathlib import Path

from track2_judge.scoring.result_gate import block


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case-id', required=True)
    p.add_argument('--skill-zip', required=True)
    p.add_argument('--output', required=True)
    # 裁判侧 Oracle 的 GT 通路（第 6 步 / G2）。**普通评测不带这个参数** ——
    # 带它等于让那次评测的 skill 拿到 GT 规划结果。
    #
    # 两种取值：
    #   （不传）        = 普通评测
    #   `--oracle-plan auto`   = **由 runtime 节点在它自己的 env 上现做 GT 规划**
    #                            （常驻链下唯一可行的一种，G2 已闭）
    #
    # ⚠️ 从 2026-10-17 起**不接受 plan 文件**：那种形态要求「裁判能自己算规划」，
    #    而常驻结构下 env 在 runtime 节点里 —— 裁判算不了。传文件会明确报错。
    p.add_argument('--oracle-plan', default=None,
                   help='"auto" 由 runtime 节点现做 GT 规划；plan 文件在常驻链下不被支持')
    # ── 实时 RGB（闭环策略的前提）
    #
    # **默认开**。比赛的题目是「参考目标图像，自主完成目标识别/任务拆解」，
    # 而参考图只说明「最终该长什么样」—— 没有实时画面，选手无法回答
    # 「杯子现在在哪 / 该往哪动 / 刚才抓成功了吗」，于是视觉+传统规划、VLM、
    # VLA、visual servo 全部做不了，只剩开环动作（随机化布局下不可用）。
    #
    # ⚠️ 关掉它 = 把这条链路降级成「没有视觉」（2026-10-17 起真的会生效）：
    #     * `images.camera` 不存在（`camera_status` 为 `NOT CONFIGURED`）
    #     * `task_inputs.reference_image` **仍在**（它来自预渲染文件，不依赖相机）
    #
    #   ★ 2026-10-19：`task_inputs.camera_calibration` **已彻底移除**（不再是公开
    #   字段）—— 与这个开关无关，开不开相机都不给。选手靠**参考图与实时帧的
    #   图像比对**定位/伺服。
    #
    #   所以它的用途只有两个：**CPU 回归**（无 GPU 环境）、以及定位
    #   「是不是渲染把链路弄挂的」。**正式评测永远不要带这个参数。**
    p.add_argument('--no-camera', dest='camera', action='store_false', default=True,
                   help='关相机（CPU 回归/排查用）：images.camera 会消失。正式评测不要带')
    # ── C2：走**常驻 PAOS flow**（唯一的选手执行链）
    #
    # 从 2026-10-17 起默认就是它。`--legacy-in-process` 保留旧路（在裁判进程里
    # 造 env 跑 skill）**只**为了在 PAOS 起不来时能对照排查 —— 它不是评测形态，
    # 两条路的判定语义不同（旧路选手拿到的是裁判选定的公开输入）。
    p.add_argument('--legacy-in-process', dest='resident',
                   action='store_false', default=True,
                   help='诊断用：退回裁判进程内执行（正式评测不要带这个参数）')
    p.add_argument('--bundle-root', default=None,
                   help='PAOS Bundle 根目录（默认取 TRACK2_BUNDLE_ROOT）')
    p.add_argument('--seeds', default=None,
                   help='只跑这些 seed，逗号分隔（默认 0-4）')
    p.add_argument('--render-gpu', default=None,
                   help='渲染/物理用哪张卡（默认 TRACK2_RENDER_GPU 或 0）')
    a = p.parse_args()
    os.umask(0o077)
    out = Path(a.output).resolve()
    if out.exists():
        raise ValueError('unique new private output required')
    try:
        if a.resident:
            r = _resident(a, out)
        else:
            r = _legacy_in_process(a, out)
        gate = r.get('result_gate', {})
        print({'status': gate.get('status', 'BLOCKED'), 'official_score': None})
        return 0 if gate.get('status') == 'PASS' else 2
    except Exception as exc:
        # No judge paths, GT or exception text enters public output.
        reason = 'GT_OR_INITIALIZATION_EXCEPTION_' + type(exc).__name__
        block(out, reason)
        print({'status': 'BLOCKED', 'reason': reason, 'official_score': None})
        return 2


def _resident(a, out):
    """走常驻 PAOS flow（正式路径）。"""
    import os
    from pathlib import Path as _P

    from track2_judge.cli.select_case import CASEBANK, REFERENCE_ROOT
    from track2_judge.runner.resident_runner import ResidentRunner

    bundle = (a.bundle_root or os.environ.get('TRACK2_BUNDLE_ROOT')
              or _P(__file__).resolve().parents[3] / 'track2_official_runtime')
    physics = os.environ.get('TRACK2_PHYSICS_ROOT') or str(
        _P(__file__).resolve().parents[2])
    python = os.environ.get('TRACK2_PYTHON') or os.environ.get(
        'TRACK2_REGRESS_PYTHON') or 'python3'
    # ── Oracle 的 GT 规划（G2 已闭）
    #
    # 规划**在 runtime 节点上做**（那里才有 env）—— 常驻结构下裁判进程没有 env，
    # 所以 `--oracle-plan auto` 的语义是「让 runtime 用**它自己的 env** 现做规划」，
    # 由 `PersistentFlow(oracle_planning=True)` 渲染进 `runtime.yaml` 打开。
    #
    # `--oracle-plan <file>`（外部算好的 plan）在这条链上**不适用** ——
    # 它要求「裁判能自己算」，而那正是常驻链没有的东西。明确失败，不静默忽略。
    oracle_auto = False
    if a.oracle_plan == 'auto':
        oracle_auto = True
    elif a.oracle_plan:
        raise ValueError(
            'ORACLE_PLAN_FILE_NOT_SUPPORTED_ON_RESIDENT_CHAIN: '
            'use --oracle-plan auto（规划由 runtime 节点做），或 --legacy-in-process')
    seeds = ([int(s) for s in a.seeds.split(',') if s.strip()]
             if a.seeds else None)
    return ResidentRunner(
        case_id=a.case_id, casebank=CASEBANK, reference_root=REFERENCE_ROOT,
        bundle_root=bundle, physics_root=physics, python=python,
        skill_zip=a.skill_zip, output=out,
        seeds=seeds, diagnostic_rgb=a.camera,
        oracle_planning=oracle_auto,
        runtime_gpu=a.render_gpu or os.environ.get('TRACK2_RENDER_GPU') or '0',
    ).run()


def _legacy_in_process(a, out):
    """旧路：裁判进程内造 env + 跑 skill（**诊断用**，不是评测形态）。"""
    from track2_judge.cli.select_case import run
    oracle_plan = None
    if a.oracle_plan == 'auto':
        oracle_plan = True
    elif a.oracle_plan:
        import json
        from pathlib import Path as _P
        oracle_plan = json.loads(_P(a.oracle_plan).read_text(encoding='utf8'))
    return run(a.case_id, a.skill_zip, out, oracle_plan=oracle_plan,
               diagnostic_rgb=a.camera)


if __name__ == '__main__':
    raise SystemExit(main())
