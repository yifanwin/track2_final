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
    # 裁判侧 Oracle 的 GT 通路（第 6 步）。**普通评测不带这个参数** ——
    # 带它等于让那次评测的 skill 拿到 GT 规划结果。
    #
    # 三种取值：
    #   （不传）        = 普通评测
    #   `--oracle-plan auto`   = **在本入口里现做 GT 规划**（推荐）
    #   `--oracle-plan <file>` = 用外部算好的 plan（兼容旧用法）
    p.add_argument('--oracle-plan', default=None,
                   help='"auto" to plan here, or a JSON file with a prebuilt plan')
    a = p.parse_args()
    os.umask(0o077)
    out = Path(a.output).resolve()
    if out.exists():
        raise ValueError('unique new private output required')
    try:
        from track2_judge.cli.select_case import run
        oracle_plan = None
        if a.oracle_plan == 'auto':
            oracle_plan = True                 # 入口自己趁 reset 之后规划
        elif a.oracle_plan:
            import json
            from pathlib import Path as _P
            oracle_plan = json.loads(_P(a.oracle_plan).read_text(encoding='utf8'))
        r = run(a.case_id, a.skill_zip, out, oracle_plan=oracle_plan)
        gate = r.get('result_gate', {})
        print({'status': gate.get('status', 'BLOCKED'), 'official_score': None})
        return 0 if gate.get('status') == 'PASS' else 2
    except Exception as exc:
        # No judge paths, GT or exception text enters public output.
        reason = 'GT_OR_INITIALIZATION_EXCEPTION_' + type(exc).__name__
        block(out, reason)
        print({'status': 'BLOCKED', 'reason': reason, 'official_score': None})
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
