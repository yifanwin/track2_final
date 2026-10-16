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
    a = p.parse_args()
    os.umask(0o077)
    out = Path(a.output).resolve()
    if out.exists():
        raise ValueError('unique new private output required')
    try:
        from track2_judge.cli.select_case import run
        r = run(a.case_id, a.skill_zip, out)
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
