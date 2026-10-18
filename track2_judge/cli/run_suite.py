"""顺序跑完 73 例的私有彩排；无队列、无回调、无选手数据。

单一权威 casebank（无 --casebank / --freeze-manifest 参数）。

⚠️ **默认口径 = 5 seed**（`release_entry.py` 的默认）—— 只有 5 seed 才出得了分
（冻结评分器 `mesa_eval.REPEATS["sim"] = 5`）。`--seeds` 是给**冒烟**用的：
少跑时 Gate 会标 `DIAGNOSTIC_PARTIAL_SEEDS_NO_SCORE`，本脚本也如实标成诊断。
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

from track2_judge.cli.select_case import CASEBANK, check_casebank

REPEATS_SIM = 5

p = argparse.ArgumentParser()
p.add_argument('--skill-zip', required=True)
p.add_argument('--output', required=True)
# ★ 冒烟用：只跑这些 seed（逗号分隔）。**不给就是全 5 seed（正式口径）**。
#   少跑产不出分数（见模块 docstring），所以汇总里会显式标出来。
p.add_argument('--seeds', default=None,
               help='只跑这些 seed（逗号分隔）。默认全 5 seed；少跑 = 冒烟，不评分')
# 透传给 release_entry 的其它开关（都是诊断用）
p.add_argument('--no-camera', dest='camera', action='store_false', default=True,
               help='关相机（CPU 回归/排查）。正式评测不要带')
p.add_argument('--render-gpu', default=None)
a = p.parse_args()
os.umask(0o077)
out = Path(a.output).resolve()
out.mkdir(parents=True, mode=0o700, exist_ok=False)
case_ids = check_casebank(CASEBANK)
release_entry = Path(__file__).with_name('release_entry.py')

# 本次预期每个 case 跑几条 episode 记录（= seed 数 × 2 个 mode）
seeds = ([int(s) for s in a.seeds.split(',') if s.strip()] if a.seeds else list(range(REPEATS_SIM)))
expected_episodes = len(seeds) * 2
partial = len(seeds) != REPEATS_SIM

results = []
start = time.monotonic()
for i, case_id in enumerate(case_ids):
    destination = out / ('case_%03d' % i)
    cmd = [sys.executable, '-B', str(release_entry), '--case-id', case_id,
           '--skill-zip', a.skill_zip, '--output', str(destination)]
    if a.seeds:
        cmd += ['--seeds', a.seeds]
    if not a.camera:
        cmd.append('--no-camera')
    if a.render_gpu:
        cmd += ['--render-gpu', a.render_gpu]
    with (out / ('case_%03d.private.log' % i)).open('x') as log:
        try:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
            code = r.returncode
        except subprocess.TimeoutExpired:
            code = -99
    d = json.loads((destination / 'diagnosis.json').read_text()) if (destination / 'diagnosis.json').exists() else {}
    gate = json.loads((destination / 'result_gate.json').read_text()) if (destination / 'result_gate.json').is_file() else {}
    # ⚠️ episode 条数按**本次跑了几个 seed** 判，不写死 10。
    #
    #   两种口径的"通过"定义不同：
    #     * 正式口径（5 seed）：要 `publishable`（Gate PASS）+ 出分
    #     * 冒烟口径（少 seed）：Gate 必然 BLOCKED（`DIAGNOSTIC_PARTIAL_SEEDS_NO_SCORE`），
    #       所以判据换成「跑完了 + episode 数对 + 每条都有结论」——
    #       这正是冒烟要回答的问题：**这一 case 起不起得来**。
    #
    # ⚠️ **可评分终态是三个，不是两个**（2026-10-18）：
    #   `success` / `failure` / `budget_timeout`（正常预算耗尽）。
    #   这里以前只认前两个 → 真策略出现正常超时时会被误判成 FAIL（假阴性）。
    #   与 `resident_runner.SCORABLE` / `result_gate.SCORABLE_TERMINAL_STATUS` 同源；
    #   改一处必须改三处（见 plan/Track2_丢包自锁修复方案.md §0.4）。
    #   `infrastructure_timeout`（链路故障）**不在**其中 —— 它要判 FAIL。
    SCORABLE_TERMINAL_STATUS = ('success', 'failure', 'budget_timeout')
    concluded = (len(d.get('episodes', [])) == expected_episodes
                 and all(x.get('status') in SCORABLE_TERMINAL_STATUS
                         for x in d.get('episodes', [])))
    if partial:
        valid = code == 0 and concluded
    else:
        valid = (code == 0 and concluded
                 and gate.get('publishable') is True
                 and d.get('scoring_status') == 'FROZEN_AUTHOR_PDF_RULES')
    # Q14：动作耗时（由 release_entry 从 runtime 节点的私有控制口取回，落在 case 目录里）
    timings = None
    timings_path = destination / 'timings.json'
    if timings_path.is_file():
        try:
            timings = json.loads(timings_path.read_text())
        except (ValueError, OSError):
            timings = None
    # ① 兜底判据的可见性：把「执行残缺」的 episode 数报出来，
    #   这样汇总一行就能看出是「链路坏了」还是「策略不行」。
    incomplete = sum(1 for e in d.get('episodes', [])
                     if e.get('execution_integrity') == 'INCOMPLETE')
    infra = sum(1 for e in d.get('episodes', [])
                if e.get('status') == 'infrastructure_timeout')
    results.append({'ordinal': i, 'case_id': case_id,
                    'status': 'PASS' if valid else 'FAIL',
                    'episodes': len(d.get('episodes', [])),
                    'frozen_rule_score_100': d.get('frozen_rule_score_100'),
                    'scoring_status': d.get('scoring_status'),
                    'episodes_incomplete': incomplete,
                    'episodes_infrastructure_timeout': infra,
                    'action_count': (timings or {}).get('count'),
                    'action_max_s': ((timings or {}).get('quantiles') or {}).get('max')})
    print(json.dumps(results[-1]), flush=True)
    good = [x for x in results if x['status'] == 'PASS']
    summary = {'status': 'PASS' if len(good) == 73 else 'RUNNING' if len(results) < 73 else 'FAIL',
               'case_count': 73, 'completed_cases': len(results), 'pass_cases': len(good),
               'fail_cases': len(results) - len(good), 'episodes': sum(x['episodes'] for x in results),
               'frozen_rule_score_100': sum(x['frozen_rule_score_100'] for x in good) / 73 if len(good) == 73 else None,
               # ★ 口径如实标注 —— 1 seed 的产物**不是**正式成绩，别让它被误读
               'repeats': len(seeds), 'seeds': seeds,
               'scoring_status': ('DIAGNOSTIC_PARTIAL_SEEDS' if partial
                                  else 'FROZEN_AUTHOR_PDF_RULES'),
               'scoring_note': ('只跑 %d 个 seed（正式口径 %d）—— 冒烟，不构成成绩'
                                % (len(seeds), REPEATS_SIM)) if partial else None,
               'strategy_scope': 'internal legal-action weak Skill; not difficulty calibration',
               'production_callback_sent': False, 'wall_s': round(time.monotonic() - start, 2)}
    (out / 'sanitized_summary.json').write_text(json.dumps(summary, indent=2))
    (out / 'private_results.json').write_text(json.dumps(results, indent=2))

# Q14：把 73 例的动作耗时汇总成一份可直接看分布的产物
_timings = []
for i in range(len(results)):
    path = out / ('case_%03d' % i) / 'timings.json'
    if path.is_file():
        try:
            _timings.append(json.loads(path.read_text()))
        except (ValueError, OSError):
            pass
if _timings:
    durations = sorted(d for t in _timings for d in (t.get('durations_s') or []))

    def _q(p):
        if not durations:
            return None
        return durations[min(len(durations) - 1, int(len(durations) * p))]

    (out / 'timings_summary.json').write_text(json.dumps({
        'case_count': len(_timings),
        'action_count': sum(t.get('count') or 0 for t in _timings),
        'quantiles_s': {'p50': _q(0.50), 'p90': _q(0.90), 'p99': _q(0.99),
                        'p999': _q(0.999), 'max': (durations or [None])[-1]},
        'current_action_timeout_s': _timings[0].get('action_timeout_s'),
        'action_timeout_source': _timings[0].get('action_timeout_source'),
        'note': 'Q14 数据源：每 case 另见 case_NNN/timings.json',
    }, indent=2))

sys.exit(summary['status'] != 'PASS')
