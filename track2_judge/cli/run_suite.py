"""顺序跑完 73 例的私有彩排；无队列、无回调、无选手数据。

单一权威 casebank（无 --casebank / --freeze-manifest 参数）。
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

from track2_judge.cli.select_case import CASEBANK, check_casebank

p = argparse.ArgumentParser()
p.add_argument('--skill-zip', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
os.umask(0o077)
out = Path(a.output).resolve()
out.mkdir(parents=True, mode=0o700, exist_ok=False)
case_ids = check_casebank(CASEBANK)
release_entry = Path(__file__).with_name('release_entry.py')
results = []
start = time.monotonic()
for i, case_id in enumerate(case_ids):
    destination = out / ('case_%03d' % i)
    cmd = [sys.executable, '-B', str(release_entry), '--case-id', case_id,
           '--skill-zip', a.skill_zip, '--output', str(destination)]
    with (out / ('case_%03d.private.log' % i)).open('x') as log:
        try:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
            code = r.returncode
        except subprocess.TimeoutExpired:
            code = -99
    d = json.loads((destination / 'diagnosis.json').read_text()) if (destination / 'diagnosis.json').exists() else {}
    gate = json.loads((destination / 'result_gate.json').read_text()) if (destination / 'result_gate.json').is_file() else {}
    valid = code == 0 and gate.get('publishable') is True and d.get('scoring_status') == 'FROZEN_AUTHOR_PDF_RULES' and len(d.get('episodes', [])) == 10 and all(x['status'] in ('success', 'failure') for x in d['episodes'])
    results.append({'ordinal': i, 'status': 'PASS' if valid else 'FAIL', 'episodes': len(d.get('episodes', [])), 'frozen_rule_score_100': d.get('frozen_rule_score_100')})
    print(json.dumps(results[-1]), flush=True)
    good = [x for x in results if x['status'] == 'PASS']
    summary = {'status': 'PASS' if len(good) == 73 else 'RUNNING' if len(results) < 73 else 'FAIL',
               'case_count': 73, 'completed_cases': len(results), 'pass_cases': len(good),
               'fail_cases': len(results) - len(good), 'episodes': sum(x['episodes'] for x in results),
               'frozen_rule_score_100': sum(x['frozen_rule_score_100'] for x in good) / 73 if len(good) == 73 else None,
               'strategy_scope': 'internal legal-action weak Skill; not difficulty calibration',
               'production_callback_sent': False, 'wall_s': round(time.monotonic() - start, 2)}
    (out / 'sanitized_summary.json').write_text(json.dumps(summary, indent=2))
    (out / 'private_results.json').write_text(json.dumps(results, indent=2))
sys.exit(summary['status'] != 'PASS')
