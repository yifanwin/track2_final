"""Judge-only audit of one selected case at a time. Emits aggregate results only."""
import argparse
import collections
import json
from pathlib import Path

from track2_judge.toolapi.dispatch import ManiSkillSimulatorAdapter
from track2_judge.scenario.prototype_backend import physical_snapshot
from track2_judge.scenario.env_factory import create_case_env
from track2_judge.scenario.initial_gate import validate_initial_scene
from track2_judge.scenario.prototype_env import SETTINGS


def run(bank_path, output_path, seeds):
    bank = Path(bank_path).resolve(strict=True)
    from track2_judge.cli.select_case import check_casebank
    rows = [{'case_id': cid} for cid in check_casebank(bank)]
    results = []
    for seed in range(seeds):
        for index, row in enumerate(rows):
            bundle = json.loads((bank / 'cases' / row['case_id'] / 'case.json').read_text())
            adapter = None
            try:
                adapter = ManiSkillSimulatorAdapter(
                    lambda: create_case_env(bundle, bank, seed, allow_unapproved_diagnostics=True),
                    snapshot=physical_snapshot,
                    metadata={name: {'pivot_m': [0, 0, 0]} for name in bundle['objects']},
                    scene_settings=SETTINGS,
                    initial_validation=validate_initial_scene,
                )
                adapter.reset(seed=seed)
                evidence = adapter.get_initialization_evidence()
                result = {'alias': f'material_{index:03d}', 'seed': seed, 'status': 'PASS',
                          'max_penetration_m': evidence['max_penetration_m'],
                          'max_actor_speed_m_s': evidence['max_actor_speed_m_s'],
                          'max_arm_velocity_rad_s': evidence['max_arm_velocity_rad_s']}
            except Exception as error:
                result = {'alias': f'material_{index:03d}', 'seed': seed, 'status': 'FAIL',
                          'error_type': type(error).__name__}
            finally:
                if adapter is not None:
                    adapter.close()
            results.append(result)
    counts = dict(collections.Counter(r['status'] for r in results))
    out = {'scope': 'candidate initial scene gate, selected private cases and five seeds',
           'cases': len(rows), 'seeds': seeds, 'trials': len(results), 'counts': counts, 'results': results,
           'score_changed': False, 'official_approval': False}
    Path(output_path).write_text(json.dumps(out, indent=2, allow_nan=False))
    print(json.dumps({'cases': len(rows), 'seeds': seeds, 'trials': len(results), 'counts': counts}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--casebank')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seeds', type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.seeds <= 5:
        parser.error('seed count must be 1..5')
    from track2_judge.cli.select_case import CASEBANK
    run(args.casebank or CASEBANK, args.output, args.seeds)
