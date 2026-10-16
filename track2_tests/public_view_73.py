"""Judge-only audit of candidate public fields; never writes task data."""
import argparse
import hashlib
import json
from pathlib import Path

from track2_judge.observation.public_task_view import project


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case-dir')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    from track2_judge.cli.select_case import CASEBANK
    base = Path(a.case_dir).resolve(strict=True) if a.case_dir else CASEBANK / 'cases'
    views = []
    for path in sorted(base.glob('*/case.json')):
        task = json.loads(path.read_text(encoding='utf8'))['contract']
        view = project(task)
        encoded = json.dumps(view, sort_keys=True, allow_nan=False)
        assert set(view) == {'text_instruction', 'predicates', 'constraints'}
        assert len(view['predicates']) == 2 and len(view['constraints']) <= 1
        assert {x['type'] for x in view['predicates']} == {'P1', 'P2'}
        assert {x['type'] for x in view['constraints']} == {'C1'}
        assert all(k not in encoded for k in ('initial_layout', 'target_layout', 'randomization',
                                            'casebank', 'private_catalog', 'source_glb', 'score',
                                            'physics_state', 'asset_path'))
        views.append(hashlib.sha256(encoded.encode()).hexdigest())
    assert len(views) == 73
    result = {'status': 'PASS', 'audited_cases': len(views),
              'unique_public_views': len(set(views)),
              'output_profile': 'frozen public target fields; input serialization WAIT_CONFIRM',
              'initial_state_or_private_path_in_view': False,
              'full_asset_pixel_audit': 'NOT TESTED'}
    Path(a.output).write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
