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
    from track2_judge.cli.select_case import CASEBANK, check_casebank
    base = Path(a.case_dir).resolve(strict=True) if a.case_dir else CASEBANK / 'cases'
    views = []
    for path in sorted(base.glob('*/case.json')):
        task = json.loads(path.read_text(encoding='utf8'))['contract']
        # ⚠️ 这里**不传 reference_image** —— 它由调用方（`select_case.run()`）注入，
        #    不在 `project()` 里现造。所以这条测试审的是**永远该有的那三项**。
        #    ★ 2026-10-19：`camera_calibration` **已彻底移除**（不再是合法公开字段），
        #    所以它连「可选注入项」都算不上 —— 上界里也不该再列它。
        view = project(task)
        encoded = json.dumps(view, sort_keys=True, allow_nan=False)
        assert {'text_instruction', 'predicates', 'constraints'} <= set(view)
        assert set(view) <= {'text_instruction', 'predicates', 'constraints',
                             'reference_image'}
        assert len(view['predicates']) == 2 and len(view['constraints']) <= 1
        assert {x['type'] for x in view['predicates']} == {'P1', 'P2'}
        assert {x['type'] for x in view['constraints']} == {'C1'}
        assert all(k not in encoded for k in ('initial_layout', 'target_layout', 'randomization',
                                            'casebank', 'private_catalog', 'source_glb', 'score',
                                            'physics_state', 'asset_path'))
        views.append(hashlib.sha256(encoded.encode()).hexdigest())
    assert len(views) == len(check_casebank())
    result = {'status': 'PASS', 'audited_cases': len(views),
              'unique_public_views': len(set(views)),
              'output_profile': 'frozen public target fields; input serialization WAIT_CONFIRM',
              'initial_state_or_private_path_in_view': False,
              'full_asset_pixel_audit': 'NOT TESTED'}
    Path(a.output).write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
