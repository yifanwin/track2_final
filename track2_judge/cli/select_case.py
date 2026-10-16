"""唯一的一例本地私有入口。单一权威 casebank，无 SHA 链。

CASEBANK 由本文件位置决定（`track2_final/judge_private/casebank`），**不接受命令行覆盖**。
启动时只做结构性检查（无哈希）—— 见 `check_casebank`。
输入契约与生产开关的确认状态另见 `diagnostics/freeze_cases.py` 产出的审计摘要。
"""
import argparse, json, os, re
from pathlib import Path

from track2_judge.scenario.env_factory import create_case_env
from track2_judge.toolapi.dispatch import ManiSkillSimulatorAdapter
from track2_judge.scenario.prototype_backend import physical_snapshot
from track2_judge.contract.trusted_state import spatial_snapshot
from track2_judge.observation.provider import ObservationProvider
from track2_judge.scenario.prototype_env import SETTINGS
from track2_judge.contract.extension import TrustedContractExtension
from track2_judge.runner.candidate_runner import CandidateRunner
from track2_judge.scoring.scorer_bridge import v3
from track2_judge.scenario.initial_gate import validate_initial_scene
from track2_judge.observation.public_task_view import project

CASEBANK = Path(__file__).resolve().parents[2] / "judge_private" / "casebank"
CASE_INDEX = CASEBANK / "cases.json"
EXPECTED_CASES = 73
MAX_CASE_BYTES = 8 * 1024 ** 2
CASE_ID_RE = re.compile('[A-Za-z0-9_-]{1,100}')


def check_casebank(bank=CASEBANK):
    """五条结构性检查，全部与版本无关（无哈希）。

    1. CASEBANK 存在、属主是裁判、mode & 0o077 == 0
    2. cases.json 列出恰好 73 个互不重复的 case_id
    3. 每个 cases/<case_id>/case.json 存在、≤8 MiB、不是符号链接
    4. 所有 hull 路径包含在 CASEBANK 内且不是符号链接（防路径穿越）
    5. 每例的 contract 通过 TrustedContractExtension.validate_task（schema 合法）

    返回 case_id 列表。
    """
    bank = Path(bank).resolve(strict=True)
    if bank.stat().st_uid != os.getuid() or bank.stat().st_mode & 0o077:
        raise PermissionError('casebank must be judge-owned 0700')
    index = json.loads((bank / 'cases.json').read_text(encoding='utf8'))
    ids = index.get('case_ids') if isinstance(index, dict) else None
    if not isinstance(ids, list) or len(ids) != EXPECTED_CASES:
        raise ValueError('casebank index is not a complete 73-case list')
    if len(set(ids)) != EXPECTED_CASES:
        raise ValueError('casebank index has duplicate case ids')
    engine = None
    for case_id in ids:
        # (3)
        if not CASE_ID_RE.fullmatch(case_id):
            raise ValueError('case id rejected: ' + str(case_id))
        path = bank / 'cases' / case_id / 'case.json'
        if not path.is_file() or path.is_symlink():
            raise ValueError('case file missing: ' + case_id)
        if path.stat().st_size > MAX_CASE_BYTES:
            raise ValueError('private case budget exceeded: ' + case_id)
        bundle = json.loads(path.read_text(encoding='utf8'))
        # (4) hull 路径包含 + 非符号链接
        for name, item in bundle['objects'].items():
            for ref in item['hulls']:
                rel = Path(ref['file'])
                if rel.is_absolute() or '..' in rel.parts:
                    raise ValueError('hull path rejected: ' + ref['file'])
                target = bank / rel
                if target.is_symlink():
                    raise ValueError('hull symlink rejected: ' + ref['file'])
                if bank not in target.resolve().parents:
                    raise ValueError('hull outside casebank: ' + ref['file'])
        # (5) schema 合法性
        if engine is None:
            engine = TrustedContractExtension()
        engine.validate_task(bundle['contract'])
    return ids


def run(case_id, skill_zip, output, *, allow_unapproved_diagnostics=False,
        candidate_public_goals=False, skill_timeout_s=20, diagnostic_rgb=False,
        casebank=None):
    bank = Path(casebank).resolve(strict=True) if casebank else CASEBANK.resolve(strict=True)
    check_casebank(bank)
    if not CASE_ID_RE.fullmatch(case_id):
        raise ValueError('case id rejected')
    path = (bank / 'cases' / case_id / 'case.json').resolve(strict=True)
    if bank not in path.parents or path.stat().st_size > MAX_CASE_BYTES:
        raise ValueError('private case budget/containment')
    bundle = json.loads(path.read_text(encoding='utf8'))
    task = bundle['contract']
    engine = TrustedContractExtension()
    engine.validate_task(task)
    from track2_judge.contract.profile import PROFILE, load_config
    public = project(task) if task.get('scoring_profile') == PROFILE or candidate_public_goals else {
        'text_instruction': 'internal legal-action diagnostic'}
    camera = None
    camera_specs = None
    if diagnostic_rgb:
        import numpy as np
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        goal = next(p['target_pose'] for p in public['predicates'] if p['type'] == 'P2')[:3]
        camera_specs = {'center_m': goal, 'height_m': 1.2, 'fov_rad': .7, 'diagnostic_only': True}
        public['text_instruction'] = 'DIAGNOSTIC_RGB_CAMERA=' + json.dumps(camera_specs, separators=(',', ':')) + '\n' + public['text_instruction']

        def camera(env):
            env.scene.update_render()
            c = env._human_render_cameras['diagnostic_public_rgb']
            c.capture()
            rgb = c.camera._render_cameras[0].get_picture('Color')
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb[:, :, :3] * 255, 0, 255).astype(np.uint8)
            return rgb[:, :, :3]

    def factory(task, seed, mode):
        cp = next((p['checkpoints'] for p in task['predicates'] if p['type'] == 'P5'), [])
        return ManiSkillSimulatorAdapter(
            lambda: create_case_env(bundle, bank, seed,
                                    allow_unapproved_diagnostics=allow_unapproved_diagnostics,
                                    diagnostic_camera=camera_specs),
            snapshot=physical_snapshot, spatial_provider=spatial_snapshot,
            metadata={n: {'pivot_m': [0, 0, 0]} for n in bundle['objects']},
            scene_settings=SETTINGS, public_fields=tuple(public),
            observation_provider=ObservationProvider(approved_fields=tuple(public), camera=camera,
                                                     camera_required=diagnostic_rgb),
            checkpoints=cp, recovery=mode == 'final', initial_validation=validate_initial_scene)

    factory.source = 'ACTUAL selected TTG candidate, source collision assets; candidate physics'
    # Confirmed public targets cross only the explicit JSON projection.
    # Input serialization remains a candidate profile; private bank never crosses.
    return CandidateRunner(factory, package_zip=skill_zip, isolated=True,
                           contract_extension=engine, timeout_s=skill_timeout_s).run(
        {'config': load_config(), 'tasks': [task]}, output, public)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--case-id')
    p.add_argument('--skill-zip')
    p.add_argument('--output')
    p.add_argument('--allow-unapproved-diagnostics', action='store_true')
    p.add_argument('--candidate-public-goals', action='store_true')
    p.add_argument('--list-cases', action='store_true', help='只做结构性检查并打印 case 列表')
    a = p.parse_args()
    if a.list_cases:
        ids = check_casebank()
        print(json.dumps({'status': 'PASS', 'cases': len(ids), 'case_ids': ids}))
        raise SystemExit(0)
    missing = [n for n in ('case_id', 'skill_zip', 'output') if not getattr(a, n)]
    if missing:
        p.error('缺少参数: ' + ', '.join('--' + n.replace('_', '-') for n in missing))
    r = run(a.case_id, a.skill_zip, a.output,
            allow_unapproved_diagnostics=a.allow_unapproved_diagnostics,
            candidate_public_goals=a.candidate_public_goals)
    print(json.dumps({'episodes': len(r['episodes']),
                      'candidate_score': r.get('extension_candidate_score_100'),
                      'official_score': None,
                      'scoring_status': r.get('scoring_status', 'CANDIDATE ONLY')}))
