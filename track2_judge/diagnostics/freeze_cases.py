"""Judge-only, read-only 73-case audit and content-addressed freeze manifest.

The detailed manifest contains task IDs and private target fingerprints. Keep it
beside the 0700 casebank; publish only the aggregate summary separately.
"""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path

from track2_judge.contract.extension import TrustedContractExtension, digest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qrotate(q, v):
    x, y, z, w = map(float, q)
    vx, vy, vz = map(float, v)
    t = [2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)]
    return [v[i] + w * t[i] + (y * t[2] - z * t[1], z * t[0] - x * t[2], x * t[1] - y * t[0])[i] for i in range(3)]


def check_layout(layout, names):
    if layout.get('scene_settings', {}).get('units') != 'centimeters' or layout['scene_settings'].get('up_axis') != 'Z':
        raise ValueError('layout units/up axis')
    records = layout.get('objects')
    if not isinstance(records, list) or len(records) != len(names):
        raise ValueError('layout object count')
    found = {}
    for row in records:
        name = row.get('instance')
        if name not in names or name in found:
            raise ValueError('layout object set')
        p, q = row.get('position'), row.get('rotation')
        if len(p) != 3 or len(q) != 4 or not all(math.isfinite(float(v)) for v in p + q):
            raise ValueError('nonfinite pose')
        if abs(sum(float(v) ** 2 for v in q) - 1) > 0.002:
            raise ValueError('nonunit quaternion')
        found[name] = row
    return found


def main(bank_path, manifest_path, summary_path):
    os.umask(0o077)
    bank = Path(bank_path).resolve(strict=True)
    if bank.stat().st_uid != os.getuid() or bank.stat().st_mode & 0o077:
        raise PermissionError('casebank must be judge-owned 0700')
    catalog = json.loads((bank / 'private_catalog.json').read_text())
    rows = [r for r in catalog if r['candidate_contract'] == 'COMPILED']
    if len(catalog) != 78 or len(rows) != 73 or len({r['case_id'] for r in catalog}) != 78:
        raise ValueError('confirmed 73-of-78 scope mismatch')
    # 1.6：把权威 case 列表写进 casebank/cases.json（运行时唯一读取的索引）
    (bank / 'cases.json').write_text(json.dumps({'schema': 'track2-casebank-index-v1',
        'scope': 'confirmed 73 cases', 'case_ids': [r['case_id'] for r in rows]},
        indent=1, ensure_ascii=False) + '\n', encoding='utf8')
    evaluator = TrustedContractExtension()
    asset_hashes, details, errors = {}, [], collections.Counter()
    tier_counts = collections.Counter()
    max_root_residual = 0.0
    for index, row in enumerate(rows):
        case_id = row['case_id']
        source = (bank / 'cases' / case_id / 'case.json').resolve(strict=True)
        if bank not in source.parents or source.is_symlink():
            raise ValueError('case containment')
        case = json.loads(source.read_text())
        item = {'case_id': case_id, 'ordinal': index, 'case_sha256': sha(source),
                'source_scene': case['source_case_id'], 'tier': case['contract']['tier'],
                'task_type': case['contract']['task_type'], 'predicates': [p['type'] for p in case['contract']['predicates']],
                'constraints': [c['kind'] for c in case['contract']['constraints']],
                'contract_sha256': digest(case['contract']),
                'target_layout_sha256': digest(case['contract']['target_layout']),
                'initial_layout_sha256': [digest(v) for v in case['initial_layouts']],
                'robot_base_sha256': digest([case['robot_base_candidate_m'], case['robot_base_candidate_q_wxyz'], case['initial_robot_qpos']]),
                'physics_parameters_sha256': digest(case['physics_parameters']),
                'pose_reference': case['pose_reference'], 'errors': []}
        try:
            evaluator.validate_task(case['contract'])
            names = set(case['objects'])
            if not names or len(case['initial_layouts']) != 5:
                raise ValueError('objects or five initials')
            goal = check_layout(case['contract']['target_layout'], names)
            initial_maps = [check_layout(v, names) for v in case['initial_layouts']]
            if digest(case['contract']['initial_layout']) != digest(case['initial_layouts'][0]):
                raise ValueError('contract initial mismatch')
            if len(set(item['initial_layout_sha256'])) != 5:
                raise ValueError('duplicate seeds')
            original = {v['instance']: v for v in case['source_goal_layout']['objects']}
            if set(original) != names:
                raise ValueError('source goal object set')
            if set(case['static_objects']) != {name for name in names if case['objects'][name]['static']}:
                raise ValueError('static object set')
            for name, body in case['objects'].items():
                if not body['hulls'] or body['mass_kg'] <= 0 or any(v <= 0 for v in body['inertia_kg_m2']):
                    raise ValueError('invalid collision/mass/inertia')
                if not all(math.isfinite(float(v)) for v in [body['mass_kg'], *body['inertia_kg_m2']]):
                    raise ValueError('nonfinite physics property')
                source_pose, new_pose = original[name], goal[name]
                rotated = qrotate(source_pose['rotation'], body['source_canonical_pivot_m'])
                expected = [float(source_pose['position'][i]) / 100 + rotated[i] for i in range(3)]
                residual = max(abs(expected[i] - float(new_pose['position'][i]) / 100) for i in range(3))
                max_root_residual = max(max_root_residual, residual)
                if residual > 1e-6:
                    raise ValueError('fixed asset root mismatch')
                for hull in body['hulls']:
                    path = (bank / hull['file']).resolve(strict=True)
                    if bank not in path.parents or path.is_symlink() or path.suffix.lower() != '.obj':
                        raise ValueError('asset containment/type')
                    key = str(path.relative_to(bank))
                    h = asset_hashes.get(key)
                    if h is None:
                        h = sha(path)
                        asset_hashes[key] = h
                    if h != hull['sha256']:
                        raise ValueError('collision asset hash')
            for pred in case['contract']['predicates']:
                if pred.get('object') and pred['object'] not in names:
                    raise ValueError('predicate target missing')
                if pred['type'] == 'P1':
                    region = pred['region_aabb_m']
                    if len(region) != 2 or any(len(v) != 3 for v in region) or any(region[0][i] >= region[1][i] for i in range(3)):
                        raise ValueError('invalid region')
                if pred['type'] == 'P3' and (pred.get('reference') not in names or pred['reference'] == pred['object']):
                    raise ValueError('relation reference missing')
            if len({(p['type'], p.get('object')) for p in case['contract']['predicates']}) != len(case['contract']['predicates']):
                raise ValueError('duplicate predicates')
            if not all(set(initial) == names for initial in initial_maps):
                raise ValueError('seed layout object set')
            tier_counts[item['tier']] += 1
        except Exception as error:
            item['errors'].append(type(error).__name__ + ': ' + str(error))
            errors[type(error).__name__] += 1
        details.append(item)
    from track2_judge.contract.profile import PROFILE
    confirmed=all(json.loads((bank/'cases'/r['case_id']/'case.json').read_text())['contract'].get('scoring_profile')==PROFILE for r in rows)
    # 1.6：不再产出任何 publish-time 哈希字段（catalog_sha256 / casebank_pin_sha256 /
    # asset_sha256 / private_manifest_sha256）。审计摘要只保留结构性事实。
    frozen = {'schema': 'track2-73-scope-judge-private-v1', 'scope': 'confirmed 73 cases from 78 source scenes',
              'cases': details,
              'scoring_profile':PROFILE if confirmed else 'diagnostic',
              'approval_status': 'SCORING_RULES_FROZEN; input interface and production switch WAIT_CONFIRM' if confirmed else 'diagnostic rules'}
    manifest = Path(manifest_path)
    manifest.write_text(json.dumps(frozen, sort_keys=True, separators=(',', ':'), allow_nan=False))
    manifest.chmod(0o600)
    summary = {'scope_cases': 73, 'source_scenes_audited': 78, 'per_case_audit_pass': sum(not d['errors'] for d in details),
               'per_case_audit_fail': sum(bool(d['errors']) for d in details), 'errors_by_type': dict(errors),
               'task_types': dict(collections.Counter(d['task_type'] for d in details)), 'tiers': dict(tier_counts),
               'predicate_types': dict(collections.Counter(p for d in details for p in d['predicates'])),
               'constraints': dict(collections.Counter(c for d in details for c in d['constraints'])),
               'unique_collision_assets_hash_verified': len(asset_hashes),
               'max_fixed_asset_root_residual_m': max_root_residual,
               'frozen_v3_modified': False, 'official_semantics_approved': confirmed}
    Path(summary_path).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    if summary['per_case_audit_fail']:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--casebank', required=True)
    parser.add_argument('--private-manifest', required=True)
    parser.add_argument('--summary', required=True)
    args = parser.parse_args()
    main(args.casebank, args.private_manifest, args.summary)
