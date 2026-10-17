"""唯一的一例本地私有入口。单一权威 casebank，无 SHA 链。

CASEBANK 由本文件位置决定（`track2_final/judge_private/casebank`），**不接受命令行覆盖**。
启动时只做结构性检查（无哈希）—— 见 `check_casebank`。
输入契约与生产开关的确认状态另见 `diagnostics/freeze_cases.py` 产出的审计摘要。
"""
import argparse, copy, json, os, re
from pathlib import Path
from typing import Any

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
#: 预渲染的参考图。**与 casebank 平级、不在它内部** —— casebank 是
#: `2casefix` 的只读证据源（红线）。
REFERENCE_ROOT = Path(__file__).resolve().parents[2] / "judge_private" / "reference_images"
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
        casebank=None, oracle_plan=None):
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
    # 模式 B（Q4 + Q10）：公开输入里**没有任何数值目标**。
    # 目标参考图由 `render_reference_image` 用 `target_layout` 现渲染
    # （casebank 里没有图像资产），机位与实时帧一致。
    if task.get('scoring_profile') == PROFILE or candidate_public_goals:
        # ⚠️ 参考图**读预渲染的**，不在这里现渲染 —— 渲染要 GPU，而回归/CPU 档
        #    会把 `CUDA_VISIBLE_DEVICES` 清空，现渲染会直接崩。
        #    73 张图由 `work/scripts/prerender_reference_images.py` 离线生成，
        #    落在 `judge_private/reference_images/`（**不放进 casebank** —— 它只读）。
        from track2_judge.observation.reference_image import load_reference_image
        reference = load_reference_image(REFERENCE_ROOT, case_id)
        if reference is None:
            raise RuntimeError(
                f"no pre-rendered reference image for {case_id}; run "
                f"work/scripts/prerender_reference_images.py")
        public = project(task, reference_image=reference)
    else:
        public = {'text_instruction': 'internal legal-action diagnostic'}

    # ── 裁判侧 Oracle 的 GT 通路（第 6 步）
    #
    # 原来的 `oracle_driver.py` 把规划结果**塞进 `text_instruction`**
    # （`PrivateOracleInput.observation()`），绕过了这个入口。
    # 现在改成**显式参数**：规划结果作为一个独立字段进公开输入，
    # 由 `ObservationProvider` 的白名单放行（`oracle_plan`）。
    #
    # **为什么不做成 `track2.admin.oracle_plan` 那个 Tool**（计划原稿的设想）：
    # 选手 manifest 的 `tool_profile` 只能声明**官方三个** Tool，
    # `track2.admin.*` 会被准入检查拒掉。而 Oracle **既是选手又是裁判** ——
    # 这双重身份没法用单一 allowlist 表达。所以走「入口的显式参数」。
    # `oracle_plan=True` = **在本入口里现做 GT 规划**（推荐）。
    # 直接传 dict = 用外部算好的 plan（兼容旧用法）。
    #
    # ⚠️ 现做规划要**入口自己的 env** —— 这正是第 6 步要修的：
    #    老实现自己另造一个 env 算 plan，与入口造的那个不是同一个，
    #    于是「Oracle 成功但入口会拒绝该 case」这种矛盾看不出来。
    oracle_payload = None
    if oracle_plan is True:
        oracle_payload = True                      # 延后到 reset 之后（那时才有 env）
    elif oracle_plan is not None:
        if not isinstance(oracle_plan, dict):
            raise ValueError('oracle_plan must be True or a mapping')
        public = {**public, 'oracle_plan': json.loads(
            json.dumps(oracle_plan, allow_nan=False))}
    camera = None
    camera_specs = None
    if diagnostic_rgb:
        import numpy as np
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        # ⚠️ 机位**不再从答案里取**（原来是从 P2 的 `target_pose` 取，
        #    那是模式 A 的做法，而且模式 B 下公开输入里已经**没有**
        #    `target_pose` 了）。改成与参考图/实时帧同一套固定机位：
        #    从**初始布局**算，与目标无关。
        from track2_judge.scenario.env_factory import fixed_camera_from_bundle
        # 机位只依赖初始布局；`_fit_distance` 对 5 个 seed 的布局给出几乎相同的
        # 结果（毫米级），所以固定用 seed 0 —— 与预渲染的参考图一致。
        camera_specs = fixed_camera_from_bundle(bundle, 0)
        public['text_instruction'] = ('DIAGNOSTIC_RGB_CAMERA=' +
                                      json.dumps({k: v for k, v in camera_specs.items()
                                                  if k != 'up'},
                                                 separators=(',', ':')) +
                                      '\n' + public['text_instruction'])

        def camera(env):
            env.scene.update_render()
            c = env._human_render_cameras['diagnostic_public_rgb']
            c.capture()
            rgb = c.camera._render_cameras[0].get_picture('Color')
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb[:, :, :3] * 255, 0, 255).astype(np.uint8)
            return rgb[:, :, :3]

    # ★ Oracle 通道：公开输入的**白名单**要多放一个 `oracle_plan`。
    #   **只在 `oracle_plan=True` 时**加 —— 普通评测的选手包拿不到它。
    adapter_public_fields = tuple(public) + (
        ('oracle_plan',) if oracle_payload is True else ())

    def _planning_factory(task, seed, mode):
        cp = next((p['checkpoints'] for p in task['predicates'] if p['type'] == 'P5'), [])
        adapter = ManiSkillSimulatorAdapter(
            lambda: create_case_env(bundle, bank, seed,
                                    allow_unapproved_diagnostics=allow_unapproved_diagnostics,
                                    diagnostic_camera=camera_specs),
            snapshot=physical_snapshot, spatial_provider=spatial_snapshot,
            metadata={n: {'pivot_m': [0, 0, 0]} for n in bundle['objects']},
            scene_settings=SETTINGS, public_fields=adapter_public_fields,
            observation_provider=ObservationProvider(approved_fields=adapter_public_fields,
                                                     camera=camera,
                                                     camera_required=diagnostic_rgb),
            checkpoints=cp, recovery=mode == 'final', initial_validation=validate_initial_scene)
        if oracle_payload is True:
            # ★ 用**这个 adapter 的 env** 做 GT 规划。
            #   env 只在 reset 之后才存在，所以先 reset 一次（不带 GT），
            #   规划完由 `dispatch_factory` 再 reset 一次（带 GT）——
            #   两次 reset 用的是**同一个 bundle / 同一个 adapter**，
            #   所以规划看到的物理与执行时的初始状态是同一个。
            #
            #   `_ManiSkillSimulatorAdapter__env` 是私有属性，但这里是裁判侧
            #   同包访问；它也是唯一能拿到「入口自己的 env」的入口。
            adapter.reset(seed=seed, public_task_inputs=public)
            env = adapter._ManiSkillSimulatorAdapter__env
            if env is None:
                raise RuntimeError('oracle planning needs a reset environment')
            from track2_judge.oracle.planning import (
                OraclePlanningError, build_oracle_plan, find_target,
                oracle_executor_payload, serializable,
            )
            plan = build_oracle_plan(env, task, find_target(task))
            payload = serializable(oracle_executor_payload(plan))
            # ⚠️ `planning.serializable` 先落成纯 JSON，`_strict` 再挡一道
            #    NaN/Infinity（JSON 允许它们，但下游会当成非法数）。
            _strict(payload)
            return adapter, payload
        return adapter, None

    factory_returns_plan = oracle_payload is True

    # ⚠️ `_planning_factory` 的签名**永远返回 `(adapter, plan_or_None)`** ——
    #    Oracle 通道需要那个第二项。但**普通评测**下 runner 要的是 adapter 本身：
    #    直接把 `_planning_factory` 交给 runner 会让它拿到一个 tuple，
    #    在 `adapter.reset(...)` 处报 `AttributeError: 'tuple' object has no
    #    attribute 'reset'`（第 6 步引入，第 7 步回归时暴露）。
    #    所以普通路径必须过一层「只取 adapter」的包装。
    #
    # ⚠️ 用**默认参数**绑 `_planning_factory`，不写成闭包自由变量 ——
    #    自由变量会在下面 `factory = ...` 之后解析到自己，变成无限递归
    #    （`RecursionError`，第 6 步踩过两次）。
    def _adapter_only(task, seed, mode, _plan=_planning_factory):
        return _plan(task, seed, mode)[0]

    factory = _adapter_only
    factory.source = 'ACTUAL selected TTG candidate, source collision assets; candidate physics'
    # Confirmed public targets cross only the explicit JSON projection.
    # Input serialization remains a candidate profile; private bank never crosses.
    if factory_returns_plan:
        # ⚠️ 一个必须绕开的顺序问题：runner 是
        #
        #     adapter = factory(task, seed, mode)      ← 我们在这里算 GT plan
        #     adapter.reset(seed, public_task_inputs=public_inputs)   ← 它自己又 reset 一次
        #
        # 也就是说**我们没法在 factory 里把 plan 塞进公开输入** ——
        # runner 随后那次 reset 会把 `public_inputs`（不含 plan）传进去。
        #
        # 而且 plan **本来就与 seed 有关**（`initial_layouts` 有 5 条，seed 是索引），
        # 所以也不能在循环外算一次。
        #
        # 做法：factory 把 plan 存进一个**共享槽**，`public_inputs` 用一个
        # 「deepcopy 时才去槽里取」的 dict 子类 —— runner 恰好在那之后立刻 deepcopy。
        slot: dict[str, Any] = {}
        _base_public_inputs = public_inputs_factory(public)

        # ⚠️ 用**默认参数**把真正的规划工厂绑进去。写成闭包自由变量 `factory`
        #    会在下面 `factory = dispatch_factory` 之后解析到自己 —— 无限递归，
        #    报的是 `RecursionError: maximum recursion depth exceeded`
        #    （踩过两次，两次都看不出是闭包问题）。
        def dispatch_factory(task, seed, mode, _plan=_planning_factory):
            adapter, plan = _plan(task, seed, mode)
            if plan is None:
                raise RuntimeError('oracle planning did not produce a plan')
            slot['plan'] = plan
            return adapter
        dispatch_factory.source = factory.source
        factory = dispatch_factory
        public_inputs_obj = _PlanInjectingInputs(_base_public_inputs, slot)
    else:
        public_inputs_obj = public

    return CandidateRunner(factory, package_zip=skill_zip, isolated=True,
                           contract_extension=engine, timeout_s=skill_timeout_s).run(
        {'config': load_config(), 'tasks': [task]}, output, public_inputs_obj)


def public_inputs_factory(public):
    """公开输入的**可变副本**（runner 会 deepcopy 它）。"""
    return dict(public)


class _PlanInjectingInputs(dict):
    """`deepcopy` 时把工厂刚算出的 GT plan 塞进去。

    `CandidateRunner` 的调用顺序是「factory → `copy.deepcopy(public_inputs)`
    → `adapter.reset(..., public_task_inputs=...)`」，所以 deepcopy 那一刻
    `slot['plan']` 里已经是**这个 seed/mode 的** plan。
    """

    def __init__(self, base, slot):
        super().__init__(base)
        self._slot = slot

    def __deepcopy__(self, memo):
        value = dict(self)
        plan = self._slot.get('plan')
        if plan is not None:
            value['oracle_plan'] = copy.deepcopy(plan, memo)
        return value


def _strict(value):
    """拒绝 NaN / Infinity —— `json.loads` 会接受它们，但下游 schema 不认。"""
    def bad(token):
        raise ValueError('non-finite number in oracle plan: ' + token)
    json.loads(json.dumps(value, allow_nan=False), parse_constant=bad)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--case-id')
    p.add_argument('--skill-zip')
    p.add_argument('--output')
    p.add_argument('--allow-unapproved-diagnostics', action='store_true')
    p.add_argument('--candidate-public-goals', action='store_true')
    p.add_argument('--oracle-plan', help='裁判侧 Oracle 的 GT 规划结果（JSON 文件）')
    p.add_argument('--list-cases', action='store_true', help='只做结构性检查并打印 case 列表')
    a = p.parse_args()
    if a.list_cases:
        ids = check_casebank()
        print(json.dumps({'status': 'PASS', 'cases': len(ids), 'case_ids': ids}))
        raise SystemExit(0)
    missing = [n for n in ('case_id', 'skill_zip', 'output') if not getattr(a, n)]
    if missing:
        p.error('缺少参数: ' + ', '.join('--' + n.replace('_', '-') for n in missing))
    oracle_plan = (json.loads(Path(a.oracle_plan).read_text(encoding='utf8'))
                   if a.oracle_plan else None)
    r = run(a.case_id, a.skill_zip, a.output,
            allow_unapproved_diagnostics=a.allow_unapproved_diagnostics,
            candidate_public_goals=a.candidate_public_goals,
            oracle_plan=oracle_plan)
    print(json.dumps({'episodes': len(r['episodes']),
                      'candidate_score': r.get('extension_candidate_score_100'),
                      'official_score': None,
                      'scoring_status': r.get('scoring_status', 'CANDIDATE ONLY')}))
