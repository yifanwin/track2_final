"""Unmodified frozen V3 entrypoints. Never score Skill output.

D3：发布期哈希门禁已删除。冻结评分器的「未被改」由发布流程保证，
不在每次运行时重算（运行时证据链见 scoring/result_gate.py 的 artifact_sha256 账本）。
"""
import importlib.util
from pathlib import Path

path = Path(__file__).parent/'frozen_scorer/mesa_eval.py'
spec = importlib.util.spec_from_file_location('frozen_v3', path)
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

class FrozenVerifier:
    def verify(self, task, initial, final, config, event_evidence=None):
        if config['event_constraints']:
            if event_evidence is None or event_evidence.get('coverage')!='physics_substep' or set(event_evidence.get('event_counts',{}))!=set(config['event_constraints']):
                raise RuntimeError('trusted physical event evidence incomplete')
        events={} if not config['event_constraints'] else event_evidence['event_counts']
        result = v3.score_state(task, final, initial, config, events)
        return {**result, 'success':result['strict_success'],
                'coverage':{'P1':'NOT IMPLEMENTED','P2':'IMPLEMENTED','P3':'NOT IMPLEMENTED',
                            'P4':'NOT IMPLEMENTED','P5':'NOT IMPLEMENTED','C1':'DISPLACEMENT + ENABLED TRUSTED EVENT COUNTS'},
                'coverage_scope':'Frozen V3 total score only; supplemental predicates are diagnostics'}
