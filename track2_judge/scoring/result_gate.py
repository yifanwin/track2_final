"""Judge-only fail-closed artifact gate. Task failure is a valid verified outcome.

Execution faults, missing evidence and integrity failures never issue a score.
The receipt is written only after trusted independent predicate replay succeeds.
No network or production callback is implemented.
"""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

SCHEMA='track2-result-gate-v1'

#: **可评分的执行终态**（2026-10-18 组委会口径）。
#
#   * `success` / `failure` —— 原有两类，判定来自物理终态。
#   * `budget_timeout`     —— **正常预算耗尽**（600 s 单次 run() 墙钟 / 控制步预算）。
#     语义是「到时间 → 停止执行 → 冻结当前物理状态 → 照常按 P1/P2/C1 判分」。
#     它的判定输入同样是物理终态，所以与上面两类一样可评分。
#
#   ⚠️ **不要写成 `status == "timeout" → 一律放行`。** 链路故障必须仍然阻断：
#     * `infrastructure_timeout` —— 丢包 / 503 / 动作超时且物理状态不明
#       （见 `sandbox/rpc.py` 的 `BudgetTimeout` 与 `toolapi/client.py` 的
#       `ActionStoppedUnknown`）。它只是**碰巧也是超时**，但与比赛预算无关。
#   见 report/requirement/Track2_评分口径变更_效率项改超时.md §1.3。
#   * `skill_error` —— ★ F2（2026-10-18）：**选手包自己抛了异常**。
#     它不是链路故障，按冻结的物理终态照常判分（可评分）。
SCORABLE_TERMINAL_STATUS = frozenset({'success', 'failure', 'budget_timeout', 'skill_error'})

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    temp=path.with_name(path.name+'.pending')
    with temp.open('w',encoding='utf8') as stream:
        json.dump(value,stream,allow_nan=False,sort_keys=True);stream.flush();os.fsync(stream.fileno())
    temp.chmod(0o600);os.replace(temp,path)

def block(output, reason):
    result={'schema':SCHEMA,'status':'BLOCKED','publishable':False,'reasons':[reason],
            'score_100':None,'official_score':None}
    try:atomic_json(Path(output)/'result_gate.json',result)
    except OSError:
        result['reasons'].append('ARTIFACT_STORAGE_FAILURE')
        sys.stderr.write(json.dumps(result)+'\n')
    return result

def write_receipt(output, score_name, *, expected_runs, verifier):
    root=Path(output)
    names=['suite.json','runs.json','artifacts/evidence.json','artifacts/runner_manifest.json',
           'diagnosis.json',score_name]
    runs=json.loads((root/'runs.json').read_text())['runs']
    names += [r[k] for r in runs for k in ('initial_layout','final_layout')]
    names += ['extension_private_evidence.json'] if verifier=='trusted_contract_extension' else []
    receipt={'schema':'track2-verification-receipt-v1','verified':True,'verifier':verifier,
             'expected_runs':expected_runs,
             'artifact_sha256':{n:sha(root/n) for n in names}}
    atomic_json(root/'verification_receipt.json',receipt)

class ResultGate:
    def __init__(self, source=None): self.source=Path(source or __file__).resolve().parent if source is None else Path(source).resolve()
    def check(self, output):
        root=Path(output).resolve();reasons=[];score=None
        def reject(code):
            if code not in reasons: reasons.append(code)
        try:
            if os.name=='posix' and (root.stat().st_uid!=os.getuid() or root.stat().st_mode&0o077): reject('OUTPUT_PERMISSIONS')
            required=['suite.json','runs.json','diagnosis.json','artifacts/evidence.json',
                      'artifacts/runner_manifest.json','verification_receipt.json']
            for rel in required:
                p=root/rel
                if p.is_symlink() or not p.is_file(): reject('MISSING_'+rel.replace('/','_').upper())
            if reasons: raise ValueError('required evidence absent')
            suite=json.loads((root/'suite.json').read_text());runs=json.loads((root/'runs.json').read_text())['runs']
            diagnosis=json.loads((root/'diagnosis.json').read_text());evidence=json.loads((root/'artifacts/evidence.json').read_text())
            receipt=json.loads((root/'verification_receipt.json').read_text());record_manifest=json.loads((root/'artifacts/runner_manifest.json').read_text())
            # ★ 2026-10-18 取消 first/final：每 seed 1 次（不再 `*2`）。
            expected=len(suite['tasks'])*suite['config']['repeats']
            if receipt.get('schema')!='track2-verification-receipt-v1' or receipt.get('verified') is not True or receipt.get('expected_runs')!=expected or receipt.get('verifier') not in ('trusted_contract_extension','frozen_v3'): reject('VERIFIER_FAILED')
            if len(runs)!=expected or record_manifest.get('run_count')!=expected or len(evidence.get('runs',[]))!=expected: reject('RECORDER_INCOMPLETE')
            # 2026-10-18：**四种可评分终态** —— `SCORABLE_TERMINAL_STATUS`
            #   （success / failure / budget_timeout / skill_error）。
            # `budget_timeout` 放行；`infrastructure_timeout` 落进 EXECUTION_EXCEPTION。
            if len(diagnosis.get('episodes',[]))!=expected or any(e.get('status') not in SCORABLE_TERMINAL_STATUS for e in diagnosis.get('episodes',[])): reject('EXECUTION_EXCEPTION')
            # ① 兜底（2026-10-18）：provider 侧动作账本不完整。
            #
            #   这一条**独立于 status** —— 专抓「选手把动作异常 try/except 吞掉、
            #   然后正常返回」那一类：那时 episode 的 status 看着是
            #   `success`/`failure`（可评分），但 provider 侧有「有 start 无 finish」
            #   的动作。加速方案的 `infrastructure_timeout` 靠异常分类，够不着它。
            #   见 plan/Track2_丢包自锁修复方案.md §3.1/§3.2。
            if any(e.get('execution_integrity') == 'INCOMPLETE'
                   for e in diagnosis.get('episodes',[])): reject('ACTION_EXECUTION_INCOMPLETE')
            # ★ 2026-10-18：新词是 `CONTINUOUS_TERMINAL_STATE`（终态连续评分）；
            #   旧词 `FROZEN_AUTHOR_PDF_RULES` 保留兼容历史产物。
            if diagnosis.get('scoring_status') not in ('CONTINUOUS_TERMINAL_STATE','FROZEN_AUTHOR_PDF_RULES',None) or diagnosis.get('candidate_status') in ('EXECUTION_FAILURE',): reject('EXECUTION_EXCEPTION')
            ledger=receipt.get('artifact_sha256',{})
            for rel in required[:-1]:
                if rel not in ledger: reject('RECEIPT_INCOMPLETE')
            for rel,expected_hash in ledger.items():
                p=root/rel
                if Path(rel).is_absolute() or '..' in Path(rel).parts or p.is_symlink() or root not in p.resolve().parents:
                    reject('EVIDENCE_PATH_REJECTED');continue
                if not p.is_file() or sha(p)!=expected_hash: reject('EVIDENCE_HASH_MISMATCH')
            by_id={e['run_id']:e for e in evidence.get('runs',[])};seen=set();tasks={t['task_id']:t for t in suite['tasks']}
            for run in runs:
                key=(run['task_id'],run['seed'])
                if key in seen or run['task_id'] not in tasks: reject('RUN_IDENTITY_INVALID')
                seen.add(key);ev=by_id.get(run['run_id'],{});verification=ev.get('verification',{})
                # 同上：四条可评分终态（含 `budget_timeout` / `skill_error`）。⚠️ 这一条查的是
                # **运行记录里的 status**（`recorder` 落盘的那份），不是选手自报的。
                if ev.get('status') not in SCORABLE_TERMINAL_STATUS or ev.get('error'): reject('EXECUTION_EXCEPTION')
                # 谓词判定仍然只有 success/failure —— `budget_timeout` 只是
                # 「执行怎么结束的」，判定本身照常二值。
                if verification.get('status') not in ('success','failure') or not isinstance(verification.get('predicates'),list) or not verification['predicates'] or any(p.get('status') not in ('success','failure') for p in verification['predicates']): reject('VERIFIER_FAILED')
                if type(run.get('elapsed_s')) not in (int,float) or not math.isfinite(run['elapsed_s']) or run['elapsed_s']<=0: reject('TIMING_INVALID')
                names=set()
                for field in ('initial_layout','final_layout'):
                    rel=run[field];p=root/rel
                    if rel not in ledger or p.is_symlink() or root not in p.resolve().parents: reject('STATE_INCOMPLETE');continue
                    value=json.loads(p.read_text());objects=value.get('objects',[])
                    actual={o['instance'] for o in objects}
                    goal={o['instance'] for o in tasks[run['task_id']]['target_layout']['objects']}
                    if not objects or len(actual)!=len(objects) or actual!=goal: reject('STATE_INCOMPLETE')
                    for o in objects:
                        if len(o.get('position',[]))!=3 or len(o.get('rotation',[]))!=4 or any(type(x) not in (int,float) or not math.isfinite(x) for x in o['position']+o['rotation']): reject('STATE_INCOMPLETE')
                    if field=='initial_layout':names=canonical(value)
            score_name='extension_candidate_score.json' if 'extension_candidate_score.json' in ledger else 'frozen_score_diagnostic.json'
            if score_name not in ledger: reject('SCORER_OUTPUT_MISSING')
            else:
                report=json.loads((root/score_name).read_text())
                # ★ 2026-10-18：新评分报告顶层是 `candidate_score_100`（终态连续评分）；
                #   旧格式曾是 `frozen_rule_score_100` / `score_100`（已停用）。
                score=report.get('candidate_score_100',report.get('frozen_rule_score_100',report.get('score_100')))
                if type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=100: reject('SCORER_OUTPUT_INVALID')
            if diagnosis.get('gt_access_status')!='PASS': reject('GT_READ_EXCEPTION')
        except Exception as exc:
            reject('GATE_EVIDENCE_ERROR_'+type(exc).__name__)
        result={'schema':SCHEMA,'status':'PASS' if not reasons else 'BLOCKED','publishable':not reasons,
                'reasons':reasons,'score_100':score if not reasons else None,'official_score':None,
                'receipt_sha256':sha(root/'verification_receipt.json') if (root/'verification_receipt.json').is_file() else None}
        try:atomic_json(root/'result_gate.json',result)
        except OSError:
            result.update(status='BLOCKED',publishable=False,score_100=None)
            result['reasons'].append('ARTIFACT_STORAGE_FAILURE')
            sys.stderr.write(json.dumps(result)+'\n')
        return result
