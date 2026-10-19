"""跑完 50 例的私有彩排；无队列、无回调、无选手数据。

单一权威 casebank（无 --casebank / --freeze-manifest 参数）。

⚠️ **默认口径 = 5 seed**（`release_entry.py` 的默认）—— 只有 5 seed 才出得了分
（冻结评分器 `mesa_eval.REPEATS["sim"] = 5`）。`--seeds` 是给**冒烟**用的：
少跑时 Gate 会标 `DIAGNOSTIC_PARTIAL_SEEDS_NO_SCORE`，本脚本也如实标成诊断。

## ★ 2026-10-19 八卡并行（见 `plan/Track2_八卡并行评测方案.md`）

`--workers N`（默认 1 = 串行，保持旧行为）：把 case 分给 N 个并发 worker，
每个 worker 用**独立的 Bundle profile**（`track2_official_runtime_workers/w<k>/`，
端口 19110+10k，GPU k）。50 例从 2.5 h 降到 ~20 min。

* 并发**只发生在执行侧** —— 每个 case 仍是一个独立 `release_entry` 子进程。
* 越界（`w >= 可用 worker 数`）会**明确报错**，不静默退化。
* `--workers > 1` 前必须先跑 `work/scripts/make_worker_bundles.sh N`。
"""
import argparse, json, os, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from track2_judge.cli.select_case import CASEBANK, check_casebank

REPEATS_SIM = 5

#: 并行 worker 的 Bundle 根目录（由 `work/scripts/make_worker_bundles.sh` 生成）。
#  ⚠️ worker Bundle **与主 Bundle 同级**（`track2_official_runtime_w<k>`）——
#  `nodes/run_player.sh` 从 `_HERE/../..` 反推 `_WENYIFAN` 找 python，
#  放进更深的子目录会推出错误的根 ⇒ `python not found`（exit 127）。
_WORKER_ROOT = Path(os.environ.get(
    "TRACK2_WORKER_ROOT",
    "/mnt/SSD_3TB/jusheng/track2_maniskill_staging_20261010/wenyifan"))
_WORKER_PREFIX = "track2_official_runtime_w"

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
# ★ 八卡并行：并发 worker 数（1 = 串行，保持旧行为）。
p.add_argument('--workers', type=int, default=1,
               help='并发 worker 数（默认 1 = 串行）。>1 需先跑 make_worker_bundles.sh')
# ★ 并行时用哪些**物理 GPU**（逗号分隔）。worker k 用 gpus[k]。
#   本机常有别的用户占卡（实测卡 0 被 VLLM 占 38GB），所以必须能指定。
p.add_argument('--gpus', default=None,
               help='并行用的物理 GPU 列表（逗号分隔）。默认 0,1,2,…（=worker 号）')
# 分阶段验证用：只跑前 N 个 case（正式评测不要带）。
p.add_argument('--limit', type=int, default=None,
               help='只跑前 N 个 case（分阶段验证用）。默认全 50')
a = p.parse_args()
os.umask(0o077)
out = Path(a.output).resolve()
out.mkdir(parents=True, mode=0o700, exist_ok=False)
case_ids = check_casebank(CASEBANK)
if a.limit is not None:
    if a.limit < 1:
        raise SystemExit(f"--limit 必须 >= 1，得到 {a.limit}")
    case_ids = case_ids[:a.limit]
release_entry = Path(__file__).with_name('release_entry.py')

workers = int(a.workers)
if workers < 1:
    raise SystemExit(f"--workers 必须 >= 1，得到 {workers}")

# worker k 用哪张**物理卡**。默认 worker 号本身（w0→卡0 …）。
gpu_list = ([int(g) for g in a.gpus.split(',') if g.strip()] if a.gpus
            else list(range(workers)))
if workers > 1 and len(gpu_list) < workers:
    raise SystemExit(f"--gpus 给了 {len(gpu_list)} 张卡，但 --workers={workers}")

def _worker_bundle(k: int) -> Path:
    """第 k 个 worker 的 Bundle 根（并行时必须存在）。"""
    return _WORKER_ROOT / f"{_WORKER_PREFIX}{k}"

if workers > 1:
    missing = [k for k in range(workers) if not _worker_bundle(k).is_dir()]
    if missing:
        raise SystemExit(
            f"--workers {workers} 需要 {workers} 份独立 Bundle，但缺 w{missing}。"
            f"先跑： bash work/scripts/make_worker_bundles.sh {workers}")
    if a.render_gpu is not None:
        raise SystemExit("--workers > 1 与 --render-gpu 不能同时用："
                         "GPU 由 worker 编号决定（w<k> → GPU k）")

# 本次预期每个 case 跑几条 episode 记录（★ 2026-10-18：取消 first/final，= seed 数）
seeds = ([int(s) for s in a.seeds.split(',') if s.strip()] if a.seeds else list(range(REPEATS_SIM)))
expected_episodes = len(seeds)
partial = len(seeds) != REPEATS_SIM

# ⚠️ `results` 按 **case 序号**预置占位，worker 回来按自己的 ordinal 落位 ——
#    这样并行完成顺序再乱，最终顺序也与串行**逐字一致**（聚合逻辑不用改）。
results: list = [None] * len(case_ids)
start = time.monotonic()


def run_case(i: int, case_id: str, k: int = 0) -> dict:
    """跑第 `i` 个 case，`k` 是**调用方显式指定的 worker 槽**。

    **并发安全**：只读全局（`a` / `out` / `release_entry` / …），
    唯一写入是 `results[i]`（由调用方在**主线程**落位）与该 case 自己的子目录。
    并发由**每 worker 一个线程、各自串行处理自己的 case 队列**实现 ——
    子进程本身才是并发的载体，线程只做调度（`subprocess.run` 释放 GIL）。

    ⚠️ **`k` 必须由调用方按「该线程独占的 slot」给**，不能在这里算
    `i % workers` —— 那样会**两个 case 同时用同一份 Bundle**（见派发处的说明）。
    """
    destination = out / ('case_%03d' % i)
    cmd = [sys.executable, '-B', str(release_entry), '--case-id', case_id,
           '--skill-zip', a.skill_zip, '--output', str(destination)]
    if a.seeds:
        cmd += ['--seeds', a.seeds]
    if not a.camera:
        cmd.append('--no-camera')
    # ★ 并行：每 worker 一个 Bundle profile（端口由 bundle 决定，GPU 按 `--gpus`）。
    #   串行（workers==1）时不传 —— 沿用主 Bundle，行为与旧版逐字一致。
    env = dict(os.environ)
    if workers > 1:
        env['TRACK2_BUNDLE_ROOT'] = str(_worker_bundle(k))
        # worker k → 物理卡 `gpu_list[k]`（默认 = k）。
        # ⚠️ 由 runtime 侧渲染进 `runtime.yaml` 的 `render_gpu` 生效；
        #    这里传 env 只是兜底（`--detach` 下节点其实读不到 env）。
        env['TRACK2_RENDER_GPU'] = str(gpu_list[k])
        env.pop('CUDA_VISIBLE_DEVICES', None)   # 由 runtime 侧按 render_gpu 选卡
    elif a.render_gpu:
        cmd += ['--render-gpu', a.render_gpu]
    with (out / ('case_%03d.private.log' % i)).open('x') as log:
        try:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                               timeout=7200, env=env)
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
    SCORABLE_TERMINAL_STATUS = ('success', 'failure', 'budget_timeout', 'skill_error')
    concluded = (len(d.get('episodes', [])) == expected_episodes
                 and all(x.get('status') in SCORABLE_TERMINAL_STATUS
                         for x in d.get('episodes', [])))
    if partial:
        valid = code == 0 and concluded
    else:
        valid = (code == 0 and concluded
                 and gate.get('publishable') is True
                 and d.get('scoring_status') == 'CONTINUOUS_TERMINAL_STATE')
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
    return {'ordinal': i, 'case_id': case_id,
            'status': 'PASS' if valid else 'FAIL',
            'episodes': len(d.get('episodes', [])),
            # ★ 2026-10-18：新分（终态连续评分，单 case 内部 100 分）。
            'score_100': d.get('extension_candidate_score_100'),
            # ⚠️ 旧分**已停用**（冻结件要求 first/final 配对），保留键只为兼容读端。
            'legacy_frozen_rule_score_100': d.get('frozen_rule_score_100'),
            'scoring_status': d.get('scoring_status'),
            'episodes_incomplete': incomplete,
            'episodes_infrastructure_timeout': infra,
            'action_count': (timings or {}).get('count'),
            'action_max_s': ((timings or {}).get('quantiles') or {}).get('max')}


def _write_summary() -> None:
    """把当前已完成的结果写进汇总产物（并行时每完成一个就刷一次，便于观察）。"""
    done = [x for x in results if x is not None]
    good = [x for x in done if x['status'] == 'PASS']
    n_cases = len(case_ids)
    summary = {'status': 'PASS' if len(good) == n_cases else 'RUNNING' if len(done) < n_cases else 'FAIL',
               'case_count': n_cases, 'completed_cases': len(done), 'pass_cases': len(good),
               'fail_cases': len(done) - len(good), 'episodes': sum(x['episodes'] for x in done),
               # ★ 总分 = 各 case **等权平均**（每 case 满分 2 分 = 100/n_cases）。
               'score_100': (sum(x['score_100'] for x in good) / n_cases
                             if len(good) == n_cases and all(x['score_100'] is not None for x in good)
                             else None),
               'score_per_case': round(100.0 / n_cases, 6) if n_cases else None,
               # ★ 口径如实标注 —— 1 seed 的产物**不是**正式成绩，别让它被误读
               'repeats': len(seeds), 'seeds': seeds,
               'scoring_status': ('DIAGNOSTIC_PARTIAL_SEEDS' if partial
                                  else 'CONTINUOUS_TERMINAL_STATE'),
               'scoring_note': ('只跑 %d 个 seed（正式口径 %d）—— 冒烟，不构成成绩'
                                % (len(seeds), REPEATS_SIM)) if partial else None,
               'strategy_scope': 'internal legal-action weak Skill; not difficulty calibration',
               'production_callback_sent': False, 'wall_s': round(time.monotonic() - start, 2)}
    (out / 'sanitized_summary.json').write_text(json.dumps(summary, indent=2))
    (out / 'private_results.json').write_text(json.dumps(done, indent=2))


# ── 派发：串行 or 并行
if workers == 1:
    for i, case_id in enumerate(case_ids):
        results[i] = run_case(i, case_id, 0)
        print(json.dumps(results[i]), flush=True)
        _write_summary()
else:
    # ★ 并行派发：**每 worker 一个线程，各自串行处理自己的 case 队列**。
    #
    #   ⚠️ **不能用「一次性全提交给线程池 + `k = i % workers`」** ——
    #   那有真实竞态（2026-10-19 发现）：池里的线程完成顺序被打乱后，
    #   慢 case（占着 w0）还在跑时，别的空闲线程就会领走**同样取模到 w0**
    #   的下一个 case（如 case 0 与 case 4 在 `workers=4` 时都是 w0）⇒
    #   **两个 case 同时用同一份 Bundle / 同一个 gateway 端口**，
    #   表现为 `gateway never ready`、`concurrency 1 exhausted` 等
    #   极其难定位的故障。
    #
    #   现在改成「按 worker 分队列、队列内串行」：第 k 个 worker 只处理
    #   `i % workers == k` 的 case，且**同一时刻只跑一个** ⇒
    #   Bundle / 端口 / GPU 的独占性由**结构**保证，不依赖调度运气。
    #   （`--limit` 下各队列长度可能不均，多出来的 worker 队列为空、自然退出。）
    assignments = {
        k: [(i, cid) for i, cid in enumerate(case_ids) if i % workers == k]
        for k in range(workers)
    }
    lock = threading.Lock()

    def _worker_loop(k: int) -> None:
        for i, cid in assignments[k]:
            row = run_case(i, cid, k)
            with lock:
                results[i] = row
                print(json.dumps(row), flush=True)
                _write_summary()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for k in range(workers):
            pool.submit(_worker_loop, k)

# Q14：把 50 例的动作耗时汇总成一份可直接看分布的产物
_timings = []
for i in range(len(case_ids)):
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

_done = [x for x in results if x is not None]
_good = [x for x in _done if x['status'] == 'PASS']
sys.exit(not (len(_good) == len(case_ids)))
