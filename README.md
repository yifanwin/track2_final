# track2_final —— Track 2 评测器的包化派生物

本目录是 `experiments/Frozen73_Final_Readiness_20261014/Frozen73_Final_Candidate_v1_2casefix`
（`READY_FOR_REVIEW` 的证据源）的**派生物**。

> **红线：`2casefix` 本身不被改动。** 每次交付前用
> `cd <2casefix> && sha256sum -c SHA256SUM` 复核（8734 项应全 OK）。

## 目录

```text
track2_final/
├── judge_private/                  # 裁判私有（0700），不进任何交付包
│   └── casebank/
│       ├── cases.json              # ★ 单一权威索引（73 个 case_id，无哈希）
│       ├── cases/<case_id>/        #   case.json + hulls/
│       ├── private_catalog.json    #   78 源场景审计元数据（冻结评分器旁证）
│       └── frozen_case_hashes.json #   历史遗留：发布期哈希链的产物，运行时不读
│
├── track2_judge/                   # 主包（原 63 个平铺 .py）
│   ├── cli/          # select_case（唯一入口）/ run_suite / release_entry
│   ├── scenario/     # ManiSkill / PhysX 场景与机器人
│   ├── observation/  # 观测、公开输入投影
│   ├── toolapi/      # ★ Tool 契约层（第 2 步的落点）
│   ├── sandbox/      # 选手进程隔离（三个独立可执行文件在此）
│   ├── contract/     # 任务合同与判定
│   ├── scoring/      # 评分与门禁（frozen_scorer 原样搬运，不可改）
│   ├── runner/       # 编排与记录
│   ├── diagnostics/  # 诊断（不进主链）
│   ├── util/
│   └── model/ upstream/ skills/   # 按 ROOT 语义留在包根
│
├── track2_tests/                   # 测试集中放
│   ├── fixtures/                   # 合成 Skill 与合成套件
│   └── server/
│
└── reports/legacy_evaluator/       # 原 evaluator 的文档与审计摘要
```

## 用法

```bash
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=''    # 本机 CPU 物理

# 结构性检查 + 列出 73 例
python track2_judge/cli/select_case.py --list-cases

# 跑单例（唯一入口，无 --casebank / --freeze-manifest）
python track2_judge/cli/select_case.py \
    --case-id 001_real_world_images__real_image_1 \
    --skill-zip path/to/skill.zip \
    --output /tmp/run_001
```

## 已完成的两步

### 第 1 步（2026-10-15）：目录分包 + 去 SHA + 单一 casebank

1. **目录分包**：63 个平铺 `.py` → 11 个职责包
2. **去 SHA 链**（D2 / D3）：删发布期哈希门禁与逐 case 哈希比对
3. **单一权威 casebank**：`CASEBANK` 由代码位置决定，启动时只做结构性检查

记录：`wenyifan/work/reports/50_step1_done.md`

### 第 2 步（2026-10-16）：toolapi 落地

`track2_judge/toolapi/` 从 1 个模块变成 6 个（1796 行）：

| 文件 | 职责 |
|---|---|
| `validation.py` | 最小 JSON Schema 校验器（环境无 `jsonschema` 包） |
| `schema.py` | **三个 ToolSpec 声明** —— 唯一权威来源 |
| `gateway.py` | 本地 Forge Gateway（Query + Action 两套路由） |
| `client.py` | 选手侧 `tools` 对象 + `SandboxBridge` + `EpisodeTools` |
| `dispatch.py` | 原 `adapter.py` + `GatewayExecutor` 适配层 |

三个 ToolSpec 的冻结哈希（PAOS 的 `canonical_sha256`）：

```
track2.env.observe               0c089c48072b5ff20bd38571df4f4d4ebc307fc8596c9872cdb387d7e5838bc2
track2.robot.joint_control       b4bf870877b16169d1f49ff5290eb3d64a005191bcd46d0221d2cfcd142c2edf
track2.robot.gripper_control     b7070934d01283a2bc689ab7ad4554a9e92b4a7ada748aa3ad1fc66d760bfa50
```

**Action 治理**（Q9 / Q12 / Q13）：三层语义刻意分离 ——
`cancel_status`（请求处理了没）/ `cancel_effective`（取消真的阻止了物理没）/
`execution_status`（物理上停没停）。等不到执行体退出就报 `unknown`，**绝不伪造 `cancelled`**。

记录：`wenyifan/work/reports/51_step2_done.md`

### 删掉的（§4.6.1）

| 位置 | 内容 |
|---|---|
| `cli/select_case.py` | `FROZEN_73_MANIFEST_SHA256` + 5 道哈希门禁 |
| `cli/verify_manifest.py` + `SHA256SUMS.json` | 整文件 |
| `cli/run_suite.py` | manifest 二次硬编码 |
| `util/environment_lock.py` + `.json` + `requirements_lock.txt` | 环境锁三件套（D3） |
| `scoring/scorer_bridge.py` | `SCORER_SHA256` 与 raise |
| `scoring/result_gate.py` | `SCORER_SHA256` 四处 |
| `diagnostics/check_install.py` | `frozen_hash` 检查 |
| `scenario/env_factory.py` | hull sha assert（D2） |
| `runner/candidate_runner.py` | Tool 白名单 + `tool_profile` 字符串门禁（第 2 步由声明集合取代） |

### 保留的（§4.6.2）

单次运行的证据完整性 / 运行内证据链 / 沙箱运行时清单 / PNG blob 完整性 /
防路径穿越 —— 这些都**不是**版本链，而是运行期完整性。

## 单一权威 casebank 的结构性检查（无哈希）

1. `CASEBANK` 存在、属主是裁判、`mode & 0o077 == 0`
2. `cases.json` 列出恰好 73 个互不重复的 `case_id`
3. 每个 `cases/<case_id>/case.json` 存在、≤8 MiB、不是符号链接
4. 所有 hull 路径包含在 `CASEBANK` 内且不是符号链接
5. 每例的 `contract` 通过 `TrustedContractExtension.validate_task`

这五条**全部与版本无关** —— 不会再出现「改一个 case 就要同步四处哈希」。

## 已知的中间状态（第 3 步接管）

| 项 | 现状 |
|---|---|
| `track2_judge/toolapi/gateway.py` 的 `GatewayServer` | 已可用，但还没接进 Dora flow 的 gateway 节点 |
| 沙箱网络隔离 | **仍是 `--net`**，选手不能直连 HTTP Gateway；靠 `SandboxBridge` 借评测器进程的手 |
| `HttpTransport` | 已实现并测过，等第 4 步去掉网络隔离后启用 |
| `track2_official_runtime/` | 未建（第 3 步） |
| `ACTION_TIMEOUT_S` | `30.0`，标 `PROVISIONAL_PENDING_MEASUREMENT`（Q14 待 73 case 实测） |
| `maniskill_sim` 节点 | 未开发（唯一的必须新开发项） |
