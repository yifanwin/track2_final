# Frozen73 Release Hardening Report

日期：2026-10-14（北京时间）。服务器：73F3-8x4090-134；用户 jusheng。范围：`/mnt/SSD_3TB/jusheng/track2_maniskill_staging_20261010` 的私人候选发布与实验域。

**当前状态：NOT_READY。发布、依赖、ResultGate、输入占位与指定安全检查已加固；尚未获得原 Frozen73 TTG 对象的成功抓持/搬运阳性证据，因此不标记 READY_FOR_PRODUCTION_REVIEW。**

该结论不是重新冻结评分，也不是要求批准新的评分规则。原评分公式、P1/P2/C1 定义、73 个 Case 文件与 8,470 个资产均未改变。剩余负责人确认事项仅为输入接口协议、正式生产切换权限。以下物理证据缺口属于技术验收，不列为新的负责人规则确认项。

## 1. 已修复事项

| 事项 | 实际改动与验证 |
| --- | --- |
| Frozen73 ZIP 名称混淆 | 建立服务器唯一 `release/evaluator.zip`，配套 SHA256SUM、VERSION、DEPLOYMENT.md；旧 Frozen73 与 73Case Review ZIP 只留在 staging/archive，另存原名与摘要的归档台账 |
| Pillow 声明与服务器不同 | 新版声明统一为实际已实测且离线可安装的 Pillow 12.3.0；未降级系统或修改原 env_ready；完整依赖锁可在新 env_locked 中离线复建 |
| 环境检查仅查“存在” | 增加完整版本锁、Python/PhysX 版本与 SAPIEN 物理库摘要检查；唯一运行入口在运行前检查清单和环境，失败阻断 |
| 成绩门禁缺少执行完整性 | 新增 ResultGate，要求可信重放 receipt、完整 Recorder、状态与配对、正常执行状态、GT 读取成功、Scorer 摘要和证据摘要一致；失败返回 BLOCKED、原因和空分数 |
| 已批准异常可构造成功回调 | legacy_callback_payload 必须重新检查真实输出目录的 ResultGate；拒绝 EXECUTION_FAILURE、超时、分数与可信评分不一致，以及无门禁的调用 |
| 文件写失败可能留下不明确结果 | 原子写 receipt/gate；写失败时门禁返回 BLOCKED、score=null，并将稳定原因写入裁判 stderr；不返回可发布成绩 |
| 初始化混入 Skill 用时 | 将 factory/reset/运行时准备与实际 Skill 执行分开；计分 elapsed_s 使用实际执行段，效率公式和 Tref 不变 |
| 慢物理服务消耗 Skill 计算预算 | 区分 tool_service_s 与 Skill 计算预算；可信工具服务耗时不消耗计算 deadline，实际执行总用时仍进入 E；保留独立360秒单次运行墙钟限、工具数量、CPU/内存等限制 |
| 健康观察器不必要几何开销 | Frozen73 没有轨迹谓词时，健康子步只读动力学/接触数据；评分仍从完整最终空间状态读取 P1/P2/C1。实际观察器开关非干扰检查通过 |
| 零采样健康被读成无异常 | 零帧明确标 NO_SAMPLES；不再用 NO_ALERTS_IN_SAMPLES 表示未采样的健康通过 |
| 错误归因过于粗糙 | 增加动作拒绝、物理控制/观测错误、Skill/协议错误、验证/状态错误与清理错误分类，并单独保留执行、计算和工具耗时 |
| 输入职责未定义 | 增加版本化 InputAdapter，支持显式 A/B，拒绝隐藏字段、路径及非有限值；保持 WAIT_CONFIRM_INPUT_CONTRACT，不默认选择模式、不调用理解模型 |
| 历史文档与现行规则冲突 | 原验收说明和报告移入包内 reports/archive；当前 README、配置、部署说明以本次加固为准 |
| baseline 使用不可达 IK 仍继续动作 | 私人 baseline 对大残差的指定姿态/路径直接记录策略失败，不用非法状态修改或不合理关节动作兜底 |

## 2. 发布目录与唯一入口

服务器发布目录：`/mnt/SSD_3TB/jusheng/track2_maniskill_staging_20261010/release`。

```text
release/
    evaluator.zip
    SHA256SUM
    VERSION
    DEPLOYMENT.md
    requirements_lock.txt
    Frozen73_Release_Hardening_Report.md
```

新版本标识：`frozen73-hardening-20261014-v1`。部署包摘要以同目录 SHA256SUM 为准；报告不嵌入自身所在 ZIP 的摘要，避免自引用。唯一逐 Case 程序为 ZIP 中 `release_entry.py`；73 Case 调度程序仅调用这个入口。

原始交付基线 ZIP 摘要为 `31255cd5ca5390ff21600e3c32f8b642e87ca2a4e21813ee1089a20efde9b7b7`，不再作为本次部署入口。原三个候选 ZIP 归档到 `archive/legacy_frozen73_20261014/`，新私人实验的中间源码/基线包和旧文档也归档，禁止按近似文件名或 mtime 选择版本。历史 Casebank、manifest 与实验结果留在原裁判私人路径，未移到公开目录。

当前目录存在的是**可审阅的加固制品，不是已上线服务**。NOT_READY 标记写入 VERSION/当前配置；没有正式 Worker、正式队列启动或平台回调客户端。

## 3. 环境冻结

| 项目 | 锁定值 | 结果 |
| --- | --- | --- |
| Python | 3.10.12 | PASS |
| ManiSkill | 3.0.1 | PASS |
| SAPIEN | 3.0.3 | PASS |
| PhysX | 105.1-physx-5.3.1.patch0，随 SAPIEN 内嵌 | PASS |
| Pillow | 12.3.0 | PASS；修正旧声明12.0.0 |
| NumPy | 2.2.6 | PASS |
| SciPy | 1.15.3 | PASS |
| PyTorch | 2.14.1+cpu | PASS |
| 完整项目依赖 | 86项元数据，pip/setuptools不作为应用安装锁约束 | PASS |
| 新环境离线安装 | 既有 wheelhouse → 新 env_locked，无网络安装 | PASS |
| 新环境与锁一致 | 所有应用包、Python、PhysX版本及物理二进制摘要 | PASS |

PhysX 所在 SAPIEN 二进制 SHA256：`bdb8967f4fb2a9cedd1bbfaed0993cd4847a0f664b83d429c937cd0aeca0347c`。

锁文件为 requirements_lock.txt、environment_lock.json。PhysX 不是独立 pip 包，不伪造 requirements 条目；其版本和库摘要另行检查。离线复建是在同一服务器新虚拟环境完成，不能外推为任意干净 OS 都已验收；Linux 系统 unshare/mount/libseccomp 仍为外置前提。原 env_ready、系统 Python 和生产环境没有被修改。

## 4. 评分与完整性保持

| 冻结对象 | 本轮结果 |
| --- | --- |
| Scorer | `10f89dab4354a8d79094d69d60f9f36f311b70dc02c71c590cdc4c7c249a7cd4`，逐字节不变 |
| P1/P2/C1实现、冻结profile、scoring_config | contract_extension.py、trusted_predicates.py、frozen73_profile.py、scoring_config.json 逐字节不变 |
| 公开目标投影 | public_task_view.py 逐字节不变；诊断RGB只附加批准的图像/相机信息，不改变目标 |
| GT进程隔离 | sandbox_bootstrap.py、sandbox_runtime.py、rpc.py 逐字节不变 |
| Piper模型与驱动参数 | piper_agent.py、model/piper_sim.urdf 逐字节不变 |
| 原私有73manifest | `b2af8ad3301a81a5d1fa14780af41f4b0526d2f0a3f80bfb0f1b6911d02e2404`，不变 |
| Case文件 | 测试后再核验73/73摘要一致 |
| 碰撞资产 | 测试后再核验8,470/8,470摘要一致 |

抓放仍为等权 P1+P2，任务可选最终非目标位移 C1，完成阈值0.6；三档容差、5 seeds、first/final、最多两次恢复、总体标准差及总分权重都不变。没有计轨迹、自动扣物理健康分或按 Skill 自报分数评分。冻结评分配置覆盖拒绝测试通过。

正确的任务失败仍可以按冻结公式得到辅助分，不能强行把正常任务失败归零。运行/GT/证据异常则不生成可发布分数；原始私有诊断分与可发布结果分开。时间边界修复改变被记录的执行用时，没有修改 `E=min(1,Tref/Tused)`、Tref=90或总分算术。

## 5. ResultGate

| 条件 | 门禁行为 | 结果 |
| --- | --- | --- |
| Recorder缺失、记录数不足、重复/配对错误 | BLOCKED + 原因 + score=null | PASS |
| Verifier未完成可信重放、receipt缺失或校验失败 | BLOCKED | PASS |
| Scorer SHA不一致 | 导入前/门禁双重拒绝；不能生成正式分数 | PASS |
| 超时、运行异常、清理失败 | BLOCKED；区分阶段/错误类别 | PASS |
| 初末状态缺失、对象集合不完整、非有限值 | BLOCKED | PASS |
| GT读取异常或没有明确成功状态 | BLOCKED | PASS |
| 分数、用时、恢复次数、状态或证据被改 | 摘要与可信重放绑定校验拒绝 | PASS |
| 输出文件写失败 | BLOCKED；stderr保留稳定原因 | PASS |
| 正常任务失败 | 验证完整时保留冻结评分，不冒充执行错误 | PASS |
| 有official_approval但没有可信门禁/存在运行失败 | 拒绝构造成功回调 | PASS |

最终门禁/输入/计时负面测试17项通过。门禁使用裁判私有 receipt，Skill没有该文件、裁判对象或分数写权限。`publishable`仅表示通用证据门禁条件通过；当前所有官方分数仍为null，正式发布还需要已确认输入协议和明确的生产授权。

旧反例 `official_approval=true + EXECUTION_FAILURE + score=30` 已被阻断。未发送任何回调请求。OfflineOutbox的本地原子领取、重复键、冲突、确认和崩溃后对账测试通过；真实平台回调/远程幂等仍未测试，需在正式接入时验证。

## 6. 输入接口占位与安全

InputAdapter模式A接受公开理解文本、标签/归一化图像bbox/置信度等；模式B接受公开原始图文、参考图/首帧。两种模式都需显式 producer/mode；不设置官方默认、不内置VLM、不开网络服务。隐藏GT、initial_pose、final_pose、完整裁判manifest、scorer/judge状态、成绩和宿主路径字段被拒绝。当前状态始终为 **WAIT_CONFIRM_INPUT_CONTRACT**。

为了物理失败诊断，另有显式 diagnostic_rgb 分支：CPU物理不变，仅GPU0渲染当前 actor-local碰撞几何的256×256 RGB，色彩按对象自身名称的固有颜色词生成；不使用原baked GLB，不传分割ID、物体真值pose、资产路径或完整manifest。相机参数由当前公开目标与固定诊断镜头参数构成。这个私人受控诊断不选择正式A/B，也不代表真实材质/全73视觉已验收；唯一正式占位入口默认关闭它。

| 访问/作弊项 | 结果 | 证据 |
| --- | --- | --- |
| 隐藏GT、初态/末态pose文件 | PASS：拒绝 | 实际 namespace/chroot canary 读写与模块隔离 |
| 完整裁判manifest、其他Case、Scorer内部状态 | PASS：拒绝 | 扩充专用 canary；最小运行时与父进程隔离保持原策略 |
| 成绩/证据文件写入 | PASS：拒绝 | 文件不可达、canary摘要不变、结果门禁篡改拒绝 |
| set_pose、translate、rotate、直接状态修改 | PASS：拒绝 | 原关节/夹爪白名单，新增越权回归覆盖；物理状态前后不变 |
| get_layout/get_final_state/内部健康getter | PASS：拒绝 | 不在 Skill RPC白名单 |
| Skill自报分数/末态 | PASS：忽略 | 评分只来自父进程物理状态 |
| 正常Skill自身manifest | 允许 | 与不可见的完整裁判manifest区分 |
| 内核逃逸穷举、生产管理器实配 | NOT TESTED | 不把指定31项安全回归通过称为全漏洞不存在 |

新扩充安全回归31项通过。最初直接筛读所有私有Case JSON的命令被自动审批拒绝，未执行；选样改用先前Recorder输出中的公开P2对象标识，随后只通过原 selected-case 裁判加载器执行正常测试。没有人工提取/下发隐藏初态，也没有绕过该拒绝去批量读取Case内容。

## 7. 真实物理可行性抽查

### 7.1 合成代表物体：阳性对照

这些是固定公开校准先验的**合成代表Case**，不是原TTG资产。它们验证合法隔离Skill→Piper→ManiSkill→CPU PhysX→TrustedRecorder→可信Verifier→未改Scorer/总分聚合→ResultGate完整链路，并帮助排除“Piper/控制接口普遍不能抓取”的假设。

| 代表物体 | 真实回合 | 抓放成功 | 最大抬升 | P1/P2及门禁 | 动作健康帧/警报 |
| --- | --- | --- | --- | --- | --- |
| 50mm盒体 | 10 | 10/10 | 0.08923m | 全通过；冻结规则分100 | 113,760 / 无 |
| 开口分段碰撞碗 | 10 | 10/10 | 0.09203m | 全通过；冻结规则分100 | 113,760 / 无 |
| 带柄复杂杯体 | 10 | 10/10 | 0.08427m | 全通过；冻结规则分100 | 113,760 / 无 |

每类均为五次first/final，真实动作，不是复制状态或旧730个noop回合。校准物体质量/惯量为测试配置，不能当作原TTG物理参数验收；原TTG参数未改变。上述100分仅是私有合成校准结果，没有进入正式成绩。

### 7.2 原Frozen73 Case：杯、葡萄、剪刀

原冻结任务通过原加载器读取，每个抽查采用一个真实first/final配对。没有凑足/复制为10回合，所以门禁明确 `DIAGNOSTIC_ONLY_TWO_RUNS_EXPECTED_TEN` 并阻断发布。每回合实际约948个合法控制调用、11,376个健康子步，P1/P2仍按原任务评分。

| 原对象 | 最终有效诊断 | 抓取 | P1/P2 | 健康/控制证据 | 归因 |
| --- | --- | --- | --- | --- | --- |
| blue_mug_on_table_000 | first/final正常完成；亦做私人RGB复查 | 无双指接触/有效抬升；峰值约0.00043m | 均false | 无非有限帧、无健康警报；最后baseline加权IK残差0.46953 | **A：所选定位/姿态路径不能成立。** 该baseline失败不是全部合理策略失败的证明 |
| bunch_of_red_grapes_on_table_000 | 时间预算修复后first/final正常完成 | 未抓取 | 均false；C1违规1 | 墙钟约176/178s，其中可信工具约160/162s；非有限帧0、无健康警报 | 先有**B：工具服务被计入Skill预算**，已修复并复测；剩余为A策略未取得抓持 |
| green_scissors_on_coffee_table_000 | 时间预算修复后first/final正常完成 | 未抓取 | 均false；C1违规1 | 墙钟约117s，可信工具约111s；非有限帧0；接触冲量警报 | B预算问题已修复；A策略未成功。冲量警报是待分析的物理接触现象，不能直接叫引擎发散或已证明环境缺陷 |

没有看到set_pose、非法动作或状态自报绕过。器件未被强行移到手内，物体pose只来自正常初始化与PhysX响应。原任务P1/P2正确给出了当前策略失败结果。

在已有73个公开目标标识索引中，没有名字含bowl的冻结评分目标；本轮使用合成开口碗作为碗类阳性对照，**没有改写任何原Case目标或谓词来制造碗任务**。原场景中的碗资产抓持尚未得到独立阳性证据，记NOT TESTED。

**可行性验收结论：合成控制链路PASS；原TTG抽查执行链路完成，但成功抓持/搬运证据不足。不能把三类合成成功外推为原73Case环境不会导致正常Skill失败，也不能把这个弱baseline的失败反过来当作环境必坏。** 原TTG阳性抓持证据仍是本版本未达到生产评审门禁的原因。没有为通过测试改动P1/P2/C1、目标、碰撞资产、物体物理参数或原Case冻结文件。

## 8. 未修复事项与未覆盖范围

1. **原TTG抓持可行性阳性证据未取得。** 已完成多次合法控制与RGB诊断；最后指定路径存在明显IK残差。需要更合适的可达姿态/视觉规划给出原资产实际抓持证据，或取得与策略无关的物理/接口错误定位。当前无法把它诚实标PASS。
2. 正式OS/Skill图文职责、观测更新、坐标/身份/依赖契约仍未确认；只实现可扩展层，没有擅定模式。
3. 原场景碗资产、全73视觉/主动操作、长时间运行、多提交并发和全域可达性未完成验收。
4. 真实平台队列、远程回调幂等/确认、活动服务映射及正式切换/回滚未操作或测试。已完成本地门禁/离线outbox，但不冒充生产接入。

已识别的发布、依赖、结果门禁、计时和诊断开销问题均已修复并验证。以上物理证据不足与未来正式接入验证不转写成新的评分规则待批准，也不以修改规则、公开GT或改动生产来解决。

## 9. PASS / FAIL / NOT TESTED总表

| 项目 | 结果 |
| --- | --- |
| 唯一release入口、旧ZIP归档、摘要及VERSION | PASS |
| 完整依赖锁、Python/PhysX二进制、新环境离线复建 | PASS |
| 原公式、P1/P2/C1、隔离关键模块、Piper模型保持 | PASS |
| 73Case和8,470资产测试后摘要一致 | PASS |
| ResultGate完整性、异常/缺失/篡改拒绝与原因 | PASS（17项） |
| GT/pose/其他Case/manifest/成绩/内部RPC越权拒绝 | PASS（31项） |
| 初始化/工具服务与Skill预算分开，保留E公式 | PASS |
| 输入A/B占位、隐藏字段拒绝、无正式模式选择 | PASS；WAIT_CONFIRM_INPUT_CONTRACT |
| 两轮完整CPU回归 | PASS（各14组） |
| 最后有新改动的核心回归 | PASS（9组）；最后存储门禁另行实测通过 |
| 三类合成物体合法Skill主动抓放 | PASS（30/30） |
| 原TTG代表抽查成功抓持的验收门禁 | **FAIL：当前策略未成功，物理可行性尚未被阳性证明** |
| 原Case执行/验证及真实失败记录 | PASS；短配对诊断正式发布门禁按设计BLOCKED |
| 原TTG碗资产抓持、全73视觉/可达性/主动物理覆盖 | NOT TESTED |
| 跨OS空白机器复建、生产管理器隔离 | NOT TESTED |
| 真实平台队列、回调、远程幂等、切换/回滚 | NOT TESTED |
| 正式Worker、服务、成绩被修改 | 未操作；无正式回调或公开选手目录写入 |

## 10. 剩余必须负责人确认事项

**只保留以下两项：**

1. **输入接口协议**：A/B职责、图文与后续观测、坐标/对象身份、依赖和资源/工具输入契约。状态 WAIT_CONFIRM_INPUT_CONTRACT。
2. **正式生产切换权限**：审阅最终制品与风险后另行明确授权。本任务不启动或改动生产Worker。

原评分规则不再待确认。原TTG抓持阳性证据缺口是技术验收事项，不新增负责人“批准物理规则/评分公式”项。

**最终状态：NOT_READY。** 加固制品和证据已形成，发布及安全工程门禁通过；由于原TTG真实抓持可行性验收尚未证明，不申请或执行正式切换，也不标记READY_FOR_PRODUCTION_REVIEW。

## 11. 证据位置

- 私人服务器工作目录：`experiments/Frozen73_Release_Hardening_20261014`。
- `environment_rebuild_sanitized.json`；`source/requirements_lock.txt`、`environment_lock.json`。
- `tests/full_cpu_001/execution.json`、`tests/final_cpu_002/execution.json`：14组全回归。
- `tests/last_validation_004/sanitized_execution.json`：9组最后核心回归。
- `tests/storage_gate_final_005/result.json`：17项最终门禁；`tests/security_final_003/result.json`：31项安全。
- `physical_controls_summary.json`：30个阳性对照；完整原始轨迹仅在私人physical_controls目录。
- `frozen_pair_probes_002`、`frozen_pair_probes_final_003`、`frozen_rgb_probe_004/005`：原Case短配对及RGB诊断；只交付sanitized_summary。
- `hardening_evidence_sanitized.json`：本地及包内可审阅汇总。最初已停止的慢采样私人实验保留，不作为通过证据。
- `release/SHA256SUM`、`VERSION`、`DEPLOYMENT.md`；`archive/legacy_frozen73_20261014/archive_manifest.json`：最终制品与旧版身份。
