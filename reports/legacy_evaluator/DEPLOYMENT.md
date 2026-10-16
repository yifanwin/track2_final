# Frozen73 唯一发布入口

唯一服务器部署候选：/mnt/SSD_3TB/jusheng/track2_maniskill_staging_20261010/release/evaluator.zip。
SHA256SUM 与 VERSION 必须同目录核验；旧近似名称 ZIP 已移入该 staging 的 archive，禁止按文件名或修改时间选择版本。

本版本只提交生产评审准备，不运行正式生产 Worker、不领取正式队列、不发平台回调。当前可部署性判断以同包 Frozen73_Release_Hardening_Report.md 为准；不得把发布目录存在当作生产就绪。

1. 在用户授权的裁判私有服务器目录，以部署账户建立新版本目录，权限0700。执行 sha256sum -c SHA256SUM；核对 VERSION 与报告。解压 evaluator.zip 到新目录，禁止覆盖旧版。
2. 使用本轮新建的私人 env_locked，或 Python3.10.12 在既有离线 wheelhouse 中依 requirements_lock.txt 重建独立环境。执行 python -B environment_lock.py 和 python -B verify_manifest.py。任何失败终止，不自动升级系统或改评分。
3. Casebank 与 frozen_final73_manifest.json 外置，仅裁判可见。Casebank0700、manifest0600，固定 manifest SHA256 b2af8ad3301a81a5d1fa14780af41f4b0526d2f0a3f80bfb0f1b6911d02e2404。不可将其复制进 Skill 包或公开选手目录。
4. 唯一逐Case运行入口：python -B release_entry.py --casebank /裁判私有目录/casebank_frozen_final73 --freeze-manifest /裁判私有目录/frozen_final73_manifest.json --case-id 冻结CaseID --skill-zip /裁判私有目录/skill.zip --output /裁判私有目录/新的唯一输出。完整73调度可调用 run_frozen73_suite.py，它仅调用 release_entry.py，没有平台客户端。
5. release_entry 自动核验源码清单和依赖锁，再调用可信裁判。ResultGate读取完整Recorder、状态、独立重放receipt及评分输出；异常/缺失/哈希失败写BLOCKED和原因，score=null。任务本身失败保留冻结分数，不冒充执行异常。通用证据门禁通过不代表正式回调获准；official_score仍为空。
6. INPUT CONTRACT保持 WAIT_CONFIRM_INPUT_CONTRACT。InputAdapter支持显式A（OS公开理解结果）和B（原始公开图文），不设默认、不调用模型。相机诊断分支仅以diagnostic_rgb显式启用，未作为正式输入协议。必须确认输入协议后接线、验证，再批准生产切换。
7. Skill仅有原最小运行时、关节/夹爪白名单和批准观测；隐藏初态、GT、完整裁判manifest、其他Case、score文件不能挂载。原baked GLB不得发布。cache/tmp只在裁判私有解压目录生成。
8. 正式切换必须另行取得明确权限。管理员核实服务映射和队列归属、停止新领取、排空在途任务，记录submitId/版本/回调确认，然后单消费者切换。不得运行旧Worker dry-run，它可能领取真实队列。
9. 回滚使用archive已登记摘要的旧版本与对应配置；先停止新领取、隔离未确认回调，按平台权威记录核对，不盲目重发、不覆盖已确认成绩。活动服务名和真实回调契约未经本轮生产操作核实，不编造systemctl命令。

源码ZIP不含私有Casebank、GT、真实轨迹、诊断图像、依赖wheel或完整生产Worker。完全空白服务器仍需要上述外置资源和Linux unshare/mount/libseccomp。所有测试输出只在私人experiments目录。
