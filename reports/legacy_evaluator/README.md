# Frozen73 Release Hardening

Deployment entry: release/evaluator.zip verified by release/SHA256SUM. Extract into a new judge-private directory and run release_entry.py. No production Worker or callback is included.

Scoring: immutable P1+P2, optional final non-target displacement C1, unchanged V3 summarize. Frozen scorer SHA256: 10f89dab4354a8d79094d69d60f9f36f311b70dc02c71c590cdc4c7c249a7cd4. Frozen private 73 manifest SHA256: b2af8ad3301a81a5d1fa14780af41f4b0526d2f0a3f80bfb0f1b6911d02e2404.

The only pending owner decisions are WAIT_CONFIRM_INPUT_CONTRACT and production switch permission. InputAdapter supports explicit A (public OS perception) and B (public raw image/text); neither is selected or advertised as the official SDK.

Requirements are frozen in requirements_lock.txt and environment_lock.json, including PhysX build and SAPIEN binary hash. Python 3.10.12; ManiSkill 3.0.1; SAPIEN 3.0.3; PhysX 105.1-physx-5.3.1.patch0; Pillow 12.3.0; NumPy 2.2.6; torch 2.14.1+cpu. No automatic install or network call.

ResultGate requires trusted replay receipt, complete Recorder/state evidence, paired repeats, valid runtime status and scorer integrity. Verified task failure remains a legal outcome with frozen auxiliary points; execution/GT/evidence failures block publication and record reason, score=null. Raw internal diagnostic scores are not official results. Official callback construction additionally requires explicit approval and revalidates ResultGate; no callback is sent.

Elapsed scoring time measures active Skill execution and tool calls. Factory/reset/runtime preparation are separately recorded and do not reduce Skill's execution deadline. This changes time measurement boundary only, not E formula or Tref.

Original documents and experiment reports are historical under reports/archive. Current findings are in Frozen73_Release_Hardening_Report.md. Synthetic positive controls never certify Frozen73 physical feasibility. Zero health samples are NO_SAMPLES.

Run checks: python -B verify_manifest.py; python -B environment_lock.py; python -B hardening_tests.py --output /private/new_test; python -B frozen73_scoring_test.py --output /private/new_score_test; python -B extension_security_test.py --output /private/new_security_test.

Run one frozen Case: python -B release_entry.py --casebank /private/casebank_frozen_final73 --freeze-manifest /private/frozen_final73_manifest.json --case-id FROZEN_CASE_ID --skill-zip /private/skill.zip --output /private/new_run

All raw runs, state, GT and physical traces must stay judge-private 0700/0600. The ZIP contains code/public robot resources and sanitized reports only; external private Casebank, manifest, dependencies and Linux isolation are required. cache/tmp under extracted private source must be writable. Production service integration and switch are not performed by this package.

Trusted physical tool service is excluded from Skill computation deadline, but retained in scored elapsed time. Independent per-run hard wall cap is 360 seconds. Diagnostic RGB is explicit/private only; no official A/B selection. Current readiness: NOT_READY; original TTG grasp feasibility positive evidence remains unproven.
