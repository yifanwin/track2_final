#!/usr/bin/env bash
set -eu
cd -- "$(dirname -- "$0")"
candidate_python="${CANDIDATE_PYTHON:-python3}"
candidate_output="${1:-replay_v2}"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
"$candidate_python" -B check_install.py
"$candidate_python" -B tests.py --isolated --output "$candidate_output/v1_regression"
"$candidate_python" -B v2_tests.py --isolated --output "$candidate_output/functional"
"$candidate_python" -B security_test.py --output "$candidate_output/security"
"$candidate_python" -B package_security_test.py --output "$candidate_output/package_security"
"$candidate_python" -B v2_physics_test.py --output "$candidate_output/physics"
"$candidate_python" -B v2_physics_test.py --no-events --non-target-drop --output "$candidate_output/non_target_drop"
"$candidate_python" -B v2_physics_test.py --instruction noop --output "$candidate_output/missing_evidence"
# CPU visualization is attempted separately: a real renderer failure must remain
# FAIL in its own report without preventing scoring/security tests or inventing RGB.
"$candidate_python" -B visual_probe_v2.py --output "$candidate_output/visual"
"$candidate_python" -B verify_manifest.py
