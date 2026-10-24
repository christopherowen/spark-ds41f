#!/bin/bash
# The CPU test runs of this experiment, on a Mac (no container, no cluster, no Triton).
#
# usage: run-suites.sh VLLM_TREE OUTPUT TEST...
#
# VENV and SHIM as in experiments/2026-10-24-prompt-logprobs-sink/runs/run-suites.sh.
# suites.txt ran, on 5acb05bc7 (candidate 4 + 0266) and bb32f16cc (+ 0267):
#   tests/v1/core/test_sink_draft_zero.py tests/v1/engine/test_late_admission.py
#   tests/v1/engine/test_late_admission_sink.py tests/v1/core/test_scheduler.py
#   tests/v1/core/test_async_scheduler.py tests/v1/engine/test_output_processor.py
# and on bb32f16cc also tests/v1/worker/test_gpu_rejection_sampler_chunking.py
# tests/v1/worker/test_gpu_verified_logprob_token_ids.py (neither imports on 5acb05bc7
# without Triton). tests-before-fix.txt ran test_gpu_verified_logprob_token_ids.py on
# 5acb05bc7 with only 0267's boundary_checkpoint.py; tests-fix.txt ran the new tests on
# bb32f16cc, and the new scheduler test on 5acb05bc7's tree.
set -u
tree=$1; out=$2; shift 2
cd "$tree" || exit 2
VLLM_TARGET_DEVICE=cpu PYTHONPATH="$tree:$SHIM" timeout 3000 "$VENV/bin/python" -m pytest -q \
  -p no:cacheprovider -p macos_shim -rfE "$@" > "$out" 2>&1
echo "exit $?" >> "$out"
