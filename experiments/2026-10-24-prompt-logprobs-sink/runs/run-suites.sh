#!/bin/bash
# The CPU test runs of this experiment, on a Mac (no container, no cluster).
#
# usage: run-suites.sh VLLM_TREE OUTPUT
#
# VENV is a Python 3.12 environment with torch 2.14.1 (CPU), torchvision 0.29.1, pytest,
# tblib and the tree's requirements/common.txt; SHIM is a directory holding macos_shim.py:
#
#   import torch
#   torch.accelerator.empty_host_cache = lambda *args, **kwargs: None
#
# (torch's host-cache release segfaults without an accelerator, in vLLM's test teardown).
# VLLM_TARGET_DEVICE=cpu selects vLLM's CPU platform; the tree is imported from PYTHONPATH.
# tests-before-fix.txt ran test_sink_draft_zero.py alone, with 0266's scheduler.py and
# request.py reverted to candidate 4's; tests-fix.txt ran the first three files.
set -u
cd "$1" || exit 2
VLLM_TARGET_DEVICE=cpu PYTHONPATH="$1:$SHIM" timeout 3000 "$VENV/bin/python" -m pytest -q \
  -p no:cacheprovider -p macos_shim -rfE \
  tests/v1/core/test_sink_draft_zero.py \
  tests/v1/engine/test_late_admission.py tests/v1/engine/test_late_admission_sink.py \
  tests/v1/core/test_scheduler.py tests/v1/core/test_async_scheduler.py \
  tests/v1/engine/test_output_processor.py > "$2" 2>&1
echo "exit $?" >> "$2"
