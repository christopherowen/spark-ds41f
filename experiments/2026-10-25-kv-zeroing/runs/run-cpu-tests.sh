#!/bin/bash
# The CPU test runs of this experiment, on a Mac (no container, no cluster).
#
# usage: run-cpu-tests.sh VLLM_TREE OUTPUT
#
# VENV is a Python 3.12 environment with torch 2.14.1 (CPU), pytest and the tree's
# requirements/common.txt; SHIM holds macos_shim.py (torch's host-cache release segfaults
# without an accelerator, in vLLM's test teardown):
#
#   import torch
#   torch.accelerator.empty_host_cache = lambda *args, **kwargs: None
#
# Offline, no TileLang: test files that import a TileLang module fail to collect on both
# trees alike, and pytest continues past them. VLLM_TARGET_DEVICE=cpu selects vLLM's CPU
# platform; the tree is imported from PYTHONPATH.
set -uo pipefail
cd "$1" && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_TARGET_DEVICE=cpu PYTHONPATH="$1:$SHIM" \
  timeout 3000 "$VENV/bin/python" -m pytest -q -p no:cacheprovider -rfE --continue-on-collection-errors \
  tests/v1/core tests/v1/worker tests/test_envs.py > "$2" 2>&1
