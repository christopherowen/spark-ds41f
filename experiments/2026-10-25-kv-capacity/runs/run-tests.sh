#!/bin/bash
# The CPU test runs of this experiment, on a Mac (no container, no cluster).
#
# usage: run-tests.sh VLLM_TREE OUTPUT
#
# VENV is a Python 3.12 environment with torch 2.14.1 (CPU), pytest and the tree's
# requirements/common.txt; SHIM holds macos_shim.py (torch's host-cache release segfaults
# without an accelerator, in vLLM's test teardown):
#
#   import torch
#   torch.accelerator.empty_host_cache = lambda *args, **kwargs: None
#
# Offline: only facebook/opt-125m's config is cached, so tests that build a ModelConfig for
# another hub id fail on both trees alike. VLLM_TARGET_DEVICE=cpu selects vLLM's CPU
# platform; the tree is imported from PYTHONPATH.
set -euo pipefail
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_TARGET_DEVICE=cpu PYTHONPATH="$1:$SHIM" \
  timeout 3000 "$VENV/bin/python" -m pytest -q -p no:cacheprovider -rf "$1/tests/v1/core" > "$2" 2>&1
