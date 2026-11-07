#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# CPU-only verification of 0731-sp-engram-prefill.patch inside spark-ds41f:1030
# with GPUs hidden (CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES=void).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VLLM_DIR="${1:-/home/chris/spark-lab/sp-engram-prefill/vllm}"
IMAGE="${SPARK_CPU_TEST_IMAGE:-spark-ds41f:1030}"

docker run --rm --entrypoint /usr/bin/python3 \
  -e CUDA_VISIBLE_DEVICES="" \
  -e NVIDIA_VISIBLE_DEVICES=void \
  -e PYTHONPATH="/repo/scripts/release_tests:/vllm" \
  -v "$REPO:/repo:ro" \
  -v "$VLLM_DIR:/vllm:ro" \
  -w /vllm \
  "$IMAGE" \
  /repo/experiments/2026-11-06-sp-engram-prefill/cpu.py
