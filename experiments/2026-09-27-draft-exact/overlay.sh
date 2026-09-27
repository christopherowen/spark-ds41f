#!/bin/bash
# usage: overlay.sh   (on dgx1)
# Writes the runtime files of vLLM patches 0012-0013 (branch spark3/r5f) to
# ~/spark3-overlay/r5f on every node.
set -eu
V=~/projects/spark3-vllm-ds41f/.work/upstreams/vllm
O=~/spark3-overlay/r5f
FILES="vllm/v1/worker/gpu/spec_decode/dspark/greedy.py vllm/v1/worker/gpu/spec_decode/dspark/speculator.py vllm/model_executor/models/qwen3_dspark.py vllm/models/deepseek_v4_1/nvidia/dspark.py tests/v1/spec_decode/test_dspark_vocab_parallel.py"
rm -rf $O
for f in $FILES; do
  mkdir -p "$O/$(dirname $f)"
  git -C $V show "spark3/r5f:$f" > "$O/$f"
done
for host in dgx2 dgx3; do
  ssh "$host" "rm -rf $O && mkdir -p ~/spark3-overlay"
  tar cf - -C ~/spark3-overlay r5f | ssh "$host" "tar xf - -C ~/spark3-overlay"
done
