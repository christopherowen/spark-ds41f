#!/bin/bash
# usage: overlay.sh   (on dgx1)
# Writes patch 0011's runtime files from the checked vLLM branch (the series
# applied to the base) to ~/spark3-overlay/r5e on every node.
set -eu
V=~/projects/spark3-vllm-ds41f/.work/upstreams/vllm
O=~/spark3-overlay/r5e
[ "$(git -C $V rev-parse spark3/r5e-check^{tree})" = 7b839dc1954f91cf1406e1775fad3c16d207ab53 ]
rm -rf $O
for f in vllm/v1/worker/gpu/input_batch.py vllm/v1/worker/gpu/model_runner.py \
    vllm/v1/worker/gpu/spec_decode/adaptive_verification.py; do
  mkdir -p "$O/$(dirname $f)"
  git -C $V show "spark3/r5e-check:$f" > "$O/$f"
done
for host in dgx2 dgx3; do
  ssh "$host" "rm -rf $O && mkdir -p ~/spark3-overlay"
  tar cf - -C ~/spark3-overlay r5e | ssh "$host" "tar xf - -C ~/spark3-overlay"
done
