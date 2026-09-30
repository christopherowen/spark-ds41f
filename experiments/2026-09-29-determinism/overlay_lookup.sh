#!/bin/bash
# usage: overlay_lookup.sh   (on dgx1, deployment checkout at this experiment's commit)
# Rebuilds, on every node, the gemv-lookup overlay (vllm-0027-gemv-smallest-capacity.patch
# applied to r5o's vLLM tree: b12x_layers.py and compressor.py) and the attn-trace debug
# overlay (debug-attn-trace.diff: the batch-trace files plus attention checksums and the
# first-call router gate capture), and prints their checksums.
set -euo pipefail
cd ~/projects/spark3-vllm-ds41f
E=$PWD/experiments/2026-09-29-determinism
TREE=c108cd6d1fe8e2d3b91c065feefe818742159020
SRC=$(for d in .work/build/vllm-04c30fa98e79-*/src/vllm; do
  [ "$(git -C "$d" rev-parse HEAD^{tree})" = $TREE ] && echo "$d" && break; done)
[ -n "$SRC" ] || { echo "no prepared r5o vLLM tree; run bin/spark3 build prepare"; exit 1; }
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
git clone -q "$SRC" "$T/vllm"
git -C "$T/vllm" apply "$E/vllm-0027-gemv-smallest-capacity.patch"
(cd "$T/vllm" && patch -s -p1 < "$E/debug-attn-trace.diff")
L=$T/gemv-lookup/vllm/models/deepseek_v4_1 A=$T/attn-trace
mkdir -p "$L" "$A"
cp "$T"/vllm/vllm/models/deepseek_v4_1/{b12x_layers.py,compressor.py} "$L/"
cp "$T/vllm/vllm/model_executor/layers/fused_moe/runner/"{checksum_debug.py,moe_runner.py} "$A/"
cp "$T/vllm/vllm/models/deepseek_v4/nvidia/model.py" "$T/vllm/vllm/v1/worker/gpu/model_runner.py" \
  "$T/vllm/vllm/models/deepseek_v4_1/attention.py" "$A/"
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" 'mkdir -p spark3-overlay/gemv-lookup/vllm/models/deepseek_v4_1 spark3-overlay/attn-trace'
  scp -q "$L"/*.py "$n:spark3-overlay/gemv-lookup/vllm/models/deepseek_v4_1/"
  scp -q "$A"/*.py "$n:spark3-overlay/attn-trace/"
  ssh -n "$n" 'cd spark3-overlay && sha256sum gemv-lookup/vllm/models/deepseek_v4_1/*.py attn-trace/*.py' | sed "s/^/$n /"
done
