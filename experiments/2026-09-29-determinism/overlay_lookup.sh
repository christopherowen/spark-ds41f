#!/bin/bash
# usage: overlay_lookup.sh   (on dgx1, deployment checkout at this experiment's commit)
# Rebuilds, on every node, the gemv-lookup overlay (vllm-0027-gemv-smallest-capacity.patch
# applied to r5o's vLLM tree: b12x_layers.py and compressor.py) and the attn-trace debug
# overlay (debug-attn-trace.diff: the batch-trace files plus attention checksums and the
# first-call router gate capture, plan-selection dumps), the moe-variant overlay
# (b12x-0006-moe-smallest-variant.patch on r5o's B12X _preparation.py) and det-variant
# (the det-masked overlay, which overlay_masked.sh builds, with the same fix), and prints
# their checksums.
set -euo pipefail
cd ~/projects/spark3-vllm-ds41f
E=$PWD/experiments/2026-09-29-determinism
TREE=c108cd6d1fe8e2d3b91c065feefe818742159020
SRC=$(for d in .work/build/vllm-04c30fa98e79-*/src/vllm; do
  [ "$(git -C "$d" rev-parse HEAD^{tree})" = $TREE ] && echo "$d" && break; done)
[ -n "$SRC" ] || { echo "no prepared r5o vLLM tree; run bin/spark build prepare"; exit 1; }
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
git clone -q "$SRC" "$T/vllm"
git -C "$T/vllm" apply "$E/vllm-0027-gemv-smallest-capacity.patch"
(cd "$T/vllm" && patch -s -p1 < "$E/debug-attn-trace.diff")
BTREE=1a8b9401584ada0372939df49e658c3dbeae7658
BSRC=$(for d in .work/build/vllm-04c30fa98e79-*/src/b12x; do
  [ "$(git -C "$d" rev-parse HEAD^{tree})" = $BTREE ] && echo "$d" && break; done)
[ -n "$BSRC" ] || { echo "no prepared r5o B12X tree; run bin/spark build prepare"; exit 1; }
git clone -q "$BSRC" "$T/b12x"
git -C "$T/b12x" apply "$E/b12x-0006-moe-smallest-variant.patch"
MV=$T/moe-variant/b12x/moe/fused_moe DV=$T/det-variant
mkdir -p "$MV" "$DV"
cp "$T/b12x/b12x/moe/fused_moe/_preparation.py" "$MV/"
cp -r ~/spark3-overlay/det-masked/b12x "$DV/"
(cd "$DV" && git apply --include=b12x/moe/fused_moe/_preparation.py "$E/b12x-0006-moe-smallest-variant.patch")
L=$T/gemv-lookup/vllm/models/deepseek_v4_1 A=$T/attn-trace
mkdir -p "$L" "$A"
cp "$T"/vllm/vllm/models/deepseek_v4_1/{b12x_layers.py,compressor.py} "$L/"
cp "$T/vllm/vllm/model_executor/layers/fused_moe/runner/"{checksum_debug.py,moe_runner.py} "$A/"
cp "$T/vllm/vllm/models/deepseek_v4/nvidia/model.py" "$T/vllm/vllm/v1/worker/gpu/model_runner.py" \
  "$T/vllm/vllm/models/deepseek_v4_1/attention.py" "$A/"
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" 'mkdir -p spark3-overlay/gemv-lookup/vllm/models/deepseek_v4_1 spark3-overlay/attn-trace \
    spark3-overlay/moe-variant/b12x/moe/fused_moe spark3-overlay/det-variant'
  scp -q "$L"/*.py "$n:spark3-overlay/gemv-lookup/vllm/models/deepseek_v4_1/"
  scp -q "$A"/*.py "$n:spark3-overlay/attn-trace/"
  scp -q "$MV/_preparation.py" "$n:spark3-overlay/moe-variant/b12x/moe/fused_moe/"
  scp -rq "$DV/b12x" "$n:spark3-overlay/det-variant/"
  ssh -n "$n" 'cd spark3-overlay && find gemv-lookup attn-trace moe-variant det-variant -name "*.py" | sort | xargs sha256sum' \
    | sed "s/^/$n /"
done
