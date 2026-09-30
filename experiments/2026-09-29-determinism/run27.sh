#!/bin/bash
# usage: run27.sh   (on dgx1, deployment checkout at this experiment's commit, config = r5o,
#                    after overlay_masked.sh; run26.sh's trace and replay done)
# Stops the cluster; on dgx3, moe_serving_replay.py (the deterministic routed MoE
# through serving's one-plan path, target rows alone and in batches of 1-512 rows).
# Then run26.sh's performance section (restores r5o at the end).
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism/lookup
IMAGE=vllm-ds41f-kkref:04c30fa98e79-r5o
log() { echo "$(date -u +%FT%TZ) $*"; }
for c in config/cluster.json $E/cluster-*.json experiments/2026-09-30-r5o/cluster-*.json \
  experiments/2026-09-30-r5n/cluster-*.json; do
  bin/spark3 --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
done
ssh -n dgx3 'mkdir -p /tmp/lookup/cache'
scp -q $E/moe_serving_replay.py dgx3:/tmp/lookup/
log "moe serving replay on dgx3"
ssh dgx3 "IMAGE=$IMAGE bash -s" > "$out/moe-serving-replay.txt" 2>&1 <<'REMOTE'
B=/opt/spark3/candidate/b12x
O=$HOME/spark3-overlay/det-masked
m="-v /tmp/lookup/moe_serving_replay.py:/r/moe_serving_replay.py:ro"
for f in b12x/moe/fused_moe/_impl.py b12x/moe/fused_moe/_preparation.py b12x/moe/fused_moe/_tuning.py \
  b12x/moe/_shared/kernels/dynamic.py b12x/moe/_shared/kernels/silu.py b12x/moe/_shared/kernels/w4a16/kernel.py; do
  m="$m -v $O/$f:$B/$f:ro"
done
docker run --rm --gpus all --ipc=host $m -w $B -e CUTE_DSL_ARCH=sm_121a -e B12X_DENSE_SPLITK_TURBO=0 \
  -e B12X_CUTE_COMPILE_CACHE_DIR=/c/cute -e CUTE_DSL_CACHE_DIR=/c/cutedsl -e B12X_COMPILE_CACHE_DIR=/c/b12x \
  -v /tmp/lookup/cache:/c --entrypoint python3 $IMAGE /r/moe_serving_replay.py
REMOTE
log "moe replay done"
grep -E "^\[|Error|Traceback" "$out/moe-serving-replay.txt" | head -20
$E/run26.sh --skip-trace --skip-replay
