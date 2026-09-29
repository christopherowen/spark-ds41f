#!/bin/bash
# usage: run5.sh   (on dgx1, deployment checkout at this experiment's commit, after overlay.sh)
# Why does detslice not repeat? With the cluster stopped, on dgx3 in a fresh
# r5m container with the det-slices overlay: moe_repeat_check.py with slice
# partials on and off (NaN-poisoned route rows, eager and graph replay), then
# compute-sanitizer initcheck (allocator caching off), racecheck and synccheck
# on one decode-sized launch. Restores r5m.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism
log() { echo "$(date -u +%FT%TZ) $*"; }
for c in config/cluster.json $E/cluster-*.json; do
  bin/spark3 --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
done
scp -q $E/moe_repeat_check.py dgx3:/tmp/moe_repeat_check.py
log "checks on dgx3"
ssh dgx3 'bash -s' > "$out/sanitizer.txt" 2>&1 <<'REMOTE'
O=$HOME/spark3-overlay/det-slices
B=/opt/spark3/candidate/b12x
mounts="-v /tmp/moe_repeat_check.py:/tmp/moe_repeat_check.py:ro -v /usr/local/cuda-13.0/compute-sanitizer:/opt/compute-sanitizer:ro"
for f in b12x/moe/fused_moe/_impl.py b12x/moe/fused_moe/_preparation.py \
  b12x/moe/fused_moe/_tuning.py b12x/moe/_shared/kernels/dynamic.py \
  b12x/moe/_shared/kernels/silu.py; do
  mounts="$mounts -v $O/$f:$B/$f:ro"
done
run() {
  docker run --rm --gpus all --ipc=host $mounts -w $B -e CUTE_DSL_ARCH=sm_121a \
    -e B12X_CUTE_COMPILE_CACHE_DIR=/tmp/cute -e CUTE_DSL_CACHE_DIR=/tmp/cutedsl \
    -e B12X_COMPILE_CACHE_DIR=/tmp/b12xc "$@"
}
echo "== repeat check, slice partials on"
run --entrypoint python3 vllm-ds41f-kkref:04c30fa98e79-r5m /tmp/moe_repeat_check.py
echo "== repeat check, slice partials off"
run --entrypoint python3 vllm-ds41f-kkref:04c30fa98e79-r5m /tmp/moe_repeat_check.py --no-slices
for tool in initcheck racecheck synccheck; do
  for m in 1 6; do
    echo "== compute-sanitizer $tool, m=$m, slice partials on"
    run -e PYTORCH_NO_CUDA_MEMORY_CACHING=1 --entrypoint /opt/compute-sanitizer/compute-sanitizer \
      vllm-ds41f-kkref:04c30fa98e79-r5m --tool $tool --print-limit 20 \
      python3 /tmp/moe_repeat_check.py --only $m --reps 1 --no-graph 2>&1 | tail -40
  done
done
REMOTE
log "checks done"
grep -E "^==|^m=|OK|FAIL|ERROR SUMMARY|Invalid|Uninitialized|hazard|Race" "$out/sanitizer.txt" | head -60
bin/spark3 cluster start --replace --apply | grep -v 'docker run'
log "done"
