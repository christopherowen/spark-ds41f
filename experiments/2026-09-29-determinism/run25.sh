#!/bin/bash
# usage: run25.sh [NODE]   (on dgx1, deployment checkout at this experiment's commit, after
#                           overlay_masked.sh)
# Stops the cluster; on NODE (default dgx3), gemv_batch_invariance.py in fresh r5o
# containers (router gate GEMV and shared-expert linears across capacities). Restores r5o.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism
NODE=${1:-dgx3}
log() { echo "$(date -u +%FT%TZ) $*"; }
for c in config/cluster.json $E/cluster-*.json experiments/2026-09-30-r5o/cluster-*.json; do
  bin/spark --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
done
scp -q $E/gemv_batch_invariance.py $NODE:/tmp/
log "gemv batch invariance on $NODE"
ssh $NODE 'bash -s' > "$out/gemv-batch-invariance.txt" 2>&1 <<'REMOTE'
B=/opt/spark3/candidate/b12x
O=$HOME/spark3-overlay/det-masked
m="-v /tmp/gemv_batch_invariance.py:/tmp/gemv_batch_invariance.py:ro"
for f in b12x/moe/fused_moe/_impl.py b12x/moe/fused_moe/_preparation.py b12x/moe/fused_moe/_tuning.py \
  b12x/moe/_shared/kernels/dynamic.py b12x/moe/_shared/kernels/silu.py b12x/moe/_shared/kernels/w4a16/kernel.py; do
  m="$m -v $O/$f:$B/$f:ro"
done
run() {
  docker run --rm --gpus all --ipc=host $m -w $B -e CUTE_DSL_ARCH=sm_121a -e B12X_DENSE_SPLITK_TURBO=0 \
    -e B12X_CUTE_COMPILE_CACHE_DIR=/tmp/cute -e CUTE_DSL_CACHE_DIR=/tmp/cutedsl -e B12X_COMPILE_CACHE_DIR=/tmp/b12xc \
    --entrypoint python3 vllm-ds41f-kkref:04c30fa98e79-r5o /tmp/gemv_batch_invariance.py "$@"
}
echo "== capacities"
run
REMOTE
log "done invariance"
grep -E "^==|mode|batch|Error|Traceback|assert" "$out/gemv-batch-invariance.txt" | head -40
bin/spark cluster start --replace --apply | grep -v 'docker run'
log "done"
