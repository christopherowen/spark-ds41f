#!/bin/bash
# usage: run11.sh   (on dgx1, deployment checkout at this experiment's commit, after overlay.sh)
# With the cluster stopped: overlap_repro.py on dgx3 in a fresh r5m container
# with the det-slices overlay (deterministic slice partials; slices off; atomic
# r5m MoE). Restores r5m.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism
log() { echo "$(date -u +%FT%TZ) $*"; }
for c in config/cluster.json $E/cluster-*.json; do
  bin/spark --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
done
scp -q $E/overlap_repro.py dgx3:/tmp/overlap_repro.py
log "overlap repro on dgx3"
ssh dgx3 'bash -s' > "$out/overlap-repro.txt" 2>&1 <<'REMOTE'
O=$HOME/spark3-overlay/det-slices
B=/opt/spark3/candidate/b12x
mounts="-v /tmp/overlap_repro.py:/tmp/overlap_repro.py:ro"
for f in b12x/moe/fused_moe/_impl.py b12x/moe/fused_moe/_preparation.py \
  b12x/moe/fused_moe/_tuning.py b12x/moe/_shared/kernels/dynamic.py \
  b12x/moe/_shared/kernels/silu.py; do
  mounts="$mounts -v $O/$f:$B/$f:ro"
done
run() {
  docker run --rm --gpus all --ipc=host $mounts -w $B -e CUTE_DSL_ARCH=sm_121a \
    -e B12X_DENSE_SPLITK_TURBO=0 -e B12X_CUTE_COMPILE_CACHE_DIR=/tmp/cute \
    -e CUTE_DSL_CACHE_DIR=/tmp/cutedsl -e B12X_COMPILE_CACHE_DIR=/tmp/b12xc \
    --entrypoint python3 vllm-ds41f-kkref:04c30fa98e79-r5m /tmp/overlap_repro.py "$@"
}
echo "== deterministic, slice partials"
run
echo "== deterministic, slices off"
run --no-slices
echo "== atomic MoE (r5m)"
run --atomic
REMOTE
log "overlap repro done"
grep -E "^==|equal|OK|FAIL|Error|Traceback" "$out/overlap-repro.txt" | head -30
bin/spark cluster start --replace --apply | grep -v 'docker run'
log "done"
