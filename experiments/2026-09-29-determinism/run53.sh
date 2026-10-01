#!/bin/bash
# usage: run53.sh   (on dgx1, deployment checkout at this experiment's commit, config = r5o)
# Cluster stopped: transition_map.py vocab_served (the target LM head as served: BF16 rank-0 shard
# through vLLM's b12x vocabulary projection, Triton for one row and F.linear above; one row padded
# to two; F.linear alone) on dgx1.
# Restores r5o.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism/map6
IMAGE=vllm-ds41f-kkref:04c30fa98e79-r5o
log() { echo "$(date -u +%FT%TZ) $*"; }
stop_all() {
  for c in config/cluster.json $E/cluster-*.json experiments/2026-09-30-r5o/cluster-*.json \
    experiments/2026-09-30-r5n/cluster-*.json; do
    bin/spark3 --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
  done
}
mkdir -p "$out"
# Production restarts from this checkout: refuse to stop unless every node runs the same published commit.
git fetch -q origin
git merge-base --is-ancestor HEAD origin/main || { log "deployment commit not on origin/main; not stopping"; exit 1; }
for n in dgx2 dgx3; do
  [ "$(ssh -n $n git -C projects/spark3-vllm-ds41f rev-parse HEAD)" = "$(git rev-parse HEAD)" ] \
    || { log "$n checkout differs from dgx1; not stopping"; exit 1; }
done
curl -s http://10.0.1.71:8000/metrics | grep -E '^vllm:num_requests_running' || true
stop_all
B=/opt/spark3/candidate/b12x
DET=""
for f in b12x/moe/fused_moe/_impl.py b12x/moe/fused_moe/_preparation.py b12x/moe/fused_moe/_tuning.py \
  b12x/moe/_shared/kernels/dynamic.py b12x/moe/_shared/kernels/silu.py b12x/moe/_shared/kernels/w4a16/kernel.py; do
  DET="$DET -v /home/swank/spark3-overlay/det-variant2/$f:$B/$f:ro"
done
log "served vocabulary projection replays"
mkdir -p /tmp/lookup/cache
cp $E/transition_map.py /tmp/lookup/
S=$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash
docker run --rm --gpus all --ipc=host $DET \
  -v $S/snapshots/dba1be0a40aa45a94ad051997016db3960a90277:/models:ro -v $S/blobs:/blobs:ro \
  -v /tmp/lookup:/r:ro -v /tmp/lookup/cache:/c -w /opt/spark3/candidate/b12x \
  -e CUTE_DSL_ARCH=sm_121a -e B12X_DENSE_SPLITK_TURBO=0 -e B12X_W4A8_TINY_DECODE=0 \
  -e B12X_CUTE_COMPILE_CACHE_DIR=/c/cute -e CUTE_DSL_CACHE_DIR=/c/cutedsl \
  -e B12X_COMPILE_CACHE_DIR=/c/b12x --entrypoint python3 $IMAGE /r/transition_map.py vocab_served \
  > "$out/map-vocab-served.txt" 2>&1
log "replays done"
grep -hE '"changes"|"family"|required_workspace|done|Error|Traceback' "$out/map-vocab-served.txt" | cut -c1-400 | head -40
bin/spark3 cluster start --replace --apply | grep -v 'docker run'
bin/spark3 doctor --live 2>&1 | tail -3
log "done"
