#!/bin/bash
# usage: run47.sh   (on dgx1, deployment checkout at this experiment's commit, config = r5o, with the
#                    det-variant2, gemv-lookup-mhc-bi-fp8, gemv-lookup-mhc-bi-fp8-rs2 and attn-exact4
#                    overlays on every node)
# The rest of the transition map, then the baseline with the complete harness. Cluster stopped:
# 1. transition_map.py in parallel: mhc (smallest and exact-or-max lookups) on dgx1, moe on dgx2
#    (det-variant2), wo on dgx3.
# 2. detm-r5o-rs2-exact4-trace (attn-exact4 with the CED decoder-row hook and the index capture for
#    layers 2, 8 and 14 at positions 1530-1544): one c8_trace.py round for the plan inventory, then
#    every scenario_trace.py scenario once.
# analyze_trace4.py and compare_index_capture.py while the cluster is stopped, then restores r5o.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism/map2
IMAGE=vllm-ds41f-kkref:04c30fa98e79-r5o
log() { echo "$(date -u +%FT%TZ) $*"; }
stop_all() {
  for c in config/cluster.json $E/cluster-*.json experiments/2026-09-30-r5o/cluster-*.json \
    experiments/2026-09-30-r5n/cluster-*.json; do
    bin/spark3 --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
  done
}
mkdir -p "$out"
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" 'cd spark3-overlay && find det-variant2 gemv-lookup-mhc-bi-fp8 gemv-lookup-mhc-bi-fp8-rs2 attn-exact4 -name "*.py" | sort | xargs sha256sum' \
    | sed "s/^/$n /" >> "$out/overlay-bytes-run47.txt"
done
curl -s http://10.0.1.71:8000/metrics | grep -E '^vllm:num_requests_running' || true
stop_all
replay() {  # node output families [docker args]
  ssh -n "$1" 'mkdir -p /tmp/lookup/cache'
  scp -q $E/transition_map.py "$1":/tmp/lookup/
  ssh "$1" "IMAGE=$IMAGE FAMILIES='$3' EXTRA='$4' bash -s" > "$out/$2" 2>&1 <<'REMOTE'
S=$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash
docker run --rm --gpus all --ipc=host $EXTRA \
  -v $S/snapshots/dba1be0a40aa45a94ad051997016db3960a90277:/models:ro -v $S/blobs:/blobs:ro \
  -v /tmp/lookup:/r:ro -v /tmp/lookup/cache:/c -w /opt/spark3/candidate/b12x \
  -e CUTE_DSL_ARCH=sm_121a -e B12X_DENSE_SPLITK_TURBO=0 -e B12X_W4A8_TINY_DECODE=0 \
  -e B12X_CUTE_COMPILE_CACHE_DIR=/c/cute -e CUTE_DSL_CACHE_DIR=/c/cutedsl \
  -e B12X_COMPILE_CACHE_DIR=/c/b12x --entrypoint python3 $IMAGE /r/transition_map.py $FAMILIES
REMOTE
}
B=/opt/spark3/candidate/b12x
DET=""
for f in b12x/moe/fused_moe/_impl.py b12x/moe/fused_moe/_preparation.py b12x/moe/fused_moe/_tuning.py \
  b12x/moe/_shared/kernels/dynamic.py b12x/moe/_shared/kernels/silu.py b12x/moe/_shared/kernels/w4a16/kernel.py; do
  DET="$DET -v /home/swank/spark3-overlay/det-variant2/$f:$B/$f:ro"
done
log "transition map replays"
replay dgx1 map-mhc.txt "mhc" "" &
replay dgx2 map-moe.txt "moe" "$DET" &
replay dgx3 map-wo.txt "wo" "" &
wait
log "replays done"
grep -hE '"changes"|"members"|"variants"|done|Error|Traceback' "$out"/map-*.txt | cut -c1-500 | head -40
log "start exact4 trace"
bin/spark3 --cluster-config $E/cluster-detm-r5o-rs2-exact4-trace.json cluster start --replace --apply | grep -v 'docker run'
for n in dgx1 dgx2 dgx3; do
  ssh -n $n "docker exec dsv41-karmic-kraken sh -c 'rm -f /cache/kkref/moe-checksums/inventory-rank*.json /cache/kkref/moe-checksums/plans-rank*.json'" || true
done
python3 $E/c8_trace.py http://10.0.1.71:8000 "$out/c8" --rounds 1 --tokens 16 > "$out/c8-runs.jsonl"
log "c8 exit $?"
for n in dgx1 dgx2 dgx3; do
  for f in $(ssh -n $n "docker exec dsv41-karmic-kraken sh -c 'cd /cache/kkref/moe-checksums && ls inventory-rank*.json plans-rank*.json'"); do
    ssh -n $n "docker exec dsv41-karmic-kraken cat /cache/kkref/moe-checksums/$f" > "$out/$n-$f"
  done
done
python3 $E/scenario_trace.py http://10.0.1.71:8000 "$out/scenarios" --repeats 1 | tee "$out/scenario-runs.jsonl"
log "scenarios exit ${PIPESTATUS[0]}"
stop_all
for node in dgx1 dgx2 dgx3; do
  docker run --rm --memory=16g -e CUDA_VISIBLE_DEVICES= -v $PWD/$E/analyze_trace4.py:/a.py:ro \
    -v $PWD/$out/scenarios:/t:ro --entrypoint python3 $IMAGE /a.py /t --node $node --chain 12 2>&1 \
    | grep -v Warn > "$out/analysis4-scenarios-$node.jsonl"
done
docker run --rm --memory=16g -e CUDA_VISIBLE_DEVICES= -v $PWD/$E/compare_index_capture.py:/c.py:ro \
  -v $PWD/$out/scenarios:/t:ro --entrypoint python3 $IMAGE /c.py /t/logs-cache-0-prefix-cold-r0/dgx1 \
  /t/logs-cache-prefix-warm-r0/dgx1 2>&1 | grep -v Warn > "$out/index-capture-cache-prefix.txt"
log "analysed"
bin/spark3 cluster start --replace --apply | grep -v 'docker run'
bin/spark3 doctor --live 2>&1 | tail -3
log "done"
