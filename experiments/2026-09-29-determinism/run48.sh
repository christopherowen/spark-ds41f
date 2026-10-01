#!/bin/bash
# usage: run48.sh   (on dgx1, deployment checkout at this experiment's commit, config = r5o, with the
#                    det-variant2, gemv-lookup-mhc-bi-fp8, gemv-lookup-mhc-bi-fp8-rs2 and attn-exact4
#                    overlays on every node)
# Cluster stopped: transition_map.py reference (mHC as served and forced to one native configuration,
# the MoE with only its 4096-token variant warm, the LM head on SIMT plans) on dgx1. Then
# detm-r5o-rs2-exact4-trace with the index capture (selection, top-k scores and inputs for layers 2,
# 8 and 14 at positions 1530-1544): scenario_trace.py cache and chunked_end. analyze_trace4.py and
# compare_index_capture.py while the cluster is stopped, then restores r5o.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism/map3
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
    | sed "s/^/$n /" >> "$out/overlay-bytes-run48.txt"
done
curl -s http://10.0.1.71:8000/metrics | grep -E '^vllm:num_requests_running' || true
stop_all
B=/opt/spark3/candidate/b12x
DET=""
for f in b12x/moe/fused_moe/_impl.py b12x/moe/fused_moe/_preparation.py b12x/moe/fused_moe/_tuning.py \
  b12x/moe/_shared/kernels/dynamic.py b12x/moe/_shared/kernels/silu.py b12x/moe/_shared/kernels/w4a16/kernel.py; do
  DET="$DET -v /home/swank/spark3-overlay/det-variant2/$f:$B/$f:ro"
done
log "reference replays"
ssh -n dgx1 'mkdir -p /tmp/lookup/cache'
cp $E/transition_map.py /tmp/lookup/
S=$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash
docker run --rm --gpus all --ipc=host $DET \
  -v $S/snapshots/dba1be0a40aa45a94ad051997016db3960a90277:/models:ro -v $S/blobs:/blobs:ro \
  -v /tmp/lookup:/r:ro -v /tmp/lookup/cache:/c -w /opt/spark3/candidate/b12x \
  -e CUTE_DSL_ARCH=sm_121a -e B12X_DENSE_SPLITK_TURBO=0 -e B12X_W4A8_TINY_DECODE=0 \
  -e B12X_CUTE_COMPILE_CACHE_DIR=/c/cute -e CUTE_DSL_CACHE_DIR=/c/cutedsl \
  -e B12X_COMPILE_CACHE_DIR=/c/b12x --entrypoint python3 $IMAGE /r/transition_map.py reference \
  > "$out/map-reference.txt" 2>&1
log "replays done"
grep -hE '"changes"|"family"|done|Error|Traceback' "$out/map-reference.txt" | cut -c1-400 | head -30
log "start exact4 trace"
bin/spark3 --cluster-config $E/cluster-detm-r5o-rs2-exact4-trace.json cluster start --replace --apply | grep -v 'docker run'
python3 $E/scenario_trace.py http://10.0.1.71:8000 "$out/scenarios" --scenarios cache,chunked_end --repeats 1 \
  | tee "$out/scenario-runs.jsonl"
log "scenarios exit ${PIPESTATUS[0]}"
stop_all
for node in dgx1 dgx2 dgx3; do
  docker run --rm --memory=16g -e CUDA_VISIBLE_DEVICES= -v $PWD/$E/analyze_trace4.py:/a.py:ro \
    -v $PWD/$out/scenarios:/t:ro --entrypoint python3 $IMAGE /a.py /t --node $node --chain 12 2>&1 \
    | grep -v Warn > "$out/analysis4-scenarios-$node.jsonl"
done
docker run --rm --memory=16g -e CUDA_VISIBLE_DEVICES= -v $PWD/$E/compare_index_capture.py:/c.py:ro \
  -v $PWD/$out/scenarios:/t:ro --entrypoint python3 $IMAGE /c.py /t/logs-cache-0-prefix-cold-r0 \
  /t/logs-cache-prefix-warm-r0 2>&1 | grep -v Warn > "$out/index-capture-cache-prefix.txt"
log "analysed"
bin/spark3 cluster start --replace --apply | grep -v 'docker run'
bin/spark3 doctor --live 2>&1 | tail -3
log "done"
