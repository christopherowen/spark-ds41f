#!/bin/bash
# usage: run50.sh   (on dgx1, deployment checkout at this experiment's commit, config = r5o, with the
#                    det-variant2, ref1 and attn-exact5 overlays on every node)
# The reference configuration (vllm-0027 to 0037, VLLM_DS41_BATCH_INVARIANT=1, deterministic MoE,
# prefill chunks aligned to a 4000-token threshold). Cluster stopped:
# 1. transition_map.py reference (mHC with only its capacity plan, the MoE with only its capacity
#    variant) on dgx1.
# 2. detm-r5o-ref-trace: c8_trace.py (one round), every scenario_trace.py scenario once, and
#    trace_mixes.py for the JSON, prose and long targets (five mixes, one repeat, 64 tokens).
#    analyze_trace4.py on every rank while the cluster is stopped.
# 3. Performance, no debug overlay, one pinned cost table, one boot per arm, r5o-pin and
#    detm-r5o-ref-pin: decode (prose and JSON, one and eight streams, three samples), cold prefill of
#    real text (1024-65536 tokens, three repeats), eight distinct concurrent prompts (c8_distinct.py)
#    and short-prompt latency (ttft_short.py).
# Restores r5o.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism/ref
PIN=cache/kkref/dspark-costs/r5o-pin-20260930
IMAGE=vllm-ds41f-kkref:04c30fa98e79-r5o
log() { echo "$(date -u +%FT%TZ) $*"; }
stop_all() {
  for c in config/cluster.json $E/cluster-*.json experiments/2026-09-30-r5o/cluster-*.json \
    experiments/2026-09-30-r5n/cluster-*.json; do
    bin/spark3 --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
  done
}
start() {
  stop_all
  log "start $1"
  bin/spark3 --cluster-config "$1" cluster start --replace --apply | grep -v 'docker run'
}
mkdir -p "$out"
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" 'cd spark3-overlay && find det-variant2 ref1 attn-exact5 -name "*.py" | sort | xargs sha256sum' \
    | sed "s/^/$n /" >> "$out/overlay-bytes-run50.txt"
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
mkdir -p /tmp/lookup/cache && cp $E/transition_map.py /tmp/lookup/
S=$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash
docker run --rm --gpus all --ipc=host $DET \
  -v $S/snapshots/dba1be0a40aa45a94ad051997016db3960a90277:/models:ro -v $S/blobs:/blobs:ro \
  -v /tmp/lookup:/r:ro -v /tmp/lookup/cache:/c -w /opt/spark3/candidate/b12x \
  -e CUTE_DSL_ARCH=sm_121a -e B12X_DENSE_SPLITK_TURBO=0 -e B12X_W4A8_TINY_DECODE=0 \
  -e B12X_CUTE_COMPILE_CACHE_DIR=/c/cute -e CUTE_DSL_CACHE_DIR=/c/cutedsl \
  -e B12X_COMPILE_CACHE_DIR=/c/b12x --entrypoint python3 $IMAGE /r/transition_map.py reference \
  > "$out/map-reference.txt" 2>&1
log "replays done"
grep -hE '"changes"|"family"|Error|Traceback' "$out/map-reference.txt" | cut -c1-300 | head -30
start $E/cluster-detm-r5o-ref-trace.json
python3 $E/c8_trace.py http://10.0.1.71:8000 "$out/c8" --rounds 1 --tokens 16 > "$out/c8-runs.jsonl"
log "c8 exit $?"
python3 $E/scenario_trace.py http://10.0.1.71:8000 "$out/scenarios" --repeats 1 > "$out/scenario-runs.jsonl"
log "scenarios exit $?"
python3 $E/trace_mixes.py http://10.0.1.71:8000 "$out/mixes" --repeats 1 --tokens 64 --prompts json,prose,long \
  > "$out/mixes-runs.jsonl"
log "mixes exit $?"
stop_all
for node in dgx1 dgx2 dgx3; do
  for d in c8 scenarios mixes; do
    docker run --rm --memory=16g -e CUDA_VISIBLE_DEVICES= -v $PWD/$E/analyze_trace4.py:/a.py:ro \
      -v $PWD/$out/$d:/t:ro --entrypoint python3 $IMAGE /a.py /t --node $node --chain 12 --all-pairs 2>&1 \
      | grep -v Warn > "$out/analysis4-$d-$node.jsonl"
  done
done
log "analysed"
measure() {  # arm label
  start $E/cluster-$1.json
  docker logs dsv41-karmic-kraken 2>&1 | grep -E "pinned DSpark cost curves|Pinned DSpark cost curves" | tail -1 \
    | tee -a "$out/pin.log"
  sha256sum $PIN/*.json | tee -a "$out/pin.log"
  bin/spark3 --cluster-config $E/cluster-$1.json bench --allow-mismatch --compare none --suites decode,prefill \
    --decode-cases prose,json-nothink --concurrency 1,8 --min-samples 3 --max-samples 3 \
    --prefill-text source --prefill-sizes 1024,4096,16384,65536 --prefill-repeats 3 \
    --output "results/private/bench/ref-$2"
  log "bench $2 exit $?"
  python3 $E/c8_distinct.py http://10.0.1.71:8000 --samples 3 --tokens 256 | tee "$out/c8-distinct-$2.jsonl"
  python3 $E/ttft_short.py http://10.0.1.71:8000 | tee "$out/ttft-$2.jsonl"
  log "extra $2 exit $?"
}
measure r5o-pin r5o
measure detm-r5o-ref-pin ref
stop_all
bin/spark3 cluster start --replace --apply | grep -v 'docker run'
bin/spark3 doctor --live 2>&1 | tail -3
log "done"
