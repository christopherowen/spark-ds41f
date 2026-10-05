#!/bin/bash
# usage: run26.sh [--skip-trace] [--skip-replay] [--skip-perf]
#        (on dgx1, deployment checkout at this experiment's commit, config = r5o, after
#         overlay_masked.sh and with the gemv-lookup and attn-trace overlays on every node)
# 1. Trace: boots detm-r5o-lookup-trace and runs trace_mixes.py (JSON and prose
#    targets, five background mixes, three repeats, per-step traces with
#    attention checksums and the first-call router gate capture).
# 2. Replay on dgx3 (cluster stopped): gemv_lookup_validation.py on the captured
#    gate inputs and served weights, then the lookup patch's GPU tests.
# 3. Performance, no debug overlay, one pinned cost table: short-prompt latency
#    (ttft_short.py) and decode (prose and JSON answers, one and eight streams,
#    six samples), alternating r5o-pin and r5o-lookup-pin twice, then
#    detm-r5o-pin and detm-r5o-lookup-pin once each. Restores r5o.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism/lookup
PIN=cache/kkref/dspark-costs/r5o-pin-20260930
IMAGE=vllm-ds41f-kkref:04c30fa98e79-r5o
mkdir -p "$out"
log() { echo "$(date -u +%FT%TZ) $*"; }
ARGS="$*"
stop_all() {
  for c in config/cluster.json $E/cluster-*.json experiments/2026-09-30-r5o/cluster-*.json \
    experiments/2026-09-30-r5n/cluster-*.json; do
    bin/spark --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
  done
}
start() {
  stop_all
  log "start $1"
  bin/spark --cluster-config "$1" cluster start --replace --apply | grep -v 'docker run'
}
pinned() {
  docker logs dsv41-karmic-kraken 2>&1 | grep -E "pinned DSpark cost curves|Pinned DSpark cost curves" | tail -1 \
    | tee -a "$out/pin.log"
  sha256sum $PIN/*.json | tee -a "$out/pin.log"
}
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" 'cd spark3-overlay && sha256sum gemv-lookup/vllm/models/deepseek_v4_1/*.py attn-trace/*.py' \
    | sed "s/^/$n /" >> "$out/overlay-bytes.txt"
done

if [[ "$ARGS" != *--skip-trace* ]]; then
  start $E/cluster-detm-r5o-lookup-trace.json
  pinned
  python3 $E/trace_mixes.py http://10.0.1.71:8000 "$out/trace" --repeats 3 --tokens 128 | tee "$out/trace-runs.jsonl"
  log "trace exit ${PIPESTATUS[0]}"
  docker exec dsv41-karmic-kraken cat /cache/kkref/moe-checksums/gate-weights-rank0.pt > "$out/gate-weights-rank0.pt"
  ls -la "$out/gate-weights-rank0.pt"
  stop_all
fi

if [[ "$ARGS" != *--skip-replay* ]]; then
  stop_all
  json=$(ls -v "$out"/trace/json-m0-r0/dgx1/rank0-gates-*.pt | tail -1)
  prose=$(ls -v "$out"/trace/prose-m0-r0/dgx1/rank0-gates-*.pt | tail -1)
  ssh -n dgx3 'rm -rf /tmp/lookup && mkdir -p /tmp/lookup/cache'
  scp -q "$json" dgx3:/tmp/lookup/json-gates.pt
  scp -q "$prose" dgx3:/tmp/lookup/prose-gates.pt
  scp -q "$out/gate-weights-rank0.pt" dgx3:/tmp/lookup/gate-weights.pt
  scp -q $E/gemv_lookup_validation.py dgx3:/tmp/lookup/
  scp -q ~/work/vllm-gemv/tests/models/test_deepseek_v4_1_prepared_projections.py dgx3:/tmp/lookup/
  log "replay on dgx3"
  ssh dgx3 "IMAGE=$IMAGE bash -s" > "$out/replay.txt" 2>&1 <<'REMOTE'
S=$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash
M="-v $S/snapshots/dba1be0a40aa45a94ad051997016db3960a90277:/models:ro -v $S/blobs:/blobs:ro"
V=/opt/spark3/candidate/vllm
L=$HOME/spark3-overlay/gemv-lookup/vllm/models/deepseek_v4_1
P="-v $L/b12x_layers.py:$V/vllm/models/deepseek_v4_1/b12x_layers.py:ro -v $L/compressor.py:$V/vllm/models/deepseek_v4_1/compressor.py:ro"
C="-e CUTE_DSL_ARCH=sm_121a -e B12X_CUTE_COMPILE_CACHE_DIR=/c/cute -e CUTE_DSL_CACHE_DIR=/c/cutedsl -e B12X_COMPILE_CACHE_DIR=/c/b12x -v /tmp/lookup/cache:/c"
echo "== validation"
docker run --rm --gpus all --ipc=host $M $C -v /tmp/lookup:/r:ro -w /opt/spark3/candidate/b12x \
  --entrypoint python3 $IMAGE /r/gemv_lookup_validation.py /r/json-gates.pt /r/prose-gates.pt /r/gate-weights.pt
echo "== patch GPU tests"
docker run --rm --gpus all --ipc=host $C $P -v /tmp/lookup/test_deepseek_v4_1_prepared_projections.py:$V/tests/models/test_deepseek_v4_1_prepared_projections.py:ro \
  -w $V --entrypoint python3 $IMAGE -m pytest -q -p no:cacheprovider --noconftest \
  tests/models/test_deepseek_v4_1_prepared_projections.py -k "compressor_split or smallest" 2>&1 | tail -5
docker run --rm --gpus all --ipc=host $C $P -w $V --entrypoint python3 $IMAGE -m pytest -q -p no:cacheprovider \
  --noconftest tests/model_executor/kernels/test_b12x_linear.py -k "v41_unquantized" 2>&1 | tail -5
REMOTE
  log "replay done"
  grep -vE "Warning|warn\(" "$out/replay.txt" | tail -40
fi

if [[ "$ARGS" != *--skip-perf* ]]; then
  measure() {  # arm label
    start $E/cluster-$1.json
    pinned
    python3 $E/ttft_short.py http://10.0.1.71:8000 | tee "$out/ttft-$2.jsonl"
    log "ttft $2 exit ${PIPESTATUS[0]}"
    bin/spark --cluster-config $E/cluster-$1.json bench --allow-mismatch --compare none --suites decode \
      --decode-cases prose,json-nothink --concurrency 1,8 --min-samples 6 --max-samples 6 \
      --output "results/private/bench/lookup-$2"
    log "decode $2 exit $?"
  }
  measure r5o-pin r5o-1
  measure r5o-lookup-pin lookup-1
  measure r5o-pin r5o-2
  measure r5o-lookup-pin lookup-2
  measure detm-r5o-pin detm-1
  measure detm-r5o-lookup-pin detm-lookup-1
fi
start config/cluster.json
bin/spark doctor --live 2>&1 | tail -3
log "done"
