#!/bin/bash
# usage: run64.sh   (on dgx1, deployment checkout at this experiment's commit, config = r5o, with the
#                    det-variant2, ref4c, gemv-geom2 and attn-exact8 overlays on every node, mhc-mt2 on dgx1)
# 1. Cluster stopped: run63's replay with overlay mhc-mt2 (B12X 0008 rebuilt on the r5o image; the first try's
#    overlay failed the tuning contract's config codec): candidate-t4/t8/t16 against the one-token candidate.
# 2. The vLLM unit tests (0038 GPU, 0039, 0040, 0041 GPU, 0042) in a fresh container.
# 3. Without tracing, one boot per arm, pinned cost table: r5o-pin, detm-r5o-ref4c-pin (ref4b with vllm-0042
#    corrected and gemv-geom2) and detm-r5o-ref4c-b4144-pin (chunks aligned to 4096 tokens, a 4144-token batch
#    budget): single-stream decode over 24 distinct prompts and the bench's prose and JSON streams, cold
#    prefill 1K-64K, eight distinct prompts, short-prompt and mixed-traffic latency.
# 4. Full validation of ref4c-b4144 (ref4c with 4000-token chunks if the 4144 budget did not boot): boot A
#    traced (attn-exact8): c8_trace.py, every scenario (chunked_end sized to the chunk; the long prefix-cache
#    case), trace_mixes.py; boot B (a restart): every scenario again under "-bootB"; analyze_trace4.py on every
#    rank while the cluster is stopped. Restores r5o.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism/mhc5
PIN=cache/kkref/dspark-costs/r5o-pin-20260930
IMAGE=vllm-ds41f-kkref:04c30fa98e79-r5o
V=/opt/spark3/candidate/vllm
W=$HOME/work/vllm-gemv
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
  return "${PIPESTATUS[0]}"
}
LOGDIR=/cache/kkref/moe-checksums
fresh_inventory() {
  for n in dgx1 dgx2 dgx3; do
    ssh -n "$n" "docker exec dsv41-karmic-kraken sh -c 'rm -f $LOGDIR/inventory-* $LOGDIR/plans-*'"
  done
}
mkdir -p "$out"
git fetch -q origin
git merge-base --is-ancestor HEAD origin/main || { log "deployment commit not on origin/main; not stopping"; exit 1; }
for n in dgx2 dgx3; do
  [ "$(ssh -n $n git -C projects/spark3-vllm-ds41f rev-parse HEAD)" = "$(git rev-parse HEAD)" ] \
    || { log "$n checkout differs from dgx1; not stopping"; exit 1; }
done
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" 'cd spark3-overlay && find det-variant2 ref4c gemv-geom2 attn-exact8 -name "*.py" | sort | xargs sha256sum' \
    | sed "s/^/$n /" >> results/private/determinism/overlay-bytes-run64.txt
done
sha256sum $HOME/spark3-overlay/mhc-mt2/b12x/norm/mhc/*.py $W/vllm/models/deepseek_v4_1/b12x_layers.py \
  > "$out/overlay-bytes-run64.txt"
curl -s http://10.0.1.71:8000/metrics | grep -E '^vllm:num_requests_running' || true
stop_all

# 1. mHC multi-token replay.
mkdir -p /tmp/lookup/cache && cp $E/transition_map.py $E/mhc_capture_replay.py $E/gemv_capture_replay.py /tmp/lookup/
S=$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash
M=$HOME/spark3-overlay/mhc-mt2/b12x/norm/mhc
MB=/opt/spark3/candidate/b12x/b12x/norm/mhc
log "mhc_capture_replay"
docker run --rm --gpus all --ipc=host -e CUTE_DSL_ARCH=sm_121a -e B12X_DENSE_SPLITK_TURBO=0 \
  -e B12X_W4A8_TINY_DECODE=0 -e B12X_AUTOTUNE=0 -e B12X_CUTE_COMPILE_CACHE_DIR=/c/cute \
  -e CUTE_DSL_CACHE_DIR=/c/cutedsl -e B12X_COMPILE_CACHE_DIR=/c/b12x \
  -v $W/vllm/models/deepseek_v4_1/b12x_layers.py:$V/vllm/models/deepseek_v4_1/b12x_layers.py:ro \
  -v $M/_kernels.py:$MB/_kernels.py:ro -v $M/_preparation.py:$MB/_preparation.py:ro -v $M/_tuning.py:$MB/_tuning.py:ro \
  -v $S/snapshots/dba1be0a40aa45a94ad051997016db3960a90277:/models:ro -v $S/blobs:/blobs:ro \
  -v /tmp/lookup:/r:ro -v /tmp/lookup/cache:/c -v $PWD/results/private/determinism/mhc/captures:/cap:ro \
  -w /opt/spark3/candidate/b12x --entrypoint python3 $IMAGE /r/mhc_capture_replay.py /cap \
  > "$out/mhc_capture_replay-run64.txt" 2>&1
log "mhc_capture_replay exit $?"
grep -hE '"groups"|"agreement"|done|Error|Traceback' "$out/mhc_capture_replay-run64.txt" | cut -c1-260 | head -40

# 2. Unit tests.
out=results/private/determinism/ref4c
mkdir -p "$out"
log "tests"
docker run --rm --gpus all --ipc=host -e CUTE_DSL_ARCH=sm_121a -e B12X_AUTOTUNE=0 \
  -e B12X_CUTE_COMPILE_CACHE_DIR=/c/cute -e CUTE_DSL_CACHE_DIR=/c/cutedsl -e B12X_COMPILE_CACHE_DIR=/c/b12x \
  -v /tmp/lookup/cache:/c \
  -v $W/vllm/model_executor/layers/logits_processor.py:$V/vllm/model_executor/layers/logits_processor.py:ro \
  -v $W/vllm/models/deepseek_v4_1/b12x_layers.py:$V/vllm/models/deepseek_v4_1/b12x_layers.py:ro \
  -v $W/vllm/distributed/device_communicators/cuda_communicator.py:$V/vllm/distributed/device_communicators/cuda_communicator.py:ro \
  -v $W/vllm/models/deepseek_v4_1/sparse_mla.py:$V/vllm/models/deepseek_v4_1/sparse_mla.py:ro \
  -v $W/vllm/models/deepseek_v4_1/ced.py:$V/vllm/models/deepseek_v4_1/ced.py:ro \
  -v $W/vllm/models/deepseek_v4_1/attention.py:$V/vllm/models/deepseek_v4_1/attention.py:ro \
  -v $W/tests/v1/sample/test_batch_invariant_vocab_projection.py:/t/test_bi_vocab.py:ro \
  -v $W/tests/models/test_deepseek_v4_1_mhc_batch_invariant.py:/t/test_bi_mhc.py:ro \
  -v $W/tests/distributed/test_reduce_scatter_rank_order.py:/t/test_bi_rs.py:ro \
  -v $W/tests/models/test_deepseek_v4_1_attention_batch_invariant.py:/t/test_bi_attn.py:ro \
  -v $PWD/$E/test_conftest.py:/t/conftest.py:ro -w /t \
  --entrypoint python3 $IMAGE -m pytest -q -p no:cacheprovider /t/test_bi_vocab.py /t/test_bi_mhc.py /t/test_bi_rs.py /t/test_bi_attn.py \
  > "$out/tests.txt" 2>&1
log "tests exit $?"
tail -1 "$out/tests.txt"

# 3. Measurement.
mout=results/private/determinism/rec3
mkdir -p "$mout"
measure() {  # arm label
  start $E/cluster-$1.json || { log "start $1 failed"; return 1; }
  docker logs dsv41-karmic-kraken 2>&1 | grep -E "pinned DSpark cost curves|Pinned DSpark cost curves" | tail -1 \
    | tee -a "$mout/pin.log"
  sha256sum $PIN/*.json | tee -a "$mout/pin.log"
  bin/spark3 --cluster-config $E/cluster-$1.json bench --allow-mismatch --compare none --suites decode,prefill \
    --decode-cases prose,json-nothink --concurrency 1 --min-samples 5 --max-samples 5 \
    --prefill-text source --prefill-sizes 1024,4096,16384,65536 --prefill-repeats 3 \
    --output "results/private/bench/rec3-$2"
  log "bench $2 exit $?"
  python3 $E/c1_distinct.py http://10.0.1.71:8000 --tokens 256 | tee "$mout/c1-distinct-$2.jsonl"
  python3 $E/c8_distinct.py http://10.0.1.71:8000 --samples 3 --tokens 256 | tee "$mout/c8-distinct-$2.jsonl"
  python3 $E/ttft_short.py http://10.0.1.71:8000 | tee "$mout/ttft-$2.jsonl"
  python3 $E/mixed_latency.py http://10.0.1.71:8000 --rounds 3 | tee "$mout/mixed-$2.jsonl"
  log "extra $2 exit $?"
}
measure r5o-pin r5o
measure detm-r5o-ref4c-pin ref4c
if measure detm-r5o-ref4c-b4144-pin ref4c-b4144; then
  TRACE=detm-r5o-ref4c-b4144-trace8 CHUNK=4096
else
  TRACE=detm-r5o-ref4c-trace8 CHUNK=4000
fi

# 4. Validation.
log "validating $TRACE (chunk $CHUNK)" | tee "$out/arm.txt"
start $E/cluster-$TRACE.json
fresh_inventory
python3 $E/c8_trace.py http://10.0.1.71:8000 "$out/c8" --rounds 1 --tokens 16 > "$out/c8-runs.jsonl"
log "c8 exit $?"
python3 $E/scenario_trace.py http://10.0.1.71:8000 "$out/scenarios" --repeats 1 --chunk $CHUNK \
  > "$out/scenario-runs.jsonl"
log "scenarios exit $?"
python3 $E/trace_mixes.py http://10.0.1.71:8000 "$out/mixes" --repeats 1 --tokens 64 --prompts json,prose,long \
  > "$out/mixes-runs.jsonl"
log "mixes exit $?"
start $E/cluster-$TRACE.json
log "boot B"
python3 $E/scenario_trace.py http://10.0.1.71:8000 "$out/scenarios" --repeats 1 --chunk $CHUNK --suffix -bootB \
  > "$out/scenario-runs-bootB.jsonl"
log "scenarios boot B exit $?"
stop_all
for node in dgx1 dgx2 dgx3; do
  for d in c8 scenarios mixes; do
    docker run --rm --memory=16g -e CUDA_VISIBLE_DEVICES= -v $PWD/$E/analyze_trace4.py:/a.py:ro \
      -v $PWD/$out/$d:/t:ro --entrypoint python3 $IMAGE /a.py /t --node $node --chain 12 --all-pairs 2>&1 \
      | grep -v Warn > "$out/analysis4-$d-$node.jsonl"
  done
done
log "analysed"
bin/spark3 cluster start --replace --apply | grep -v 'docker run'
bin/spark3 doctor --live 2>&1 | tail -3
log "done"
