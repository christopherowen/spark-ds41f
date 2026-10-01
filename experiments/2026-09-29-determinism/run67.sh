#!/bin/bash
# usage: run67.sh CANDIDATE CHUNK   (on dgx1, deployment checkout at this experiment's commit, config = r5o,
#                    with the det-variant2, ref4d, ref4e, gemv-geom2, mhc-mt2 and attn-exact8 overlays on every node)
# The candidate detm-r5o-CANDIDATE-{pin,trace8} (for example ref4d-b4144 4096: ref4c with vllm-0044 and
# B12X 0008, 4096-token chunks and a 4144-token budget; CHUNK is its long-prefill threshold):
# 1. The vLLM unit tests (0038 GPU, 0039 with 0044, 0040, 0041 GPU, 0042) in a fresh container.
# 2. Without tracing, one boot per arm, pinned cost table: r5o-pin and detm-r5o-CANDIDATE-pin: single-stream
#    decode over 24 distinct prompts and the bench's prose and JSON streams, cold prefill 1K-64K, eight distinct
#    prompts, short-prompt and mixed-traffic latency.
# 3. Full validation of detm-r5o-CANDIDATE-trace8: boot A: c8_trace.py, every scenario (chunked_end sized to the
#    chunk; the long prefix-cache case), trace_mixes.py; boot B (a restart): every scenario again under
#    "-bootB"; analyze_trace4.py on every rank while the cluster is stopped. Restores r5o.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
CAND=$1
CHUNK=$2
out=results/private/determinism/$CAND
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
  ssh -n "$n" 'cd spark3-overlay && find det-variant2 ref4d ref4e gemv-geom2 mhc-mt2 attn-exact8 -name "*.py" | sort | xargs sha256sum' \
    | sed "s/^/$n /" >> results/private/determinism/overlay-bytes-run67.txt
done
curl -s http://10.0.1.71:8000/metrics | grep -E '^vllm:num_requests_running' || true
stop_all

# 1. Unit tests.
mkdir -p /tmp/lookup/cache
M=$HOME/spark3-overlay/mhc-mt2/b12x/norm/mhc
MB=/opt/spark3/candidate/b12x/b12x/norm/mhc
log "tests"
docker run --rm --gpus all --ipc=host -e CUTE_DSL_ARCH=sm_121a -e B12X_AUTOTUNE=0 \
  -e B12X_CUTE_COMPILE_CACHE_DIR=/c/cute -e CUTE_DSL_CACHE_DIR=/c/cutedsl -e B12X_COMPILE_CACHE_DIR=/c/b12x \
  -v /tmp/lookup/cache:/c \
  -v $M/_kernels.py:$MB/_kernels.py:ro -v $M/_preparation.py:$MB/_preparation.py:ro -v $M/_tuning.py:$MB/_tuning.py:ro \
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

# 2. Measurement.
mout=results/private/determinism/rec5
mkdir -p "$mout"
measure() {  # arm label
  start $E/cluster-$1.json || { log "start $1 failed"; return 1; }
  docker logs dsv41-karmic-kraken 2>&1 | grep -E "pinned DSpark cost curves|Pinned DSpark cost curves" | tail -1 \
    | tee -a "$mout/pin.log"
  sha256sum $PIN/*.json | tee -a "$mout/pin.log"
  bin/spark3 --cluster-config $E/cluster-$1.json bench --allow-mismatch --compare none --suites decode,prefill \
    --decode-cases prose,json-nothink --concurrency 1 --min-samples 5 --max-samples 5 \
    --prefill-text source --prefill-sizes 1024,4096,16384,65536 --prefill-repeats 3 \
    --output "results/private/bench/rec5-$2"
  log "bench $2 exit $?"
  python3 $E/c1_distinct.py http://10.0.1.71:8000 --tokens 256 | tee "$mout/c1-distinct-$2.jsonl"
  python3 $E/c8_distinct.py http://10.0.1.71:8000 --samples 3 --tokens 256 | tee "$mout/c8-distinct-$2.jsonl"
  python3 $E/ttft_short.py http://10.0.1.71:8000 | tee "$mout/ttft-$2.jsonl"
  python3 $E/mixed_latency.py http://10.0.1.71:8000 --rounds 3 | tee "$mout/mixed-$2.jsonl"
  log "extra $2 exit $?"
}
measure r5o-pin r5o
measure detm-r5o-$CAND-pin $CAND || {
  stop_all; bin/spark3 cluster start --replace --apply | grep -v 'docker run'; bin/spark3 doctor --live 2>&1 | tail -3
  log "done ($CAND did not boot)"; exit 1
}
TRACE=detm-r5o-$CAND-trace8

# 3. Validation.
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
