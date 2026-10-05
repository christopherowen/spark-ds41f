#!/bin/bash
# usage: run57.sh   (on dgx1, deployment checkout at this experiment's commit, config = r5o, with the
#                    det-variant2, ref2 and ref3m overlays on every node)
# mHC recovery, step 2. Cluster stopped: mhc_capture_replay.py again (timings on pre-gathered
# inputs, decode sizes from CUDA graphs), the vllm-0038 GPU regression test and the vllm-0039 unit
# test in fresh containers. Then the affected workloads without tracing, one boot per arm, pinned
# cost table: r5o-pin, detm-r5o-ref2-pin (frozen reference) and detm-r5o-ref3m-pin (ref2 with
# vllm-0039): one-stream decode (prose and JSON, five samples), cold prefill of real text
# (1024-65536 tokens, three repeats), eight distinct concurrent prompts, short-prompt latency and
# mixed-traffic latency. Restores r5o.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism/mhc2
PIN=cache/kkref/dspark-costs/r5o-pin-20260930
IMAGE=vllm-ds41f-kkref:04c30fa98e79-r5o
V=/opt/spark3/candidate/vllm
log() { echo "$(date -u +%FT%TZ) $*"; }
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
mkdir -p "$out"
git fetch -q origin
git merge-base --is-ancestor HEAD origin/main || { log "deployment commit not on origin/main; not stopping"; exit 1; }
for n in dgx2 dgx3; do
  [ "$(ssh -n $n git -C projects/spark3-vllm-ds41f rev-parse HEAD)" = "$(git rev-parse HEAD)" ] \
    || { log "$n checkout differs from dgx1; not stopping"; exit 1; }
done
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" 'cd spark3-overlay && find det-variant2 ref2 ref3m -name "*.py" | sort | xargs sha256sum' \
    | sed "s/^/$n /" >> "$out/overlay-bytes-run57.txt"
done
curl -s http://10.0.1.71:8000/metrics | grep -E '^vllm:num_requests_running' || true
stop_all
log "replay"
mkdir -p /tmp/lookup/cache && cp $E/transition_map.py $E/mhc_capture_replay.py /tmp/lookup/
S=$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash
COMMON="-e CUTE_DSL_ARCH=sm_121a -e B12X_DENSE_SPLITK_TURBO=0 -e B12X_W4A8_TINY_DECODE=0 -e B12X_AUTOTUNE=0
  -e B12X_CUTE_COMPILE_CACHE_DIR=/c/cute -e CUTE_DSL_CACHE_DIR=/c/cutedsl -e B12X_COMPILE_CACHE_DIR=/c/b12x"
docker run --rm --gpus all --ipc=host $COMMON \
  -v $HOME/spark3-overlay/ref3m/vllm/models/deepseek_v4_1/b12x_layers.py:$V/vllm/models/deepseek_v4_1/b12x_layers.py:ro \
  -v $S/snapshots/dba1be0a40aa45a94ad051997016db3960a90277:/models:ro -v $S/blobs:/blobs:ro \
  -v /tmp/lookup:/r:ro -v /tmp/lookup/cache:/c -v $PWD/results/private/determinism/mhc/captures:/cap:ro \
  -w /opt/spark3/candidate/b12x --entrypoint python3 $IMAGE /r/mhc_capture_replay.py /cap > "$out/replay.txt" 2>&1
log "replay exit $?"
grep -hE '"groups"|"agreement"|"timing_us"|done|Error|Traceback' "$out/replay.txt" | cut -c1-300 | head -40
log "tests"
W=$HOME/work/vllm-gemv
docker run --rm --gpus all --ipc=host $COMMON -v /tmp/lookup/cache:/c \
  -v $W/vllm/model_executor/layers/logits_processor.py:$V/vllm/model_executor/layers/logits_processor.py:ro \
  -v $W/vllm/models/deepseek_v4_1/b12x_layers.py:$V/vllm/models/deepseek_v4_1/b12x_layers.py:ro \
  -v $W/tests/v1/sample/test_batch_invariant_vocab_projection.py:/t/test_bi_vocab.py:ro \
  -v $W/tests/models/test_deepseek_v4_1_mhc_batch_invariant.py:/t/test_bi_mhc.py:ro \
  -v /tmp/bi-conftest/conftest.py:/t/conftest.py:ro -w /t \
  --entrypoint python3 $IMAGE -m pytest -q -p no:cacheprovider /t/test_bi_vocab.py /t/test_bi_mhc.py \
  > "$out/tests.txt" 2>&1
log "tests exit $?"
tail -3 "$out/tests.txt"
measure() {  # arm label
  start $E/cluster-$1.json
  docker logs dsv41-karmic-kraken 2>&1 | grep -E "pinned DSpark cost curves|Pinned DSpark cost curves" | tail -1 \
    | tee -a "$out/pin.log"
  sha256sum $PIN/*.json | tee -a "$out/pin.log"
  bin/spark --cluster-config $E/cluster-$1.json bench --allow-mismatch --compare none --suites decode,prefill \
    --decode-cases prose,json-nothink --concurrency 1 --min-samples 5 --max-samples 5 \
    --prefill-text source --prefill-sizes 1024,4096,16384,65536 --prefill-repeats 3 \
    --output "results/private/bench/mhc2-$2"
  log "bench $2 exit $?"
  python3 $E/c8_distinct.py http://10.0.1.71:8000 --samples 3 --tokens 256 | tee "$out/c8-distinct-$2.jsonl"
  python3 $E/ttft_short.py http://10.0.1.71:8000 | tee "$out/ttft-$2.jsonl"
  python3 $E/mixed_latency.py http://10.0.1.71:8000 --rounds 3 | tee "$out/mixed-$2.jsonl"
  log "extra $2 exit $?"
}
measure r5o-pin r5o
measure detm-r5o-ref2-pin ref2
measure detm-r5o-ref3m-pin ref3m
stop_all
bin/spark cluster start --replace --apply | grep -v 'docker run'
bin/spark doctor --live 2>&1 | tail -3
log "done"
