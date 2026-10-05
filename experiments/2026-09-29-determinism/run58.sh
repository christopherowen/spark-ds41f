#!/bin/bash
# usage: run58.sh   (on dgx1, deployment checkout at this experiment's commit, config = r5o, with the
#                    ref3m2 overlay on dgx1)
# Cluster stopped, GPU replays on the run56 mHC captures (no serving):
# 1. mhc_capture_replay.py with the revised vllm-0039 (ref3m2: the capacity plan computes all 25
#    partial sums per CTA): bits across batch sizes, candidate variants grouping 4/13/25 partial sums
#    must agree bit for bit and equal production at decode sizes; timings.
# 2. gemv_capture_replay.py: router gate and compressors on real rows under production, ref2 (SIMT),
#    the TMA prefill kernel and the MMA kernel at every capacity: bits and timings.
# 3. The vllm-0038 GPU regression test, the vllm-0039 unit tests and the vllm-0041 fused-sum GPU test
#    (fresh container, the patched files from the vLLM clone).
# Restores r5o.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism/mhc3
IMAGE=vllm-ds41f-kkref:04c30fa98e79-r5o
V=/opt/spark3/candidate/vllm
log() { echo "$(date -u +%FT%TZ) $*"; }
stop_all() {
  for c in config/cluster.json $E/cluster-*.json experiments/2026-09-30-r5o/cluster-*.json \
    experiments/2026-09-30-r5n/cluster-*.json; do
    bin/spark --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
  done
}
mkdir -p "$out"
git fetch -q origin
git merge-base --is-ancestor HEAD origin/main || { log "deployment commit not on origin/main; not stopping"; exit 1; }
for n in dgx2 dgx3; do
  [ "$(ssh -n $n git -C projects/spark3-vllm-ds41f rev-parse HEAD)" = "$(git rev-parse HEAD)" ] \
    || { log "$n checkout differs from dgx1; not stopping"; exit 1; }
done
sha256sum $HOME/spark3-overlay/ref3m2/vllm/models/deepseek_v4_1/b12x_layers.py > "$out/overlay-bytes-run58.txt"
curl -s http://10.0.1.71:8000/metrics | grep -E '^vllm:num_requests_running' || true
stop_all
mkdir -p /tmp/lookup/cache && cp $E/transition_map.py $E/mhc_capture_replay.py $E/gemv_capture_replay.py /tmp/lookup/
S=$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash
COMMON="-e CUTE_DSL_ARCH=sm_121a -e B12X_DENSE_SPLITK_TURBO=0 -e B12X_W4A8_TINY_DECODE=0 -e B12X_AUTOTUNE=0
  -e B12X_CUTE_COMPILE_CACHE_DIR=/c/cute -e CUTE_DSL_CACHE_DIR=/c/cutedsl -e B12X_COMPILE_CACHE_DIR=/c/b12x"
for script in mhc_capture_replay gemv_capture_replay; do
  log "$script"
  docker run --rm --gpus all --ipc=host $COMMON \
    -v $HOME/spark3-overlay/ref3m2/vllm/models/deepseek_v4_1/b12x_layers.py:$V/vllm/models/deepseek_v4_1/b12x_layers.py:ro \
    -v $S/snapshots/dba1be0a40aa45a94ad051997016db3960a90277:/models:ro -v $S/blobs:/blobs:ro \
    -v /tmp/lookup:/r:ro -v /tmp/lookup/cache:/c -v $PWD/results/private/determinism/mhc/captures:/cap:ro \
    -w /opt/spark3/candidate/b12x --entrypoint python3 $IMAGE /r/$script.py /cap > "$out/$script.txt" 2>&1
  log "$script exit $?"
  grep -hE '"groups"|"agreement"|done|Error|Traceback' "$out/$script.txt" | cut -c1-260 | head -40
done
log "tests"
W=$HOME/work/vllm-gemv
docker run --rm --gpus all --ipc=host $COMMON -v /tmp/lookup/cache:/c \
  -v $W/vllm/model_executor/layers/logits_processor.py:$V/vllm/model_executor/layers/logits_processor.py:ro \
  -v $W/vllm/models/deepseek_v4_1/b12x_layers.py:$V/vllm/models/deepseek_v4_1/b12x_layers.py:ro \
  -v $W/vllm/distributed/device_communicators/cuda_communicator.py:$V/vllm/distributed/device_communicators/cuda_communicator.py:ro \
  -v $W/tests/v1/sample/test_batch_invariant_vocab_projection.py:/t/test_bi_vocab.py:ro \
  -v $W/tests/models/test_deepseek_v4_1_mhc_batch_invariant.py:/t/test_bi_mhc.py:ro \
  -v $W/tests/distributed/test_reduce_scatter_rank_order.py:/t/test_bi_rs.py:ro \
  -v $HOME/projects/spark3-r5/$E/test_conftest.py:/t/conftest.py:ro -w /t \
  --entrypoint python3 $IMAGE -m pytest -q -p no:cacheprovider /t/test_bi_vocab.py /t/test_bi_mhc.py /t/test_bi_rs.py \
  > "$out/tests.txt" 2>&1
log "tests exit $?"
tail -3 "$out/tests.txt"
bin/spark cluster start --replace --apply | grep -v 'docker run'
bin/spark doctor --live 2>&1 | tail -3
log "done"
