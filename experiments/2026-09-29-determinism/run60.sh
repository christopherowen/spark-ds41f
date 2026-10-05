#!/bin/bash
# usage: run60.sh   (on dgx1, deployment checkout at this experiment's commit, config = r5o, with the
#                    det-variant2, ref3c and attn-exact6 overlays on every node)
# Full validation of the recovered reference ref3c (vllm-0027-0031, 0036-0041 with 0039 in place of 0033
# and 0040 in place of 0032, b12x-0006, deterministic MoE), traced (attn-exact6). Cluster stopped:
# 1. The vLLM unit tests for 0038 (GPU), 0039, 0040 and 0041 (GPU) in a fresh container.
# 2. Boot A: c8_trace.py, every scenario_trace.py scenario (now with the long prefix-cache case) and
#    trace_mixes.py for the JSON, prose and long targets; a fresh plan inventory.
# 3. Boot B (a restart of the same arm): every scenario again under the suffix "-bootB", so the
#    analysis compares each group's rows across the two boots.
# analyze_trace4.py on every rank while the cluster is stopped. Restores r5o.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism/ref3c
IMAGE=vllm-ds41f-kkref:04c30fa98e79-r5o
V=/opt/spark3/candidate/vllm
LOGDIR=/cache/kkref/moe-checksums
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
  ssh -n "$n" 'cd spark3-overlay && find det-variant2 ref3c attn-exact6 -name "*.py" | sort | xargs sha256sum' \
    | sed "s/^/$n /" >> "$out/overlay-bytes-run60.txt"
done
curl -s http://10.0.1.71:8000/metrics | grep -E '^vllm:num_requests_running' || true
stop_all
log "tests"
W=$HOME/work/vllm-gemv
docker run --rm --gpus all --ipc=host -e CUTE_DSL_ARCH=sm_121a -e B12X_AUTOTUNE=0 \
  -e B12X_CUTE_COMPILE_CACHE_DIR=/c/cute -e CUTE_DSL_CACHE_DIR=/c/cutedsl -e B12X_COMPILE_CACHE_DIR=/c/b12x \
  -v /tmp/lookup/cache:/c \
  -v $W/vllm/model_executor/layers/logits_processor.py:$V/vllm/model_executor/layers/logits_processor.py:ro \
  -v $W/vllm/models/deepseek_v4_1/b12x_layers.py:$V/vllm/models/deepseek_v4_1/b12x_layers.py:ro \
  -v $W/vllm/distributed/device_communicators/cuda_communicator.py:$V/vllm/distributed/device_communicators/cuda_communicator.py:ro \
  -v $W/tests/v1/sample/test_batch_invariant_vocab_projection.py:/t/test_bi_vocab.py:ro \
  -v $W/tests/models/test_deepseek_v4_1_mhc_batch_invariant.py:/t/test_bi_mhc.py:ro \
  -v $W/tests/distributed/test_reduce_scatter_rank_order.py:/t/test_bi_rs.py:ro \
  -v $PWD/$E/test_conftest.py:/t/conftest.py:ro -w /t \
  --entrypoint python3 $IMAGE -m pytest -q -p no:cacheprovider /t/test_bi_vocab.py /t/test_bi_mhc.py /t/test_bi_rs.py \
  > "$out/tests.txt" 2>&1
log "tests exit $?"
tail -1 "$out/tests.txt"
start $E/cluster-detm-r5o-ref3c-trace6.json
fresh_inventory
python3 $E/c8_trace.py http://10.0.1.71:8000 "$out/c8" --rounds 1 --tokens 16 > "$out/c8-runs.jsonl"
log "c8 exit $?"
python3 $E/scenario_trace.py http://10.0.1.71:8000 "$out/scenarios" --repeats 1 > "$out/scenario-runs.jsonl"
log "scenarios exit $?"
python3 $E/trace_mixes.py http://10.0.1.71:8000 "$out/mixes" --repeats 1 --tokens 64 --prompts json,prose,long \
  > "$out/mixes-runs.jsonl"
log "mixes exit $?"
start $E/cluster-detm-r5o-ref3c-trace6.json
log "boot B"
python3 $E/scenario_trace.py http://10.0.1.71:8000 "$out/scenarios" --repeats 1 --suffix -bootB \
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
bin/spark cluster start --replace --apply | grep -v 'docker run'
bin/spark doctor --live 2>&1 | tail -3
log "done"
