#!/bin/bash
# usage: run4.sh   (on dgx1, deployment checkout at this experiment's commit, after overlay.sh)
# Round 4 (0006). With the cluster stopped, runs the B12X GPU tests for slice
# partials and the neighbouring W4A8 paths on dgx3 in a fresh r5m container.
# Then boots detslice (0004-0006, deterministic MoE, split-K through the FP32
# reducer): determinism probe with saved tokens and one round of prefill
# timings; decode (JSON answers and prose at one and eight streams, six
# samples) alternates detslice, r5m, detslice. Leaves r5m running.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism
mkdir -p "$out"
log() { echo "$(date -u +%FT%TZ) $*"; }
stop_all() {
  for c in config/cluster.json $E/cluster-*.json; do
    bin/spark3 --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
  done
}
start() {
  stop_all
  log "start $1"
  bin/spark3 --cluster-config "$1" cluster start --replace --apply | grep -v 'docker run'
}
decode() {
  bin/spark3 --cluster-config "$1" bench --allow-mismatch --compare none --suites decode \
    --decode-cases prose,json-nothink --concurrency 1,8 --min-samples 6 --max-samples 6 \
    --output "results/private/bench/determinism-$2"
  log "decode $2 exit $?"
}
stop_all
log "gpu tests on dgx3"
ssh dgx3 'bash -s' > "$out/gpu-tests.txt" 2>&1 <<'REMOTE'
O=$HOME/spark3-overlay/det-slices
B=/opt/spark3/candidate/b12x
mounts=""
for f in b12x/moe/fused_moe/_impl.py b12x/moe/fused_moe/_preparation.py \
  b12x/moe/fused_moe/_tuning.py b12x/moe/_shared/kernels/dynamic.py \
  b12x/moe/_shared/kernels/silu.py tests/moe/test_w4a8_migration_corpus.py \
  tests/preparation/test_tuning_predicates.py; do
  mounts="$mounts -v $O/$f:$B/$f:ro"
done
run() {
  docker run --rm --gpus all --ipc=host $mounts -w $B -e CUTE_DSL_ARCH=sm_121a \
    -e B12X_CUTE_COMPILE_CACHE_DIR=/tmp/cute -e CUTE_DSL_CACHE_DIR=/tmp/cutedsl \
    -e B12X_COMPILE_CACHE_DIR=/tmp/b12xc --entrypoint python3 \
    vllm-ds41f-kkref:04c30fa98e79-r5m -m pytest -q -p no:cacheprovider "$@"
}
run tests/moe/test_w4a8_migration_corpus.py -k "slice_partials or materialized_routing or (direct_tile and m16 and mx and silu)"
echo "corpus exit $?"
run tests/moe/test_tp_moe_scratch_bindings.py -k deterministic
echo "bindings exit $?"
REMOTE
tail -15 "$out/gpu-tests.txt"
if ! grep -q "corpus exit 0" "$out/gpu-tests.txt" || ! grep -q "bindings exit 0" "$out/gpu-tests.txt"; then
  log "gpu tests failed; restoring r5m"
  start config/cluster.json
  exit 1
fi
log "gpu tests passed"
start $E/cluster-detslice.json
python3 $E/determinism.py http://10.0.1.71:8000 --repeats 5 --tokens 256 \
  --save "$out/tokens-detslice.json" | tee "$out/probe-detslice.jsonl"
log "probe detslice exit ${PIPESTATUS[0]}"
python3 experiments/2026-09-29-indexer-split/capture_depth.py http://10.0.1.71:8000 --rounds 1 \
  | tee "$out/depth-detslice.jsonl"
log "prefill detslice exit ${PIPESTATUS[0]}"
decode $E/cluster-detslice.json detslice-a
start config/cluster.json
decode config/cluster.json r5m-c
start $E/cluster-detslice.json
decode $E/cluster-detslice.json detslice-b
start config/cluster.json
log "done"
