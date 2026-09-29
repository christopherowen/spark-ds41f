#!/bin/bash
# usage: run2.sh   (on dgx1, deployment checkout at this experiment's commit, after overlay.sh)
# Round 2: detfast (0004+0005, deterministic MoE, split-K through the FP32
# reducer) and detfast-t (split-K turbo kept). Each boot runs the determinism
# probe (saving run 0 for cross-arm comparison) and one round of prefill
# timings; decode (JSON answers and prose at one and eight streams, six
# samples) alternates r5m, detfast, detfast-t, r5m, detfast. A short detsk probe
# first saves the grouped deterministic path's tokens as the reference that
# detfast must reproduce bit for bit. Leaves r5m running.
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
measure() {
  python3 $E/determinism.py http://10.0.1.71:8000 --repeats 5 --tokens 256 \
    --save "$out/tokens-$2.json" | tee "$out/probe-$2.jsonl"
  log "probe $2 exit ${PIPESTATUS[0]}"
  python3 experiments/2026-09-29-indexer-split/capture_depth.py http://10.0.1.71:8000 --rounds 1 \
    | tee "$out/depth-$2.jsonl"
  log "prefill $2 exit ${PIPESTATUS[0]}"
}
decode() {
  bin/spark3 --cluster-config "$1" bench --allow-mismatch --compare none --suites decode \
    --decode-cases prose,json-nothink --concurrency 1,8 --min-samples 6 --max-samples 6 \
    --output "results/private/bench/determinism-$2"
  log "decode $2 exit $?"
}
start $E/cluster-detsk.json
python3 $E/determinism.py http://10.0.1.71:8000 --repeats 2 --tokens 256 \
  --save "$out/tokens-detsk.json" | tee "$out/probe-detsk-ref.jsonl"
log "probe detsk-ref exit ${PIPESTATUS[0]}"
start config/cluster.json
decode config/cluster.json r5m-a
start $E/cluster-detfast.json
measure $E/cluster-detfast.json detfast
decode $E/cluster-detfast.json detfast-a
start $E/cluster-detfast-t.json
measure $E/cluster-detfast-t.json detfast-t
decode $E/cluster-detfast-t.json detfast-t-a
start config/cluster.json
decode config/cluster.json r5m-b
start $E/cluster-detfast.json
decode $E/cluster-detfast.json detfast-b
start config/cluster.json
log "done"
