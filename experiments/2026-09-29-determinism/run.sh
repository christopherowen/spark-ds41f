#!/bin/bash
# usage: run.sh   (on dgx1, deployment checkout at this experiment's commit, after overlay.sh)
# Lean screen, one boot per arm (det, detsk, r5m): determinism probe (three
# prompts, five identical temperature-0 requests each), one round of prefill
# chunk timings, and decode (JSON answers and prose at one and eight streams,
# three samples). Leaves r5m running.
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
arm() {
  start "$1"
  python3 $E/determinism.py http://10.0.1.71:8000 --repeats 5 --tokens 256 | tee "$out/probe-$2.jsonl"
  log "probe $2 exit ${PIPESTATUS[0]}"
  python3 experiments/2026-09-29-indexer-split/capture_depth.py http://10.0.1.71:8000 --rounds 1 \
    | tee "$out/depth-$2.jsonl"
  log "prefill $2 exit ${PIPESTATUS[0]}"
  bin/spark3 --cluster-config "$1" bench --allow-mismatch --compare none --suites decode \
    --decode-cases prose,json-nothink --concurrency 1,8 --min-samples 3 --max-samples 3 \
    --output "results/private/bench/determinism-$2"
  log "decode $2 exit $?"
}
arm $E/cluster-det.json det
arm $E/cluster-detsk.json detsk
arm config/cluster.json r5m
log "done"
