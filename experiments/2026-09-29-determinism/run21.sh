#!/bin/bash
# usage: run21.sh   (on dgx1, deployment checkout at this experiment's commit, config = r5n,
#                    after overlay_masked.sh)
# Deterministic MoE with 0008 against r5n, verification policy held fixed by one
# pinned cost table (created by the first r5n-pin-prof boot; its SHA-256 is
# recorded and every later boot must report reusing it). Alternates r5n-pin-prof,
# detm-pin-prof twice: decode bench (prose and JSON answers, one and eight
# streams, six samples) each boot; on the first boot of each, an eight-stream
# JSON profile with spec-decode counters (profile_c8.py); on the first detm boot,
# the sequential and the concurrency repeatability probes. Restores r5n.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism/r5n-pin
PIN=cache/kkref/dspark-costs/r5n-pin-20260930
mkdir -p "$out"
log() { echo "$(date -u +%FT%TZ) $*"; }
stop_all() {
  for c in config/cluster.json $E/cluster-*.json experiments/2026-09-30-r5n/cluster-*.json; do
    bin/spark --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
  done
}
start() {
  stop_all
  log "start $1"
  bin/spark --cluster-config "$1" cluster start --replace --apply | grep -v 'docker run'
}
pinned() {
  docker logs dsv41-karmic-kraken 2>&1 | grep -E "pinned DSpark cost curves|Pinned DSpark cost curves" | tail -2 | tee -a "$out/pin.log"
  sha256sum $PIN/*.json | tee -a "$out/pin.log"
}
decode() {
  bin/spark --cluster-config "$1" bench --allow-mismatch --compare none --suites decode \
    --decode-cases prose,json-nothink --concurrency 1,8 --min-samples 6 --max-samples 6 \
    --output "results/private/bench/r5n-pin-$2"
  log "decode $2 exit $?"
}
profile() {
  python3 $E/profile_c8.py http://10.0.1.71:8000 --case json | tee "$out/profile-$1.json"
  log "profile $1 exit ${PIPESTATUS[0]}"
}
[ -e "$PIN" ] && { log "$PIN already exists; pick a new pin directory"; exit 1; }
start $E/cluster-r5n-pin-prof.json
pinned
decode $E/cluster-r5n-pin-prof.json r5n-1
profile r5n-1
start $E/cluster-detm-pin-prof.json
pinned
python3 $E/determinism.py http://10.0.1.71:8000 --repeats 5 --tokens 256 | tee "$out/probe-sequential.jsonl"
log "sequential probe exit ${PIPESTATUS[0]}"
python3 $E/determinism_concurrent.py http://10.0.1.71:8000 --tokens 256 | tee "$out/probe-concurrent.jsonl"
log "concurrent probe exit ${PIPESTATUS[0]}"
decode $E/cluster-detm-pin-prof.json detm-1
profile detm-1
start $E/cluster-r5n-pin-prof.json
pinned
decode $E/cluster-r5n-pin-prof.json r5n-2
start $E/cluster-detm-pin-prof.json
pinned
decode $E/cluster-detm-pin-prof.json detm-2
start config/cluster.json
log "done"
