#!/bin/bash
# usage: run_split.sh   (on dgx1, deployment checkout at this experiment's commit,
# after overlay.sh)
# 1. split-check: capture_depth.py with four background decode streams; every
#    rank compares its split indexer rows with the full-row indexer.
# 2. split: capture_depth.py timings, then the lean bench screen.
# Restores the promoted configuration at the end.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-indexer-split
out=results/private/indexer-split
mkdir -p "$out"
log() { echo "$(date -u +%FT%TZ) $*"; }
stop_all() {
  for c in config/cluster.json $E/cluster-*.json experiments/2026-09-29-r5k/cluster-*.json; do
    bin/spark --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
  done
}
start() {
  stop_all
  log "start $1"
  bin/spark --cluster-config "$E/cluster-$1.json" cluster start --replace --apply | grep -v 'docker run'
}
start split-check
python3 $E/capture_depth.py http://10.0.1.71:8000 --rounds 1 --background 4 | tee "$out/split-check-depth.jsonl"
log "check capture exit ${PIPESTATUS[0]}"
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" "docker logs dsv41-karmic-kraken 2>&1 | grep -c 'indexer split check: .* match the full' | sed \"s/^/\$(hostname) matching reports: /\";" \
    "docker logs dsv41-karmic-kraken 2>&1 | grep -c 'indexer split check: .* differ from the full' | sed \"s/^/\$(hostname) differing reports: /\";" \
    "docker logs dsv41-karmic-kraken 2>&1 | grep 'indexer split check' | tail -3 | cut -c1-260"
done | tee "$out/split-check-logs.txt"
start split
python3 $E/capture_depth.py http://10.0.1.71:8000 --rounds 2 | tee "$out/split-depth.jsonl"
log "split capture exit ${PIPESTATUS[0]}"
bin/spark --cluster-config "$E/cluster-split.json" bench --allow-mismatch --compare none \
  --suites quality,decode,prefill --decode-cases prose,code \
  --concurrency 1,8 --min-samples 3 --max-samples 3 --prefill-text source \
  --output "results/private/bench/indexer-split-split"
log "bench exit $?"
stop_all
log "restore promoted configuration"
bin/spark cluster start --replace --apply | grep -v 'docker run'
log "done"
