#!/bin/bash
# usage: run_profile.sh   (on dgx1, deployment checkout at this experiment's commit)
# Boots cluster-baseline-profile.json, runs capture_depth.py --profile into
# results/private/indexer-split/, then restores the promoted configuration.
# Traces land in cache/kkref/profiles/idx-baseline/ on every node.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-indexer-split
out=results/private/indexer-split
mkdir -p "$out"
log() { echo "$(date -u +%FT%TZ) $*"; }
stop_all() {
  for c in config/cluster.json $E/cluster-*.json experiments/2026-09-29-r5k/cluster-*.json; do
    bin/spark3 --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
  done
}
stop_all
log "start baseline-profile"
bin/spark3 --cluster-config $E/cluster-baseline-profile.json cluster start --replace --apply | grep -v 'docker run'
python3 $E/capture_depth.py http://10.0.1.71:8000 --profile | tee "$out/baseline-depth.jsonl"
log "capture exit ${PIPESTATUS[0]}"
sleep 60  # let every rank finish writing its trace
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" "ls -la ~/projects/spark3-vllm-ds41f/cache/kkref/profiles/idx-baseline/ 2>&1 | tail -n +2"
done
stop_all
log "restore promoted configuration"
bin/spark3 cluster start --replace --apply | grep -v 'docker run'
log "done"
