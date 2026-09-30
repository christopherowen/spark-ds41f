#!/bin/bash
# usage: run13.sh [ARM]   (on dgx1, deployment checkout at this experiment's commit, after the
#                          moe-checksum overlay; ARM defaults to detslice-static)
# Boots ARM and runs the determinism probe, then restores r5m.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
ARM=${1:-detslice-static}
out=results/private/determinism
log() { echo "$(date -u +%FT%TZ) $*"; }
stop_all() {
  for c in config/cluster.json $E/cluster-*.json; do
    bin/spark3 --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
  done
}
stop_all
log "start $ARM"
bin/spark3 --cluster-config $E/cluster-$ARM.json cluster start --replace --apply | grep -v 'docker run'
python3 $E/determinism.py http://10.0.1.71:8000 --repeats 5 --tokens 256 | tee "$out/probe-$ARM.jsonl"
log "probe $ARM exit ${PIPESTATUS[0]}"
stop_all
bin/spark3 cluster start --replace --apply | grep -v 'docker run'
log "done"
