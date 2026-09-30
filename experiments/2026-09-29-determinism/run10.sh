#!/bin/bash
# usage: run10.sh   (on dgx1, deployment checkout at this experiment's commit, after overlay.sh)
# Bisection: determinism probe on detslice-noeng (synchronous Engram rows) and
# detslice-nol2 (no L2 weight prefetch), overlap kept. Restores r5m.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism
log() { echo "$(date -u +%FT%TZ) $*"; }
stop_all() {
  for c in config/cluster.json $E/cluster-*.json; do
    bin/spark3 --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
  done
}
for arm in detslice-noeng detslice-nol2; do
  stop_all
  log "start $arm"
  bin/spark3 --cluster-config $E/cluster-$arm.json cluster start --replace --apply | grep -v 'docker run'
  python3 $E/determinism.py http://10.0.1.71:8000 --repeats 5 --tokens 256 | tee "$out/probe-$arm.jsonl"
  log "probe $arm exit ${PIPESTATUS[0]}"
done
stop_all
bin/spark3 cluster start --replace --apply | grep -v 'docker run'
log "done"
