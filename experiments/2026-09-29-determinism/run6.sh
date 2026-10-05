#!/bin/bash
# usage: run6.sh   (on dgx1, deployment checkout at this experiment's commit, after overlay.sh)
# Is detslice's serving nondeterminism tied to the shared-expert side stream?
# Boots detslice-noovl (overlap off) and runs the determinism probe, then
# restores r5m.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism
log() { echo "$(date -u +%FT%TZ) $*"; }
for c in config/cluster.json $E/cluster-*.json; do
  bin/spark --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
done
log "start detslice-noovl"
bin/spark --cluster-config $E/cluster-detslice-noovl.json cluster start --replace --apply | grep -v 'docker run'
python3 $E/determinism.py http://10.0.1.71:8000 --repeats 5 --tokens 256 \
  --save "$out/tokens-detslice-noovl.json" | tee "$out/probe-detslice-noovl.jsonl"
log "probe detslice-noovl exit ${PIPESTATUS[0]}"
for c in config/cluster.json $E/cluster-*.json; do
  bin/spark --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
done
bin/spark cluster start --replace --apply | grep -v 'docker run'
log "done"
