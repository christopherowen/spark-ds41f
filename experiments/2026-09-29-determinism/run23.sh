#!/bin/bash
# usage: run23.sh   (on dgx1, deployment checkout at this experiment's commit, config = r5o,
#                    after overlay_masked.sh and the batch-trace overlay)
# Boots detm-r5o-trace and runs trace_mixes.py (JSON and prose targets, five
# background mixes, three repeats each, per-step traces on every rank), then
# restores r5o. Analysis: analyze_trace.py OUT json|prose.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism/batch-trace
mkdir -p "$out"
log() { echo "$(date -u +%FT%TZ) $*"; }
for c in config/cluster.json $E/cluster-*.json experiments/2026-09-30-r5o/cluster-*.json; do
  bin/spark --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
done
log "start detm-r5o-trace"
bin/spark --cluster-config $E/cluster-detm-r5o-trace.json cluster start --replace --apply | grep -v 'docker run'
docker logs dsv41-karmic-kraken 2>&1 | grep -E "pinned DSpark cost curves" | tail -1
python3 $E/trace_mixes.py http://10.0.1.71:8000 "$out" --repeats 3 --tokens 128 | tee "$out/runs.jsonl"
log "trace exit ${PIPESTATUS[0]}"
for c in $E/cluster-*.json; do
  bin/spark --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
done
bin/spark cluster start --replace --apply | grep -v 'docker run'
log "done"
