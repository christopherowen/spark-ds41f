#!/bin/bash
# usage: run30.sh [--skip-trace]   (on dgx1, deployment checkout at this experiment's commit,
#        config = r5o, with the det-masked, det-variant, moe-variant, gemv-lookup and attn-trace
#        overlays on every node: overlay_masked.sh and overlay_lookup.sh)
# 1. Trace with both fixes: boots detm-r5o-lookup-variant-trace and runs trace_mixes.py
#    (JSON and prose targets, five background mixes, three repeats).
# 2. Performance, no debug overlay, one pinned cost table, one boot per arm: short-prompt
#    latency (ttft_short.py) and decode (prose and JSON answers, one and eight streams, six
#    samples) for r5o-pin, r5o-lookup-pin, r5o-lookup-variant-pin, detm-r5o-pin and
#    detm-r5o-lookup-variant-pin. Restores r5o.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism/lookup
PIN=cache/kkref/dspark-costs/r5o-pin-20260930
mkdir -p "$out"
log() { echo "$(date -u +%FT%TZ) $*"; }
stop_all() {
  for c in config/cluster.json $E/cluster-*.json experiments/2026-09-30-r5o/cluster-*.json \
    experiments/2026-09-30-r5n/cluster-*.json; do
    bin/spark3 --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
  done
}
start() {
  stop_all
  log "start $1"
  bin/spark3 --cluster-config "$1" cluster start --replace --apply | grep -v 'docker run'
}
pinned() {
  docker logs dsv41-karmic-kraken 2>&1 | grep -E "pinned DSpark cost curves|Pinned DSpark cost curves" | tail -1 \
    | tee -a "$out/pin.log"
  sha256sum $PIN/*.json | tee -a "$out/pin.log"
}
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" 'cd spark3-overlay && find gemv-lookup attn-trace det-variant moe-variant -name "*.py" | sort | xargs sha256sum' \
    | sed "s/^/$n /" >> "$out/overlay-bytes-run30.txt"
done
if [[ "$*" != *--skip-trace* ]]; then
  start $E/cluster-detm-r5o-lookup-variant-trace.json
  pinned
  python3 $E/trace_mixes.py http://10.0.1.71:8000 "$out/trace2" --repeats 3 --tokens 128 | tee "$out/trace2-runs.jsonl"
  log "trace exit ${PIPESTATUS[0]}"
  docker exec dsv41-karmic-kraken sh -c 'cat /cache/kkref/moe-checksums/plans-rank0.json' > "$out/plans-rank0.json"
fi
measure() {  # arm label
  start $E/cluster-$1.json
  pinned
  python3 $E/ttft_short.py http://10.0.1.71:8000 | tee "$out/ttft-$2.jsonl"
  log "ttft $2 exit ${PIPESTATUS[0]}"
  bin/spark3 --cluster-config $E/cluster-$1.json bench --allow-mismatch --compare none --suites decode \
    --decode-cases prose,json-nothink --concurrency 1,8 --min-samples 6 --max-samples 6 \
    --output "results/private/bench/lookup-$2"
  log "decode $2 exit $?"
}
measure r5o-pin r5o
measure r5o-lookup-pin lookup
measure r5o-lookup-variant-pin lookup-variant
measure detm-r5o-pin detm
measure detm-r5o-lookup-variant-pin detm-lookup-variant
start config/cluster.json
bin/spark3 doctor --live 2>&1 | tail -3
log "done"
