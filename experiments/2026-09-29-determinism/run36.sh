#!/bin/bash
# usage: run36.sh [--skip-trace]   (on dgx1, deployment checkout at this experiment's commit, config =
#        r5o, with the det-variant, moe-variant, gemv-lookup-mhc, gemv-lookup-mhc-bi-fp8 and attn-exact2
#        overlays on every node)
# 1. Trace 6: detm-r5o-bi-trace (exact fingerprints; all five fixes, batch-invariant mode), JSON
#    and prose, five mixes, three repeats.
# 2. Performance, no debug overlay, one pinned cost table, one boot per arm: short-prompt
#    latency and decode (prose and JSON answers, one and eight streams, six samples) for
#    r5o-pin, r5o-lookup-mhc-variant-pin, detm-r5o-pin and detm-r5o-bi-pin.
# Restores r5o, then analyze_trace2.py (--main --chain 12) and analyze_trace3.py on trace 6.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism/lookup
PIN=cache/kkref/dspark-costs/r5o-pin-20260930
IMAGE=vllm-ds41f-kkref:04c30fa98e79-r5o
log() { echo "$(date -u +%FT%TZ) $*"; }
stop_all() {
  for c in config/cluster.json $E/cluster-*.json experiments/2026-09-30-r5o/cluster-*.json \
    experiments/2026-09-30-r5n/cluster-*.json; do
    bin/spark --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
  done
}
start() {
  stop_all
  log "start $1"
  bin/spark --cluster-config "$1" cluster start --replace --apply | grep -v 'docker run'
}
pinned() {
  docker logs dsv41-karmic-kraken 2>&1 | grep -E "pinned DSpark cost curves|Pinned DSpark cost curves" | tail -1 \
    | tee -a "$out/pin.log"
  sha256sum $PIN/*.json | tee -a "$out/pin.log"
}
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" 'cd spark3-overlay && find gemv-lookup-mhc gemv-lookup-mhc-bi-fp8 moe-variant det-variant attn-exact -name "*.py" | sort | xargs sha256sum' \
    | sed "s/^/$n /" >> "$out/overlay-bytes-run36.txt"
done
if [[ "$*" != *--skip-trace* ]]; then
  start $E/cluster-detm-r5o-bi-trace.json
  for n in dgx1 dgx2 dgx3; do
    ssh -n $n "docker exec dsv41-karmic-kraken sh -c 'rm -f /cache/kkref/moe-checksums/plans-rank*.json'" || true
  done
  python3 $E/trace_mixes.py http://10.0.1.71:8000 "$out/trace6" --repeats 3 --tokens 128 | tee "$out/trace6-runs.jsonl"
  log "trace6 exit ${PIPESTATUS[0]}"
fi
measure() {  # arm label
  start $E/cluster-$1.json
  pinned
  python3 $E/ttft_short.py http://10.0.1.71:8000 | tee "$out/ttft-$2.jsonl"
  log "ttft $2 exit ${PIPESTATUS[0]}"
  bin/spark --cluster-config $E/cluster-$1.json bench --allow-mismatch --compare none --suites decode \
    --decode-cases prose,json-nothink --concurrency 1,8 --min-samples 6 --max-samples 6 \
    --output "results/private/bench/lookup-$2"
  log "decode $2 exit $?"
}
measure r5o-pin r5o-b
measure r5o-lookup-mhc-variant-pin lookup-mhc-variant
measure detm-r5o-pin detm-b
measure detm-r5o-bi-pin detm-bi
start config/cluster.json
bin/spark doctor --live 2>&1 | tail -3
for prompt in json prose; do
  docker run --rm -e CUDA_VISIBLE_DEVICES= -v $PWD/$E/analyze_trace2.py:/a.py:ro -v $PWD/$out/trace6:/t:ro \
    --entrypoint python3 $IMAGE /a.py /t $prompt --main --chain 12 2>&1 | grep -v Warn \
    > "$out/analysis-trace6-$prompt-main.jsonl"
  docker run --rm -e CUDA_VISIBLE_DEVICES= -v $PWD/$E/analyze_trace3.py:/a.py:ro -v $PWD/$out/trace6:/t:ro \
    --entrypoint python3 $IMAGE /a.py /t $prompt --chain 10 2>&1 | grep -v Warn \
    > "$out/analysis3-trace6-$prompt.jsonl"
done
log "done"
