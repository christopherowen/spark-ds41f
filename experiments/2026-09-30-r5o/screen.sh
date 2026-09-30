#!/bin/bash
# usage: screen.sh   (on dgx1, deployment checkout at this experiment's commit, config = r5n,
#                     after overlay.sh and the determinism experiment's overlay_masked.sh)
# 1. Long-prompt repeatability on the deterministic MoE without and with the
#    fenced files (determinism_long.py: one ~6K-token prompt, five runs).
# 2. The files B12X 0005 ships as an overlay on r5n, one pinned cost table for
#    both arms (created by the first r5n-pin boot, SHA-256 recorded): prefill
#    chunk timings (three rounds each), then decode alternating r5n-pin,
#    overlay-pin, r5n-pin, overlay-pin (prose and JSON answers, one and eight
#    streams, six samples). Leaves r5n running.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-30-r5o
out=results/private/r5o
PIN=cache/kkref/dspark-costs/r5o-pin-20260930
mkdir -p "$out"
log() { echo "$(date -u +%FT%TZ) $*"; }
declare -A MEASURED=(
  [b12x/attention/_shared/contiguous/forward.py]=94e95478f0d2d2f20c2d61af5ceb2c8ffe08286ee7d3bab8695774e6f8e9fb2e
  [b12x/gemm/bf16_gemv/_prefill.py]=969fb9b20dcdb5ad9cf92674c687a4b55d7319908adede9928236cd34ead0bd1
  [b12x/norm/mhc/_kernels.py]=fe5b00e967f80dbffe751bb6e140a65842540c6ea664e1b1ce0e9f0ea21908bb
)
for n in dgx1 dgx2 dgx3; do
  for f in "${!MEASURED[@]}"; do
    got=$(ssh -n "$n" sha256sum "spark3-overlay/r5o-fence/$f" | cut -d" " -f1)
    echo "$n $f $got" >> "$out/overlay-bytes.txt"
    [ "$got" = "${MEASURED[$f]}" ] || { log "overlay $f differs on $n"; exit 1; }
  done
done
[ -e "$PIN" ] && { log "$PIN already exists; pick a new pin directory"; exit 1; }
stop_all() {
  for c in config/cluster.json $E/cluster-*.json experiments/2026-09-29-determinism/cluster-*.json \
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
  docker logs dsv41-karmic-kraken 2>&1 | grep -E "pinned DSpark cost curves|Pinned DSpark cost curves" | tail -1 | tee -a "$out/pin.log"
  sha256sum $PIN/*.json | tee -a "$out/pin.log"
}
prefill() {
  python3 experiments/2026-09-29-indexer-split/capture_depth.py http://10.0.1.71:8000 --rounds 3 \
    | tee "$out/depth-$1.jsonl"
  log "prefill $1 exit ${PIPESTATUS[0]}"
}
decode() {
  bin/spark3 --cluster-config "$1" bench --allow-mismatch --compare none --suites decode \
    --decode-cases prose,json-nothink --concurrency 1,8 --min-samples 6 --max-samples 6 \
    --output "results/private/bench/r5o-$2"
  log "decode $2 exit $?"
}
for arm in detm detm-fence; do
  start $E/cluster-$arm.json
  python3 $E/determinism_long.py http://10.0.1.71:8000 | tee "$out/long-$arm.jsonl"
  log "long probe $arm exit ${PIPESTATUS[0]}"
done
start $E/cluster-r5n-pin.json
pinned
prefill r5n
decode $E/cluster-r5n-pin.json r5n-1
start $E/cluster-overlay-pin.json
pinned
prefill overlay
decode $E/cluster-overlay-pin.json overlay-1
start $E/cluster-r5n-pin.json
pinned
decode $E/cluster-r5n-pin.json r5n-2
start $E/cluster-overlay-pin.json
pinned
decode $E/cluster-overlay-pin.json overlay-2
start config/cluster.json
log "done"
