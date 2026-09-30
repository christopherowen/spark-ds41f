#!/bin/bash
# usage: screen.sh   (on dgx1, deployment checkout at this experiment's commit, after
#                     experiments/2026-09-29-determinism/overlay_fence.sh)
# Measures the file 0004 ships as an overlay on r5m before building: requires
# every node's overlay dense_gemm.py to be the measured one, then prefill chunk
# timings (two rounds) on the overlay and on r5m, and decode alternating
# overlay, r5m, overlay, r5m (JSON answers and prose at one and eight streams,
# six samples each). Leaves r5m running.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-30-r5n
D=experiments/2026-09-29-determinism
out=results/private/r5n
mkdir -p "$out"
log() { echo "$(date -u +%FT%TZ) $*"; }
MEASURED=ffc7c4a491621018ab54982878ebc5334e5aa31bea86194a8772c1e4af3bb9a8
for n in dgx1 dgx2 dgx3; do
  got=$(ssh -n "$n" sha256sum spark3-overlay/gemm-fence/b12x/_lib/dense_gemm.py | cut -d" " -f1)
  echo "$n overlay dense_gemm.py $got" | tee -a "$out/overlay-bytes.txt"
  [ "$got" = "$MEASURED" ] || { log "overlay differs on $n"; exit 1; }
done
stop_all() {
  for c in config/cluster.json $E/cluster-*.json $D/cluster-*.json; do
    bin/spark3 --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
  done
}
start() {
  stop_all
  log "start $1"
  bin/spark3 --cluster-config "$1" cluster start --replace --apply | grep -v 'docker run'
}
prefill() {
  python3 experiments/2026-09-29-indexer-split/capture_depth.py http://10.0.1.71:8000 --rounds 2 \
    | tee "$out/depth-$1.jsonl"
  log "prefill $1 exit ${PIPESTATUS[0]}"
}
decode() {
  bin/spark3 --cluster-config "$1" bench --allow-mismatch --compare none --suites decode \
    --decode-cases prose,json-nothink --concurrency 1,8 --min-samples 6 --max-samples 6 \
    --output "results/private/bench/r5n-$2"
  log "decode $2 exit $?"
}
start $E/cluster-overlay.json
prefill overlay
decode $E/cluster-overlay.json overlay-1
start config/cluster.json
prefill r5m
decode config/cluster.json r5m-1
start $E/cluster-overlay.json
decode $E/cluster-overlay.json overlay-2
start config/cluster.json
decode config/cluster.json r5m-2
log "done"
