#!/bin/bash
# usage: [REC=rec4] run65.sh LABEL...   (on dgx1, deployment checkout at this experiment's commit, config = r5o, with the
#                    det-variant2, ref4d, ref4e, gemv-geom2 and mhc-mt2 overlays on every node)
# The mHC screen: without tracing, one boot per arm, pinned cost table, r5o-pin and then detm-r5o-LABEL-pin for
# each label (ref4d-b4144: the native lagged route with 16 rows per CTA at capacity; ref4e-s16-b4144 and
# ref4e-s40-b4144: the TF32 TMA projection at every capacity with 16 or 40 K slices): single-stream decode over
# 24 distinct prompts and the bench's prose and JSON streams, cold prefill 1K-64K, eight distinct prompts,
# short-prompt and mixed-traffic latency. Restores r5o.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
REC=${REC:-rec4}
out=results/private/determinism/$REC
PIN=cache/kkref/dspark-costs/r5o-pin-20260930
IMAGE=vllm-ds41f-kkref:04c30fa98e79-r5o
V=/opt/spark3/candidate/vllm
W=$HOME/work/vllm-gemv
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
  return "${PIPESTATUS[0]}"
}
LOGDIR=/cache/kkref/moe-checksums
fresh_inventory() {
  for n in dgx1 dgx2 dgx3; do
    ssh -n "$n" "docker exec dsv41-karmic-kraken sh -c 'rm -f $LOGDIR/inventory-* $LOGDIR/plans-*'"
  done
}
mkdir -p "$out"
git fetch -q origin
git merge-base --is-ancestor HEAD origin/main || { log "deployment commit not on origin/main; not stopping"; exit 1; }
for n in dgx2 dgx3; do
  [ "$(ssh -n $n git -C projects/spark3-vllm-ds41f rev-parse HEAD)" = "$(git rev-parse HEAD)" ] \
    || { log "$n checkout differs from dgx1; not stopping"; exit 1; }
done
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" 'cd spark3-overlay && find det-variant2 ref4d ref4e gemv-geom2 mhc-mt2 -name "*.py" | sort | xargs sha256sum' \
    | sed "s/^/$n /" >> results/private/determinism/overlay-bytes-run65.txt
done
curl -s http://10.0.1.71:8000/metrics | grep -E '^vllm:num_requests_running' || true
stop_all

mout=results/private/determinism/$REC
mkdir -p "$mout"
measure() {  # arm label
  start $E/cluster-$1.json || { log "start $1 failed"; return 1; }
  docker logs dsv41-karmic-kraken 2>&1 | grep -E "pinned DSpark cost curves|Pinned DSpark cost curves" | tail -1 \
    | tee -a "$mout/pin.log"
  sha256sum $PIN/*.json | tee -a "$mout/pin.log"
  bin/spark --cluster-config $E/cluster-$1.json bench --allow-mismatch --compare none --suites decode,prefill \
    --decode-cases prose,json-nothink --concurrency 1 --min-samples 5 --max-samples 5 \
    --prefill-text source --prefill-sizes 1024,4096,16384,65536 --prefill-repeats 3 \
    --output "results/private/bench/$REC-$2"
  log "bench $2 exit $?"
  python3 $E/c1_distinct.py http://10.0.1.71:8000 --tokens 256 | tee "$mout/c1-distinct-$2.jsonl"
  python3 $E/c8_distinct.py http://10.0.1.71:8000 --samples 3 --tokens 256 | tee "$mout/c8-distinct-$2.jsonl"
  python3 $E/ttft_short.py http://10.0.1.71:8000 | tee "$mout/ttft-$2.jsonl"
  python3 $E/mixed_latency.py http://10.0.1.71:8000 --rounds 3 | tee "$mout/mixed-$2.jsonl"
  log "extra $2 exit $?"
}
measure r5o-pin r5o
for label in "$@"; do
  measure detm-r5o-$label-pin $label
done
stop_all
bin/spark cluster start --replace --apply | grep -v 'docker run'
bin/spark doctor --live 2>&1 | tail -3
log "done"
