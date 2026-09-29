#!/bin/bash
# usage: run.sh   (on dgx1, deployment checkout at this experiment's commit, after overlay.sh)
# 1. ties-check: capture_depth.py with four background decode streams under the
#    split check; check_summary.py --exact requires zero differences.
# 2. ties: capture_depth.py prefill timings (two rounds).
# 3. Decode, alternating r5k, r5l, ties twice: JSON answers and prose at one and
#    eight streams, six samples each. Leaves r5k running.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-topk-ties
L=experiments/2026-09-29-r5l
out=results/private/topk-ties
mkdir -p "$out"
log() { echo "$(date -u +%FT%TZ) $*"; }
stop_all() {
  for c in config/cluster.json $L/cluster-*.json $E/cluster-*.json; do
    bin/spark3 --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
  done
}
start() {
  stop_all
  log "start $1"
  bin/spark3 --cluster-config "$1" cluster start --replace --apply | grep -v 'docker run'
}
start $E/cluster-ties-check.json
python3 experiments/2026-09-29-indexer-split/capture_depth.py http://10.0.1.71:8000 --rounds 1 --background 4 \
  | tee "$out/check-depth.jsonl"
log "check capture exit ${PIPESTATUS[0]}"
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" "docker logs dsv41-karmic-kraken 2>&1 | grep 'indexer split check' > /tmp/ties-check-\$(hostname).log; true"
  scp -q "$n:/tmp/ties-check-$n.log" "$out/check-$n.log"
done
python3 $L/check_summary.py --exact "$out"/check-dgx*.log | tee "$out/check-summary.txt"
log "check summary exit ${PIPESTATUS[0]}"
start $E/cluster-ties.json
python3 experiments/2026-09-29-indexer-split/capture_depth.py http://10.0.1.71:8000 --rounds 2 \
  | tee "$out/ties-depth.jsonl"
log "prefill capture exit ${PIPESTATUS[0]}"
decode() {
  bin/spark3 --cluster-config "$1" bench --allow-mismatch --compare none --suites decode \
    --decode-cases prose,json-nothink --concurrency 1,8 --min-samples 6 --max-samples 6 \
    --output "results/private/bench/ties-$2"
  log "decode $2 exit $?"
}
decode $E/cluster-ties.json ties-1
for round in 1 2; do
  start config/cluster.json
  decode config/cluster.json r5k-$round
  start $L/cluster-candidate.json
  decode $L/cluster-candidate.json r5l-$round
  if [ "$round" = 1 ]; then
    start $E/cluster-ties.json
    decode $E/cluster-ties.json ties-2
  fi
done
start config/cluster.json
log "done"
