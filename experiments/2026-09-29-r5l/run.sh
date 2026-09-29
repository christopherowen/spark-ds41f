#!/bin/bash
# usage: run.sh   (on dgx1, deployment checkout at this experiment's commit, after build.sh)
# 1. candidate-check: the carve-out unit tests inside the running image, then
#    capture_depth.py with four background decode streams. Every rank compares
#    its split indexer rows with the full-row indexer; any differing row, or a
#    node without a passed carve-out check, stops the run here.
# 2. candidate: the reference bench, real-text prefill to 200K and the needle
#    check. The candidate stays up.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-r5l
out=results/private/r5l
mkdir -p "$out"
log() { echo "$(date -u +%FT%TZ) $*"; }
stop_all() {
  for c in config/cluster.json $E/cluster-*.json experiments/2026-09-29-indexer-split/cluster-*.json \
      experiments/2026-09-29-r5k/cluster-*.json; do
    bin/spark3 --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
  done
}
start() {
  stop_all
  log "start $1"
  bin/spark3 --cluster-config "$E/cluster-$1.json" cluster start --replace --apply | grep -v 'docker run'
}
start candidate-check
docker cp -q "$HOME/projects/spark3-vllm-ds41f/.work/upstreams/vllm/tests/v1/worker/test_display_carveout.py" \
  dsv41-karmic-kraken:/tmp/test_display_carveout.py 2>/dev/null
docker exec -w /tmp dsv41-karmic-kraken python3 -m pytest -q -p no:cacheprovider /tmp/test_display_carveout.py 2>&1 \
  | tail -3 | tee "$out/carveout-tests.txt"
python3 experiments/2026-09-29-indexer-split/capture_depth.py http://10.0.1.71:8000 --rounds 1 --background 4 \
  | tee "$out/check-depth.jsonl"
log "check capture exit ${PIPESTATUS[0]}"
sleep 70  # at least one more carve-out check on every node
bad=0
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" "docker logs dsv41-karmic-kraken 2>&1 | grep -E 'indexer split check|Display carve-out' > /tmp/r5l-check-\$(hostname).log; true"
  scp -q "$n:/tmp/r5l-check-$n.log" "$out/check-$n.log"
  match=$(grep -c 'match the full' "$out/check-$n.log")
  differ=$(grep -c 'differ from the full' "$out/check-$n.log")
  carve=$(grep -c 'Display carve-out integrity check .* passed' "$out/check-$n.log")
  echo "$n: $match matching reports, $differ differing reports, $carve carve-out checks passed" | tee -a "$out/check-summary.txt"
  if [ "$differ" != 0 ] || [ "$carve" = 0 ]; then bad=1; fi
done
if [ "$bad" != 0 ]; then log "check failed; stopping before the candidate"; exit 1; fi
start candidate
C=$E/cluster-candidate.json
bin/spark3 --cluster-config $C doctor --live || true
bin/spark3 --cluster-config $C bench --allow-mismatch --compare none \
  --suites quality,decode,prefill,prefix,admission \
  --decode-cases prose,code,prose-nothink,code-nothink,json-nothink \
  --output results/private/bench/r5l-reference
log "bench exit $?"
bin/spark3 --cluster-config $C bench --allow-mismatch --compare none --suites prefill \
  --prefill-text source --prefill-sizes 4096,16384,32768,65536,131072,200000 --prefill-repeats 2 \
  --output results/private/bench/r5l-prefill-source
log "real-text prefill exit $?"
python3 experiments/2026-09-29-r5k/needle.py http://10.0.1.71:8000 180000 | tee "$out/needle.txt"
log "needle exit ${PIPESTATUS[0]}"
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" "docker logs dsv41-karmic-kraken 2>&1 | grep -E 'Display carve-out' | tail -2 | cut -c1-200"
done
log "done"
