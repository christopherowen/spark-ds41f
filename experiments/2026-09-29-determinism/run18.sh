#!/bin/bash
# usage: run18.sh   (on dgx1, deployment checkout at this experiment's commit, after
#                    overlay.sh and overlay_fence.sh)
# 0007 in serving, side-stream shared-expert overlap on. Boots detslice-fence
# (0004-0007, deterministic MoE, split-K through the FP32 reducer): determinism
# probe with saved tokens, then a lean decode screen (JSON answers and prose at
# one and eight streams, three samples). Then the same screen on r5m-fence
# (production plus 0007) and r5m. Leaves r5m running.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism
mkdir -p "$out"
log() { echo "$(date -u +%FT%TZ) $*"; }
stop_all() {
  for c in config/cluster.json $E/cluster-*.json; do
    bin/spark3 --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
  done
}
start() {
  stop_all
  log "start $1"
  bin/spark3 --cluster-config "$1" cluster start --replace --apply | grep -v 'docker run'
}
decode() {
  bin/spark3 --cluster-config "$1" bench --allow-mismatch --compare none --suites decode \
    --decode-cases prose,json-nothink --concurrency 1,8 --min-samples 3 --max-samples 3 \
    --output "results/private/bench/determinism-$2"
  log "decode $2 exit $?"
}
start $E/cluster-detslice-fence.json
python3 $E/determinism.py http://10.0.1.71:8000 --repeats 5 --tokens 256 \
  --save "$out/tokens-detslice-fence.json" | tee "$out/probe-detslice-fence.jsonl"
log "probe detslice-fence exit ${PIPESTATUS[0]}"
decode $E/cluster-detslice-fence.json detslice-fence
start $E/cluster-r5m-fence.json
decode $E/cluster-r5m-fence.json r5m-fence
start config/cluster.json
decode config/cluster.json r5m-d
log "done"
