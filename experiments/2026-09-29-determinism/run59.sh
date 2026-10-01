#!/bin/bash
# usage: run59.sh   (on dgx1, deployment checkout at this experiment's commit, config = r5o, with the
#                    det-variant2 and ref3a/ref3b/ref3c overlays on every node)
# Cost recovery measured step by step, without tracing, one boot per arm, pinned cost table: r5o-pin,
# detm-r5o-ref3a-pin (ref2 + vllm-0039), detm-r5o-ref3b-pin (+ vllm-0040), detm-r5o-ref3c-pin
# (+ vllm-0041): single-stream decode over twelve distinct prompts (c1_distinct.py) and the bench's
# prose and JSON streams (five samples), cold prefill of real text (1024-65536 tokens, three
# repeats), eight distinct concurrent prompts, short-prompt latency and mixed-traffic latency.
# Restores r5o.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism/rec1
PIN=cache/kkref/dspark-costs/r5o-pin-20260930
IMAGE=vllm-ds41f-kkref:04c30fa98e79-r5o
V=/opt/spark3/candidate/vllm
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
mkdir -p "$out"
git fetch -q origin
git merge-base --is-ancestor HEAD origin/main || { log "deployment commit not on origin/main; not stopping"; exit 1; }
for n in dgx2 dgx3; do
  [ "$(ssh -n $n git -C projects/spark3-vllm-ds41f rev-parse HEAD)" = "$(git rev-parse HEAD)" ] \
    || { log "$n checkout differs from dgx1; not stopping"; exit 1; }
done
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" 'cd spark3-overlay && find det-variant2 ref3a ref3b ref3c -name "*.py" | sort | xargs sha256sum' \
    | sed "s/^/$n /" >> "$out/overlay-bytes-run59.txt"
done
curl -s http://10.0.1.71:8000/metrics | grep -E '^vllm:num_requests_running' || true
measure() {  # arm label
  start $E/cluster-$1.json
  docker logs dsv41-karmic-kraken 2>&1 | grep -E "pinned DSpark cost curves|Pinned DSpark cost curves" | tail -1 \
    | tee -a "$out/pin.log"
  sha256sum $PIN/*.json | tee -a "$out/pin.log"
  bin/spark3 --cluster-config $E/cluster-$1.json bench --allow-mismatch --compare none --suites decode,prefill \
    --decode-cases prose,json-nothink --concurrency 1 --min-samples 5 --max-samples 5 \
    --prefill-text source --prefill-sizes 1024,4096,16384,65536 --prefill-repeats 3 \
    --output "results/private/bench/rec1-$2"
  log "bench $2 exit $?"
  python3 $E/c1_distinct.py http://10.0.1.71:8000 --tokens 256 | tee "$out/c1-distinct-$2.jsonl"
  python3 $E/c8_distinct.py http://10.0.1.71:8000 --samples 3 --tokens 256 | tee "$out/c8-distinct-$2.jsonl"
  python3 $E/ttft_short.py http://10.0.1.71:8000 | tee "$out/ttft-$2.jsonl"
  python3 $E/mixed_latency.py http://10.0.1.71:8000 --rounds 3 | tee "$out/mixed-$2.jsonl"
  log "extra $2 exit $?"
}
measure r5o-pin r5o
measure detm-r5o-ref3a-pin ref3a
measure detm-r5o-ref3b-pin ref3b
measure detm-r5o-ref3c-pin ref3c
stop_all
bin/spark3 cluster start --replace --apply | grep -v 'docker run'
bin/spark3 doctor --live 2>&1 | tail -3
log "done"
