#!/bin/bash
# usage: run55.sh   (on dgx1, deployment checkout at this experiment's commit, config = r5o, with the
#                    det-variant2 and ref2 overlays on every node)
# Cost attribution by kernel timings of captured workloads (no leave-one-out throughput): r5o-pin-prof
# and detm-r5o-ref2-pin-prof (the frozen reference), one boot each, the torch profiler around
# profile_decode.py (one JSON stream, thinking off), profile_c8.py (eight distinct JSON prompts at
# once, with verification counters) and profile_prefill.py (one cold 16384-token prompt).
# summarize_kernels.py reduces rank 0's traces while the cluster is stopped. Restores r5o.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism/costs
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
settle() {  # profile dir: wait until rank 0's newest trace exists and stops growing
  local dir=$1 before=-1 now
  for _ in $(seq 120); do
    now=$(cat "$dir"/*rank0*.json.gz 2>/dev/null | wc -c)
    [ "$now" -gt 0 ] && [ "$now" = "$before" ] && return 0
    before=$now
    sleep 5
  done
  log "trace in $dir did not settle"
}
mkdir -p "$out"
git fetch -q origin
git merge-base --is-ancestor HEAD origin/main || { log "deployment commit not on origin/main; not stopping"; exit 1; }
for n in dgx2 dgx3; do
  [ "$(ssh -n $n git -C projects/spark3-vllm-ds41f rev-parse HEAD)" = "$(git rev-parse HEAD)" ] \
    || { log "$n checkout differs from dgx1; not stopping"; exit 1; }
done
curl -s http://10.0.1.71:8000/metrics | grep -E '^vllm:num_requests_running' || true
profile() {  # arm (the profile directory is the container's, root-owned: clear and move inside it)
  local P=/cache/kkref/profiles/det-$1 H=cache/kkref/profiles/det-$1
  start $E/cluster-$1.json
  mkdir -p "$out/$1"
  for w in decode c8 prefill; do
    docker exec dsv41-karmic-kraken sh -c "mkdir -p $P && rm -rf $P/*.json.gz $P/$w"
    case $w in
      decode) python3 $E/profile_decode.py http://10.0.1.71:8000 --tokens 128 ;;
      c8) python3 $E/profile_c8.py http://10.0.1.71:8000 --case json --streams 8 --tokens 128 ;;
      prefill) python3 $E/profile_prefill.py http://10.0.1.71:8000 --tokens 16384 ;;
    esac > "$out/$1/$w.json"
    log "$1 $w exit $?"
    settle $H
    docker exec dsv41-karmic-kraken sh -c "mkdir -p $P/$w && mv $P/*rank0*.json.gz $P/$w/"
    mkdir -p "$out/$1/$w-trace" && cp $H/$w/*.json.gz "$out/$1/$w-trace/"
  done
}
profile r5o-pin-prof
profile detm-r5o-ref2-pin-prof
stop_all
for arm in r5o-pin-prof detm-r5o-ref2-pin-prof; do
  for w in decode c8 prefill; do
    t=$(ls "$out/$arm/$w-trace/"*.json.gz | head -1)
    docker run --rm --memory=16g -e CUDA_VISIBLE_DEVICES= -v $PWD/$E/summarize_kernels.py:/s.py:ro \
      -v $PWD/$out:/o --entrypoint python3 $IMAGE /s.py /o/$arm/$w-kernels.json "/o/$arm/$w-trace/$(basename $t)"
  done
done
log "summarized"
bin/spark cluster start --replace --apply | grep -v 'docker run'
bin/spark doctor --live 2>&1 | tail -3
log "done"
