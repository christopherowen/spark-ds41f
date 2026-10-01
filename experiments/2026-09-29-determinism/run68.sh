#!/bin/bash
# usage: run68.sh   (on dgx1, deployment checkout at this experiment's commit, config = r5o, with the
#                    det-variant2, ref4d, ref4e, gemv-geom2 and mhc-mt2 overlays on every node)
# Kernel timings of the two mHC options against r5o (run55's method): r5o-pin-prof,
# detm-r5o-ref4d-b4144-pin-prof (native mHC) and detm-r5o-ref4e-s40-b4144-pin-prof (TF32 mHC, 40 K
# slices), one boot each, the torch profiler around profile_decode.py (one JSON stream, thinking off),
# profile_c8.py (eight distinct JSON prompts) and profile_prefill.py (one cold 16384-token prompt).
# summarize_kernels.py reduces rank 0's traces and analyze_costs.py compares each candidate with r5o while
# the cluster is stopped. Restores r5o.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism/costs2
IMAGE=vllm-ds41f-kkref:04c30fa98e79-r5o
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
ARMS="r5o-pin-prof detm-r5o-ref4d-b4144-pin-prof detm-r5o-ref4e-s40-b4144-pin-prof"
for arm in $ARMS; do profile $arm; done
stop_all
for arm in $ARMS; do
  for w in decode c8 prefill; do
    t=$(ls "$out/$arm/$w-trace/"*.json.gz | head -1)
    docker run --rm --memory=16g -e CUDA_VISIBLE_DEVICES= -v $PWD/$E/summarize_kernels.py:/s.py:ro \
      -v $PWD/$out:/o --entrypoint python3 $IMAGE /s.py /o/$arm/$w-kernels.json "/o/$arm/$w-trace/$(basename $t)"
  done
done
log "summarized"
for arm in detm-r5o-ref4d-b4144-pin-prof detm-r5o-ref4e-s40-b4144-pin-prof; do
  for w in decode c8; do
    docker run --rm --memory=16g -e CUDA_VISIBLE_DEVICES= -v $PWD/$E/analyze_costs.py:/a.py:ro -v $PWD/$out:/o \
      --entrypoint python3 $IMAGE /a.py /o/r5o-pin-prof/$w-kernels.json /o/$arm/$w-kernels.json --top 25 \
      > "$out/costs-$arm-$w.txt" 2>&1
  done
  docker run --rm --memory=16g -e CUDA_VISIBLE_DEVICES= -v $PWD/$E/analyze_costs.py:/a.py:ro -v $PWD/$out:/o \
    --entrypoint python3 $IMAGE /a.py /o/r5o-pin-prof/prefill-kernels.json /o/$arm/prefill-kernels.json --total --top 25 \
    > "$out/costs-$arm-prefill.txt" 2>&1
done
log "costs"
bin/spark3 cluster start --replace --apply | grep -v 'docker run'
bin/spark3 doctor --live 2>&1 | tail -3
log "done"
