#!/bin/bash
# usage: run3.sh   (on dgx1, deployment checkout at this experiment's commit, after overlay.sh)
# Profiles one single-stream decode request on r5m and on detfast, then
# compares rank 0's kernel time. Traces land in cache/kkref/profiles/det-*/.
# Leaves r5m running.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism
mkdir -p "$out"
log() { echo "$(date -u +%FT%TZ) $*"; }
stop_all() {
  for c in config/cluster.json $E/cluster-*.json; do
    bin/spark --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
  done
}
start() {
  stop_all
  log "start $1"
  bin/spark --cluster-config "$1" cluster start --replace --apply | grep -v 'docker run'
}
for arm in r5m-prof detfast-prof; do
  rm -rf cache/kkref/profiles/det-$arm
  start $E/cluster-$arm.json
  python3 $E/profile_decode.py http://10.0.1.71:8000 | tee "$out/profile-$arm.json"
  log "profile $arm exit ${PIPESTATUS[0]}"
  sleep 60  # let every rank finish writing its trace
  ls -la cache/kkref/profiles/det-$arm/ | tail -n +2
done
start config/cluster.json
b=$(ls cache/kkref/profiles/det-r5m-prof/*rank0*.gz cache/kkref/profiles/det-r5m-prof/*.gz 2>/dev/null | head -1)
c=$(ls cache/kkref/profiles/det-detfast-prof/*rank0*.gz cache/kkref/profiles/det-detfast-prof/*.gz 2>/dev/null | head -1)
python3 $E/analyze_decode.py "$b" "$c" --top 40 | tee "$out/profile-compare.txt"
log "done"
