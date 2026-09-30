#!/bin/bash
# usage: run7.sh [ARM]   (on dgx1, deployment checkout at this experiment's commit, after
#                        overlay.sh and the moe-checksum overlay; ARM defaults to detslice-dbg)
# Boots detslice-dbg (debug MoE checksum log), sends one request twice, dumps
# every rank's log, copies them to results, then restores r5m.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
ARM=${1:-detslice-dbg}
out=results/private/determinism/checksums-$ARM
mkdir -p "$out"
log() { echo "$(date -u +%FT%TZ) $*"; }
stop_all() {
  for c in config/cluster.json $E/cluster-*.json; do
    bin/spark3 --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
  done
}
stop_all
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" "rm -f ~/projects/spark3-vllm-ds41f/cache/kkref/moe-checksums/dump ~/projects/spark3-vllm-ds41f/cache/kkref/moe-checksums/reset ~/projects/spark3-vllm-ds41f/cache/kkref/moe-checksums/rank*.pt 2>/dev/null; true"
done
log "start $ARM"
bin/spark3 --cluster-config $E/cluster-$ARM.json cluster start --replace --apply | grep -v 'docker run'
dump() {
  for n in dgx1 dgx2 dgx3; do
    ssh -n "$n" "docker exec dsv41-karmic-kraken sh -c 'mkdir -p /cache/kkref/moe-checksums; touch /cache/kkref/moe-checksums/dump'"
  done
  sleep 8
}
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" "docker exec dsv41-karmic-kraken sh -c 'rm -f /cache/kkref/moe-checksums/rank*.pt'"
done
python3 $E/checksum_requests.py http://10.0.1.71:8000 --tokens 128 --once > "$out/request1.json"
log "request 1 exit $?"
dump
python3 $E/checksum_requests.py http://10.0.1.71:8000 --tokens 128 --once > "$out/request2.json"
log "request 2 exit $?"
dump
python3 -c "import json,sys; a,b=(json.load(open(f))['text'] for f in sys.argv[1:]); print('identical', a==b)" "$out/request1.json" "$out/request2.json"
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" "docker exec dsv41-karmic-kraken ls /cache/kkref/moe-checksums/"
  mkdir -p "$out/$n"
  for f in $(ssh -n "$n" "docker exec dsv41-karmic-kraken sh -c 'cd /cache/kkref/moe-checksums && ls rank*.pt'"); do
    ssh -n "$n" "docker exec dsv41-karmic-kraken cat /cache/kkref/moe-checksums/$f" > "$out/$n/$f"
  done
done
ls -la "$out"
stop_all
bin/spark3 cluster start --replace --apply | grep -v 'docker run'
log "done"
