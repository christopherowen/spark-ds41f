#!/bin/bash
# usage: run7.sh   (on dgx1, deployment checkout at this experiment's commit, after overlay.sh
#                  and the moe-checksum overlay)
# Boots detslice-dbg (debug MoE checksum log), sends one request twice, dumps
# every rank's log, copies them to results, then restores r5m.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism/checksums
mkdir -p "$out"
log() { echo "$(date -u +%FT%TZ) $*"; }
stop_all() {
  for c in config/cluster.json $E/cluster-*.json; do
    bin/spark3 --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
  done
}
stop_all
log "start detslice-dbg"
bin/spark3 --cluster-config $E/cluster-detslice-dbg.json cluster start --replace --apply | grep -v 'docker run'
python3 $E/checksum_requests.py http://10.0.1.71:8000 --tokens 128 | tee "$out/requests.json"
log "requests exit ${PIPESTATUS[0]}"
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" "docker exec dsv41-karmic-kraken sh -c 'rm -f /cache/kkref/moe-checksums/rank*.pt; touch /cache/kkref/moe-checksums/dump'"
done
sleep 15
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" "docker exec dsv41-karmic-kraken ls /cache/kkref/moe-checksums/"
  ssh -n "$n" "docker exec dsv41-karmic-kraken sh -c 'cat /cache/kkref/moe-checksums/rank*.pt'" > "$out/$n.pt"
done
ls -la "$out"
stop_all
bin/spark3 cluster start --replace --apply | grep -v 'docker run'
log "done"
