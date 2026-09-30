#!/bin/bash
# usage: run9.sh   (on dgx1, deployment checkout at this experiment's commit, after overlay.sh)
# Profiles one single-stream decode request on detslice and lists the kernels
# that overlap the side-stream shared-expert kernels. Restores r5m.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism
log() { echo "$(date -u +%FT%TZ) $*"; }
stop_all() {
  for c in config/cluster.json $E/cluster-*.json; do
    bin/spark3 --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
  done
}
rm -rf cache/kkref/profiles/det-detslice-prof 2>/dev/null
stop_all
log "start detslice-prof"
bin/spark3 --cluster-config $E/cluster-detslice-prof.json cluster start --replace --apply | grep -v 'docker run'
python3 $E/profile_decode.py http://10.0.1.71:8000 | tee "$out/profile-detslice-prof.json"
log "profile exit ${PIPESTATUS[0]}"
sleep 60
t=$(ls cache/kkref/profiles/det-detslice-prof/*rank0*.gz | head -1)
docker exec -e CUDA_VISIBLE_DEVICES= dsv41-karmic-kraken true 2>/dev/null
python3 $E/analyze_overlap.py "$t" | tee "$out/overlap-detslice.txt" || true
stop_all
bin/spark3 cluster start --replace --apply | grep -v 'docker run'
log "done"
