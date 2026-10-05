#!/bin/bash
# usage: run.sh   (on dgx1, deployment checkout at this experiment's commit, after build.sh)
# Boots the r5m candidate, confirms every node's tiled_topk.py is the exact file
# measured as an overlay in experiments/2026-09-29-topk-ties, then runs the
# quality gate, the needle check and doctor --live. The candidate stays up.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-r5m
C=$E/cluster-candidate.json
out=results/private/r5m
mkdir -p "$out"
log() { echo "$(date -u +%FT%TZ) $*"; }
MEASURED=f1bdacb57aab25fde6156d6c27d4181ae6c7d22ced1bfc3ab538703eafb6a8aa
for c in config/cluster.json $C; do
  bin/spark --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
done
log "start candidate"
bin/spark --cluster-config $C cluster start --replace --apply | grep -v 'docker run'
bad=0
for n in dgx1 dgx2 dgx3; do
  got=$(ssh -n "$n" docker exec dsv41-karmic-kraken sha256sum \
    /opt/spark3/candidate/b12x/b12x/attention/dsa_indexer/tiled_topk.py | cut -d" " -f1)
  echo "$n tiled_topk.py $got" | tee -a "$out/topk-bytes.txt"
  [ "$got" = "$MEASURED" ] || bad=1
done
if [ "$bad" != 0 ]; then log "tiled_topk.py differs from the measured overlay"; exit 1; fi
log "tiled_topk.py matches the measured overlay on every node"
bin/spark --cluster-config $C bench --allow-mismatch --compare none --suites quality \
  --output results/private/bench/r5m-quality
log "quality exit $?"
python3 experiments/2026-09-29-r5k/needle.py http://10.0.1.71:8000 180000 | tee "$out/needle.txt"
log "needle exit ${PIPESTATUS[0]}"
bin/spark --cluster-config $C doctor --live | tee "$out/doctor.txt"
log "doctor exit ${PIPESTATUS[0]}"
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" "docker logs dsv41-karmic-kraken 2>&1 | grep -o 'Display carve-out holds [0-9.]* MiB' | head -1 | sed \"s/^/\$(hostname): /\""
done | tee "$out/carveout.txt"
log "done"
