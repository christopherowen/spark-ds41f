#!/bin/bash
# usage: run.sh   (on dgx1, deployment checkout at this experiment's commit, after build.sh)
# Boots the r5n candidate, confirms every node's dense_gemm.py is the exact file
# measured as an overlay by screen.sh, then runs the quality gate, the needle
# check and doctor --live. The candidate stays up.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-30-r5n
C=$E/cluster-candidate.json
out=results/private/r5n
mkdir -p "$out"
log() { echo "$(date -u +%FT%TZ) $*"; }
MEASURED=ffc7c4a491621018ab54982878ebc5334e5aa31bea86194a8772c1e4af3bb9a8
for c in config/cluster.json $E/cluster-*.json; do
  bin/spark --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
done
log "start candidate"
bin/spark --cluster-config $C cluster start --replace --apply | grep -v 'docker run'
bad=0
for n in dgx1 dgx2 dgx3; do
  got=$(ssh -n "$n" docker exec dsv41-karmic-kraken sha256sum \
    /opt/spark3/candidate/b12x/b12x/_lib/dense_gemm.py | cut -d" " -f1)
  echo "$n dense_gemm.py $got" | tee -a "$out/gemm-bytes.txt"
  [ "$got" = "$MEASURED" ] || bad=1
done
if [ "$bad" != 0 ]; then log "dense_gemm.py differs from the measured overlay"; exit 1; fi
log "dense_gemm.py matches the measured overlay on every node"
bin/spark --cluster-config $C bench --allow-mismatch --compare none --suites quality \
  --output results/private/bench/r5n-quality
log "quality exit $?"
python3 experiments/2026-09-29-r5k/needle.py http://10.0.1.71:8000 180000 | tee "$out/needle.txt"
log "needle exit ${PIPESTATUS[0]}"
bin/spark --cluster-config $C doctor --live | tee "$out/doctor.txt"
log "doctor exit ${PIPESTATUS[0]}"
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" "docker logs dsv41-karmic-kraken 2>&1 | grep -o 'Display carve-out holds [0-9.]* MiB' | head -1 | sed \"s/^/\$(hostname): /\""
done | tee "$out/carveout.txt"
log "done"
