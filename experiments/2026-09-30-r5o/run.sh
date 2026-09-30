#!/bin/bash
# usage: run.sh   (on dgx1, deployment checkout at this experiment's commit, after build.sh)
# Boots the r5o candidate, confirms every node's three 0005 files are the exact
# files measured as an overlay by screen.sh, requires the compiled BF16 prefill
# and mHC TF32 projections (and any contiguous-attention build) to have no shared
# loads pending at a stage release (sass_gate.sh), then runs the quality gate,
# the needle check and doctor --live. The candidate stays up.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-30-r5o
C=$E/cluster-candidate.json
out=results/private/r5o
mkdir -p "$out"
log() { echo "$(date -u +%FT%TZ) $*"; }
declare -A MEASURED=(
  [b12x/attention/_shared/contiguous/forward.py]=94e95478f0d2d2f20c2d61af5ceb2c8ffe08286ee7d3bab8695774e6f8e9fb2e
  [b12x/gemm/bf16_gemv/_prefill.py]=969fb9b20dcdb5ad9cf92674c687a4b55d7319908adede9928236cd34ead0bd1
  [b12x/norm/mhc/_kernels.py]=fe5b00e967f80dbffe751bb6e140a65842540c6ea664e1b1ce0e9f0ea21908bb
  [b12x/_lib/dense_gemm.py]=ffc7c4a491621018ab54982878ebc5334e5aa31bea86194a8772c1e4af3bb9a8
)
for c in config/cluster.json $E/cluster-*.json; do
  bin/spark3 --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
done
log "start candidate"
bin/spark3 --cluster-config $C cluster start --replace --apply | grep -v 'docker run'
bad=0
for n in dgx1 dgx2 dgx3; do
  for f in "${!MEASURED[@]}"; do
    got=$(ssh -n "$n" docker exec dsv41-karmic-kraken sha256sum /opt/spark3/candidate/b12x/$f | cut -d" " -f1)
    echo "$n $f $got" | tee -a "$out/file-bytes.txt"
    [ "$got" = "${MEASURED[$f]}" ] || bad=1
  done
done
if [ "$bad" != 0 ]; then log "a shipped file differs from the measured one"; exit 1; fi
log "shipped files match the measured ones on every node"
bash $E/sass_gate.sh --expect-fenced > "$out/sass-gate.txt" 2>&1
log "sass gate exit $?"
tail -1 "$out/sass-gate.txt"
bin/spark3 --cluster-config $C bench --allow-mismatch --compare none --suites quality \
  --output results/private/bench/r5o-quality
log "quality exit $?"
python3 experiments/2026-09-29-r5k/needle.py http://10.0.1.71:8000 180000 | tee "$out/needle.txt"
log "needle exit ${PIPESTATUS[0]}"
bin/spark3 --cluster-config $C doctor --live | tee "$out/doctor.txt"
log "doctor exit ${PIPESTATUS[0]}"
for n in dgx1 dgx2 dgx3; do
  ssh -n "$n" "docker logs dsv41-karmic-kraken 2>&1 | grep -o 'Display carve-out holds [0-9.]* MiB' | head -1 | sed \"s/^/\$(hostname): /\""
done | tee "$out/carveout.txt"
log "done"
