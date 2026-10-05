#!/bin/bash
# usage: run17.sh   (on dgx1, deployment checkout at this experiment's commit, after overlay.sh)
# With the cluster stopped: gemm_race_stress.py (down, rows 6 and 48, 60 rounds)
# on dgx3 in a fresh r5m container with the det-slices overlay, first with the
# shipped dense GEMM, then with 0007 (proxy fence before stage release).
# Restores r5m.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism
log() { echo "$(date -u +%FT%TZ) $*"; }
for c in config/cluster.json $E/cluster-*.json; do
  bin/spark --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
done
# The r5m B12X tree plus 0007, built like overlay.sh builds its overlays.
B=$PWD/.work/upstreams/b12x
T=$(mktemp -d /tmp/b12x-0007.XXXX)
trap "git -C $B worktree remove --force $T" EXIT
git -C "$B" worktree add -q --detach "$T" f8069b2c0be1311df3b112591c6b8876a843f8be
for p in $(grep -v "^#" patches/b12x/series | grep -v "^$") ../../$E/0007-gemm-fence-stage-reads-before-tma-refill.patch; do
  case $p in ../../*) f=$PWD/${p#../../} ;; *) f=$PWD/patches/b12x/$p ;; esac
  git -C "$T" -c user.name="Christopher Owen" \
    -c user.email="3221756+christopherowen@users.noreply.github.com" \
    am --quiet --committer-date-is-author-date "$f"
done
[ "$(git -C "$T" rev-parse HEAD^{tree})" = 693aaed550673dcd86655ad4b2c3589882f0bd29 ] || { log "0007 tree mismatch"; exit 1; }
scp -q $E/gemm_race_stress.py dgx3:/tmp/gemm_race_stress.py
scp -q "$T/b12x/_lib/dense_gemm.py" dgx3:/tmp/dense_gemm-0007.py
log "gemm race stress: shipped vs 0007"
ssh dgx3 'bash -s' > "$out/gemm-race-0007.txt" 2>&1 <<'REMOTE'
O=$HOME/spark3-overlay/det-slices
B=/opt/spark3/candidate/b12x
base="-v /tmp/gemm_race_stress.py:/tmp/gemm_race_stress.py:ro"
for f in b12x/moe/fused_moe/_impl.py b12x/moe/fused_moe/_preparation.py \
  b12x/moe/fused_moe/_tuning.py b12x/moe/_shared/kernels/dynamic.py \
  b12x/moe/_shared/kernels/silu.py; do
  base="$base -v $O/$f:$B/$f:ro"
done
run() {
  mounts=$1; shift
  docker run --rm --gpus all --ipc=host $mounts -w $B -e CUTE_DSL_ARCH=sm_121a \
    -e B12X_DENSE_SPLITK_TURBO=0 -e B12X_CUTE_COMPILE_CACHE_DIR=/tmp/cute \
    -e CUTE_DSL_CACHE_DIR=/tmp/cutedsl -e B12X_COMPILE_CACHE_DIR=/tmp/b12xc \
    --entrypoint python3 vllm-ds41f-kkref:04c30fa98e79-r5m /tmp/gemm_race_stress.py "$@"
}
echo "== shipped dense GEMM"
run "$base" --shape down --caps 6,48 --rounds 60
echo "== 0007 proxy fence"
run "$base -v /tmp/dense_gemm-0007.py:$B/b12x/_lib/dense_gemm.py:ro" --shape down --caps 6,48 --rounds 60
REMOTE
log "stress done"
grep -E "^==|cap|OK|FAIL|Error|Traceback" "$out/gemm-race-0007.txt" | head -30
bin/spark cluster start --replace --apply | grep -v 'docker run'
log "done"
