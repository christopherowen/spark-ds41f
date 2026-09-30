#!/bin/bash
# usage: run_stress.sh   (on dgx3, cluster stopped, inputs staged in /tmp/fence-audit)
# fence_race_stress.py in fresh r5m containers: the dense GEMM control, then
# the BF16 prefill projection, the mHC TF32 prefill projection and the
# contiguous varlen attention, first as shipped, then with 0008 (fence before
# each TMA stage release). The routed MoE co-runner uses a private copy of the
# det-slices overlay, as run17.sh did. Starts no cluster config.
set -uo pipefail
W=/tmp/fence-audit
IMG=vllm-ds41f-kkref:04c30fa98e79-r5m
B=/opt/spark3/candidate/b12x
log() { echo "$(date -u +%FT%TZ) $*"; }
base="-v $W/fence_race_stress.py:/tmp/fence_race_stress.py:ro"
for f in b12x/moe/fused_moe/_impl.py b12x/moe/fused_moe/_preparation.py \
  b12x/moe/fused_moe/_tuning.py b12x/moe/_shared/kernels/dynamic.py \
  b12x/moe/_shared/kernels/silu.py; do
  base="$base -v $W/det-slices/$f:$B/$f:ro"
done
fenced="$base"
for f in b12x/gemm/bf16_gemv/_prefill.py b12x/norm/mhc/_kernels.py \
  b12x/attention/_shared/contiguous/forward.py; do
  fenced="$fenced -v $W/0008/$f:$B/$f:ro"
done
run() {
  variant=$1 mounts=$2; shift 2
  mkdir -p "$W/cache-$variant"
  log "$variant $*"
  docker run --rm --name fence-audit-stress --gpus all --ipc=host $mounts \
    -v "$W/cache-$variant:/work/cache" -w $B -e CUTE_DSL_ARCH=sm_121a \
    -e B12X_DENSE_SPLITK_TURBO=0 -e B12X_CUTE_COMPILE_CACHE_DIR=/work/cache/cute \
    -e CUTE_DSL_CACHE_DIR=/work/cache/cutedsl -e B12X_COMPILE_CACHE_DIR=/work/cache/b12x \
    --entrypoint python3 $IMG /tmp/fence_race_stress.py "$@"
  log "$variant $1 $2 exit $?"
}
log "image files (shipped r5m, 0008 inputs):"
docker run --rm --entrypoint sha256sum -w $B $IMG b12x/gemm/bf16_gemv/_prefill.py \
  b12x/norm/mhc/_kernels.py b12x/attention/_shared/contiguous/forward.py
(cd $W/0008 && sha256sum b12x/gemm/bf16_gemv/_prefill.py b12x/norm/mhc/_kernels.py \
  b12x/attention/_shared/contiguous/forward.py)
echo "== shipped"
run shipped "$base" --target dense --sizes 6,48 --rounds 60 --calls 200
run shipped "$base" --target bf16prefill --sizes 256,1024 --rounds 60 --calls 200
run shipped "$base" --target mhc --sizes 256,1024 --rounds 60 --calls 200
run shipped "$base" --target varlen --sizes 1024,4096 --rounds 30 --calls 100
echo "== 0008"
run fenced "$fenced" --target bf16prefill --sizes 256,1024 --rounds 60 --calls 200
run fenced "$fenced" --target mhc --sizes 256,1024 --rounds 60 --calls 200
run fenced "$fenced" --target varlen --sizes 1024,4096 --rounds 30 --calls 100
docker run --rm -v "$W:/w" --entrypoint chmod $IMG -R a+rwX /w/cache-shipped /w/cache-fenced
log "done"
