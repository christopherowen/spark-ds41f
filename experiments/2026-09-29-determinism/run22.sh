#!/bin/bash
# usage: run22.sh [NODE]   (on dgx1, deployment checkout at this experiment's commit, after
#                           overlay.sh and overlay_masked.sh)
# Stops the cluster, then on NODE (default dgx3) in fresh containers:
# 1. the B12X planner tests with the det-masked overlay and the default
#    environment (run20 exported B12X_DENSE_SPLITK_TURBO=0, which changes the
#    code-generation snapshots those tests pin);
# 2. Nsight Compute sysmem traffic (L2 fills from memory, L2 read misses, sysmem
#    write sectors, L2 bytes) for the atomic, slice and masked combines at 6 and
#    48 rows, 0 and 50% dead (GB10 exposes no dram__ counters);
# 3. gemm_fence_timing.py in the r5m image (shipped dense GEMM) and the r5n image
#    (0004's fence): identical inputs, alone and beside the routed MoE.
# Restores config/cluster.json.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism
NODE=${1:-dgx3}
log() { echo "$(date -u +%FT%TZ) $*"; }
for c in config/cluster.json $E/cluster-*.json experiments/2026-09-30-r5n/cluster-*.json; do
  bin/spark3 --cluster-config "$c" cluster stop --remove --apply >/dev/null 2>&1 || true
done
scp -q $E/moe_combine_bench.py $E/gemm_fence_timing.py $NODE:/tmp/
log "run22 on $NODE"
ssh $NODE 'bash -s' > "$out/run22.txt" 2>&1 <<'REMOTE'
B=/opt/spark3/candidate/b12x
MOE="b12x/moe/fused_moe/_impl.py b12x/moe/fused_moe/_preparation.py b12x/moe/fused_moe/_tuning.py \
b12x/moe/_shared/kernels/dynamic.py b12x/moe/_shared/kernels/silu.py"
mounts_for() {
  local o=$HOME/spark3-overlay/$1 files=$MOE
  local m="-v /tmp/moe_combine_bench.py:/tmp/moe_combine_bench.py:ro -v /tmp/gemm_fence_timing.py:/tmp/gemm_fence_timing.py:ro -v /opt/nvidia/nsight-compute:/opt/nvidia/nsight-compute:ro"
  [ "$1" = det-masked ] && files="$files b12x/moe/_shared/kernels/w4a16/kernel.py tests/moe/test_w4a8_dynamic_kernel.py tests/preparation/test_tuning_predicates.py"
  [ "$1" = none ] && files=""
  for f in $files; do m="$m -v $o/$f:$B/$f:ro"; done
  echo "$m"
}
run() {
  local image=$1 overlay=$2 envs=$3 entry=$4; shift 4
  docker run --rm --gpus all --ipc=host --cap-add SYS_ADMIN $(mounts_for $overlay) $envs -w $B -e CUTE_DSL_ARCH=sm_121a \
    -e B12X_CUTE_COMPILE_CACHE_DIR=/tmp/cute -e CUTE_DSL_CACHE_DIR=/tmp/cutedsl -e B12X_COMPILE_CACHE_DIR=/tmp/b12xc \
    --entrypoint "$entry" vllm-ds41f-kkref:04c30fa98e79-$image "$@"
}
echo "== planner tests (det-masked, default environment)"
run r5m det-masked "" python3 -m pytest -q -p no:cacheprovider tests/preparation/test_tuning_predicates.py
echo "planner tests exit $?"
M=gpu__time_duration.sum,lts__t_bytes.sum,lts__d_sectors_fill_sysmem.sum,lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum,lts__t_sectors_aperture_sysmem_op_write.sum
for pair in atomic:det-slices slices:det-slices masked:det-masked; do
  mode=${pair%%:*}; overlay=${pair#*:}
  echo "== ncu $mode"
  run r5m $overlay "-e B12X_DENSE_SPLITK_TURBO=0" /opt/nvidia/nsight-compute/2025.3.1/ncu --csv --page raw --metrics $M \
    -k regex:"MoEDynamic|TopKSum" python3 /tmp/moe_combine_bench.py --mode $mode --ncu --rows 6,28,48 --dead 0,0.5
done
for image in r5m r5n; do
  echo "== gemm timing $image"
  run $image none "" python3 /tmp/gemm_fence_timing.py
done
REMOTE
log "run22 done"
grep -E "^==|passed|failed|exit|cap [0-9]|Error|Traceback" "$out/run22.txt" | head -40
bin/spark3 cluster start --replace --apply | grep -v 'docker run'
log "done"
