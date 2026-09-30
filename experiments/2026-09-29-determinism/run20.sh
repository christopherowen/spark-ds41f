#!/bin/bash
# usage: run20.sh [NODE]   (on dgx1, deployment checkout at this experiment's commit, cluster
#                           stopped, after overlay.sh and overlay_masked.sh)
# On NODE (default dgx2), fresh r5m containers: the 0008 GPU tests (slice
# partials with dead routes checked bitwise against the masked order, the
# neighbouring W4A8 dynamic oracles, the planner), then moe_combine_bench.py:
# atomic, collapsed and slices with the det-slices overlay, masked with
# det-masked (timing at 6-48 rows, 0-50% dead), and Nsight Compute DRAM/L2
# bytes for atomic, slices and masked at 6 and 48 rows. Does not start the cluster.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism
NODE=${1:-dgx2}
log() { echo "$(date -u +%FT%TZ) $*"; }
scp -q $E/moe_combine_bench.py $NODE:/tmp/moe_combine_bench.py
log "0008 tests and combine bench on $NODE"
ssh $NODE 'bash -s' > "$out/moe-combine-bench.txt" 2>&1 <<'REMOTE'
B=/opt/spark3/candidate/b12x
MOE="b12x/moe/fused_moe/_impl.py b12x/moe/fused_moe/_preparation.py b12x/moe/fused_moe/_tuning.py \
b12x/moe/_shared/kernels/dynamic.py b12x/moe/_shared/kernels/silu.py"
mounts_for() {
  local o=$HOME/spark3-overlay/$1 files=$MOE m="-v /tmp/moe_combine_bench.py:/tmp/moe_combine_bench.py:ro -v /opt/nvidia/nsight-compute:/opt/nvidia/nsight-compute:ro"
  [ "$1" = det-masked ] && files="$files b12x/moe/_shared/kernels/w4a16/kernel.py tests/moe/test_w4a8_dynamic_kernel.py tests/preparation/test_tuning_predicates.py"
  for f in $files; do m="$m -v $o/$f:$B/$f:ro"; done
  echo "$m"
}
run() {
  local overlay=$1 entry=$2; shift 2
  docker run --rm --gpus all --ipc=host --cap-add SYS_ADMIN $(mounts_for $overlay) -w $B -e CUTE_DSL_ARCH=sm_121a \
    -e B12X_DENSE_SPLITK_TURBO=0 -e B12X_CUTE_COMPILE_CACHE_DIR=/tmp/cute \
    -e CUTE_DSL_CACHE_DIR=/tmp/cutedsl -e B12X_COMPILE_CACHE_DIR=/tmp/b12xc \
    --entrypoint "$entry" vllm-ds41f-kkref:04c30fa98e79-r5m "$@"
}
echo "== 0008 tests"
run det-masked python3 -m pytest -q -p no:cacheprovider tests/moe/test_w4a8_dynamic_kernel.py \
  -k "slice_partials or dynamic_matches_oracle or ignores_inactive or small_tile_parallel or boundary_m_sizes"
echo "kernel tests exit $?"
run det-masked python3 -m pytest -q -p no:cacheprovider tests/preparation/test_tuning_predicates.py
echo "planner tests exit $?"
for pair in atomic:det-slices collapsed:det-slices slices:det-slices masked:det-masked; do
  mode=${pair%%:*}; overlay=${pair#*:}
  echo "== timing $mode"
  run $overlay python3 /tmp/moe_combine_bench.py --mode $mode
done
for pair in atomic:det-slices slices:det-slices masked:det-masked; do
  mode=${pair%%:*}; overlay=${pair#*:}
  echo "== ncu $mode"
  run $overlay /opt/nvidia/nsight-compute/2025.3.1/ncu --csv --page raw \
    --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum \
    -k regex:"MoEDynamic|TopKSum" python3 /tmp/moe_combine_bench.py --mode $mode --ncu --rows 6,48 --dead 0,0.5
done
REMOTE
log "done"
grep -E "^==|passed|failed|exit|us_per_call|Error|Traceback|ERR" "$out/moe-combine-bench.txt" | head -80
