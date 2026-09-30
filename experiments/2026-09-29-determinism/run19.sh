#!/bin/bash
# usage: run19.sh [NODE]   (on dgx1, deployment checkout at this experiment's commit,
#                           cluster stopped, after overlay.sh)
# moe_combine_bench.py on NODE (default dgx2) in a fresh r5m container with the
# det-slices overlay: atomic, collapsed and slice-partial combines at 6-48
# verification rows with 0, 25 and 50% dead rows (graph replay timing and the
# profiler's kernel split), then Nsight Compute DRAM/L2 bytes for 6 and 48 rows
# at 0 and 50% dead. Does not start the cluster.
set -uo pipefail
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-29-determinism
out=results/private/determinism
NODE=${1:-dgx2}
log() { echo "$(date -u +%FT%TZ) $*"; }
scp -q $E/moe_combine_bench.py $NODE:/tmp/moe_combine_bench.py
log "combine bench on $NODE"
ssh $NODE 'bash -s' > "$out/moe-combine-bench.txt" 2>&1 <<'REMOTE'
O=$HOME/spark3-overlay/det-slices
B=/opt/spark3/candidate/b12x
mounts="-v /tmp/moe_combine_bench.py:/tmp/moe_combine_bench.py:ro -v /opt/nvidia/nsight-compute:/opt/nvidia/nsight-compute:ro"
for f in b12x/moe/fused_moe/_impl.py b12x/moe/fused_moe/_preparation.py \
  b12x/moe/fused_moe/_tuning.py b12x/moe/_shared/kernels/dynamic.py \
  b12x/moe/_shared/kernels/silu.py; do
  mounts="$mounts -v $O/$f:$B/$f:ro"
done
run() {
  docker run --rm --gpus all --ipc=host --cap-add SYS_ADMIN $mounts -w $B -e CUTE_DSL_ARCH=sm_121a \
    -e B12X_DENSE_SPLITK_TURBO=0 -e B12X_CUTE_COMPILE_CACHE_DIR=/tmp/cute \
    -e CUTE_DSL_CACHE_DIR=/tmp/cutedsl -e B12X_COMPILE_CACHE_DIR=/tmp/b12xc \
    --entrypoint "$1" vllm-ds41f-kkref:04c30fa98e79-r5m "${@:2}"
}
for mode in atomic collapsed slices; do
  echo "== timing $mode"
  run python3 /tmp/moe_combine_bench.py --mode $mode
done
for mode in atomic slices; do
  echo "== ncu $mode"
  run /opt/nvidia/nsight-compute/2025.3.1/ncu --csv --page raw \
    --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum \
    -k regex:"MoEDynamic|TopKSum" python3 /tmp/moe_combine_bench.py --mode $mode --ncu --rows 6,48 --dead 0,0.5
done
REMOTE
log "combine bench done"
grep -E "^==|us_per_call|Error|Traceback|ERR" "$out/moe-combine-bench.txt" | head -60
