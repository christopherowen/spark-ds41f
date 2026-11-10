#!/bin/bash
# One boot of the TP3-rank engine grid harness: run.sh <arm> <boot> [allreduce_us] [extra args]
# candidate-50 image + the arm's files where they differ from the image (ro).
set -euo pipefail
W=/home/chris/spark-lab/agents/tp3-grid-attribution
arm=$1; boot=$2; ar=${3:-30}; shift 3 || shift $#
extra="$*"
tree=$W/trees/$arm; [ "$arm" = c54 ] && tree=$W/c54
tag=${arm}-ar${ar}-boot${boot}${TAG:-}
mounts=()
while read -r p; do [ -n "$p" ] && mounts+=(-v "$tree/$p:/opt/spark-ds41f/candidate/vllm/$p:ro"); done < $W/out/ovl-$arm.txt
mkdir -p $W/out/runs $W/jit
# EXTRA_ENV: more docker -e flags (an arm variant); NSYS=1: run under the host's Nsight Systems
envs=(); for e in ${EXTRA_ENV:-}; do envs+=(-e "$e"); done
entry=(--entrypoint python3 spark-ds41f:candidate-50)
if [ -n "${NSYS:-}" ]; then
  mounts+=(-v /opt/nvidia/nsight-systems:/opt/nvidia/nsight-systems:ro)
  entry=(--entrypoint /opt/nvidia/nsight-systems/2025.3.2/target-linux-sbsa-armv8/nsys spark-ds41f:candidate-50
         profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none
         --force-overwrite=true -o /out/runs/$tag python3)
fi
if [ -e ~/spark-hold.json ]; then echo "hold file present; not running"; exit 3; fi
{
echo "### $tag"
echo "host: $(hostname) image: $(docker image inspect spark-ds41f:candidate-50 --format '{{.Id}}')"
echo "arm: $arm tree: $(git -C $tree rev-parse HEAD^{tree}) head: $(git -C $tree rev-parse --short=12 HEAD) overlay files: ${#mounts[@]}/2"
echo "variant: EXTRA_ENV=${EXTRA_ENV:-} NSYS=${NSYS:-} CARVEOUT=${CARVEOUT:-1} args: $extra"
echo "issued_utc: $(date -u +%FT%TZ)"
} >> $W/out/runs/commands.log
flock /tmp/spark-lab-gpu.lock bash -c '
  echo "start_utc: $(date -u +%FT%TZ) loadavg: $(cut -d" " -f1-3 /proc/loadavg) hold: $(ls ~/spark-hold.json 2>/dev/null || echo absent)" >> '"$W"'/out/runs/commands.log
  [ -e ~/spark-hold.json ] && { echo "hold file appeared; not running"; exit 3; }
  docker run --rm --name tp3ga-'"$tag"' --gpus all --ipc=host --shm-size=16g --cpuset-cpus=5-9,15-19 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HOME=/jit/hf \
    -e CUDA_DEVICE_MAX_CONNECTIONS=32 -e MALLOC_ARENA_MAX=2 -e MALLOC_MMAP_THRESHOLD_=1048576 \
    -e OMP_NUM_THREADS=8 -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6" \
    -e SPARK_DS41F_DSPARK_DEAD_ROWS_TAU=0.2 -e SPARK_DS41F_DSPARK_MAIN_PROJ_TP=1 -e SPARK_DS41F_DSPARK_VOCAB_PARALLEL=1 \
    -e SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=${CARVEOUT:-1} \
    --mount type=bind,source=/dev/dri/by-path/pci-000f:01:00.0-card,target=/dev/dri/card0 --device-cgroup-rule="c 226:* rw" \
    -e VLLM_DS41_HEAD_DTYPE=float32 -e VLLM_DS41_PACKED_BF16_LM_HEAD=1 \
    -e VLLM_L2_PREFETCH_GRID=2 -e VLLM_L2_PREFETCH_KERNELS=tilelang -e VLLM_LATE_ADMISSION_LAYER=9 \
    -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    -e TILELANG_CACHE_DIR=/jit/tilelang -e VLLM_CACHE_ROOT=/jit/vllm -e VLLM_CACHE_DIR=/jit/vllm \
    -e TRITON_CACHE_DIR=/jit/triton -e CUDA_CACHE_PATH=/jit/nv -e CUDA_CACHE_MAXSIZE=4294967296 -e XDG_CACHE_HOME=/jit \
    -v '"$W"'/jit:/jit -v /home/chris/.cache/huggingface/hub:/hf/hub:ro \
    -v '"$W"'/h:/h:ro -v '"$W"'/out:/out \
    '"$(printf '%q ' "${mounts[@]}" "${envs[@]}")"' \
    -w /opt/spark-ds41f/candidate/vllm '"$(printf '%q ' "${entry[@]}")"' \
    /h/tp3_grid.py --arm '"$arm"' --boot '"$boot"' --allreduce-us '"$ar"' --expect /out/expect-'"$arm"'.json \
    --out /out/runs/'"$tag"'.json '"$extra"' > '"$W"'/out/runs/'"$tag"'.txt 2>&1
  rc=$?
  echo "end_utc: $(date -u +%FT%TZ) exit: $rc loadavg: $(cut -d" " -f1-3 /proc/loadavg)" >> '"$W"'/out/runs/commands.log
  exit $rc
'
