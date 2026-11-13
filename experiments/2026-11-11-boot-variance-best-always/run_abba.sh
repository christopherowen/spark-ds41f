#!/usr/bin/env bash
set -euo pipefail
exec 9>>/tmp/spark-lab-gpu.lock
flock 9
if [ -e "$HOME/spark-hold.json" ]; then echo "HOLD PRESENT"; exit 3; fi
if docker ps --no-trunc --format "{{.Names}} {{.Command}}" | grep -q "vllm serve"; then echo "SERVING RUNNING"; exit 4; fi

trap "sudo -n nvidia-smi -rgc >/dev/null 2>&1 || true" EXIT
sudo -n nvidia-smi -lgc 2418,2418 >/dev/null

WT=/tmp/rc-v2/src
DST=/opt/spark-ds41f/candidate/vllm
B=/tmp/rc-v2

run_one() {
  local tag=$1
  local mode=$2
  local replays=$3
  local extra_env=()
  local mounts=()
  sudo -n sh -c "sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory"
  python3 /tmp/rc-v2/seed_frag.py
  if [ "$mode" = "base" ]; then
    extra_env=(-e VLLM_ROCE_CPU_AFFINITY=0 -e VLLM_DUMMY_WEIGHT_ARENA=0)
  else
    # Pre-launch host buddy defragmentation (matches cluster.py defragment_buddy_memory)
    sudo -n sh -c "sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory"
    extra_env=(-e VLLM_ROCE_CPU_AFFINITY=1 -e VLLM_DUMMY_WEIGHT_ARENA=1)
    mounts=(
      -v "$WT/vllm/utils/cpu_affinity.py:$DST/vllm/utils/cpu_affinity.py:ro"
      -v "$WT/vllm/v1/worker/gpu_worker.py:$DST/vllm/v1/worker/gpu_worker.py:ro"
      -v "$WT/vllm/model_executor/model_loader/base_loader.py:$DST/vllm/model_executor/model_loader/base_loader.py:ro"
      -v "$WT/vllm/model_executor/model_loader/direct_loader.py:$DST/vllm/model_executor/model_loader/direct_loader.py:ro"
      -v "$WT/vllm/model_executor/model_loader/dummy_loader.py:$DST/vllm/model_executor/model_loader/dummy_loader.py:ro"
      -v "$WT/vllm/model_executor/model_loader/weight_utils.py:$DST/vllm/model_executor/model_loader/weight_utils.py:ro"
    )
  fi
  echo "=== STARTING $tag ($mode, lgc=2418, replays=$replays) at $(date -u +%H:%M:%SZ) ==="
  docker run --rm --name "rc-$tag" --gpus all --ipc=host \
    --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
    --cpuset-cpus=5-9,15-19 \
    -e HF_HUB_OFFLINE=1 \
    -e PYTHONPATH=$DST \
    -e TILELANG_CACHE_DIR=/tl_cache \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${extra_env[@]}" \
    -v "$HOME/.cache/huggingface:/hf:ro" \
    -v "$B/tl-cache:/tl_cache" \
    -v "$B:/bench" \
    "${mounts[@]}" \
    -w $DST \
    --entrypoint python3 \
    spark-ds41f:candidate-54 /bench/bench_one_boot.py "/bench/runs-lgc/${tag}.json" "$tag" "$replays" > "$B/runs-lgc/${tag}.log" 2>&1
  grep -E "Buddy hugepage sponge|BOOT_SUMMARY" "$B/runs-lgc/${tag}.log"
}

mkdir -p /tmp/rc-v2/runs-lgc
for item in "$@"; do
  tag="${item%%:*}"
  rest="${item#*:}"
  mode="${rest%%:*}"
  replays="${rest#*:}"
  run_one "$tag" "$mode" "$replays"
done
