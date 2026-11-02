#!/bin/bash
# runimg.sh IMAGE WORKDIR_TREE -- args...  (python3 in the image, the tree's changed files mounted)
set -euo pipefail
IMAGE=$1; TREE=$2; shift 2
W=/tmp/c37-ladder-739
mkdir -p $W/cache
R=/opt/spark-ds41f/candidate/vllm
MOUNTS=()
for f in vllm/models/deepseek_v4_1/nvidia/dspark.py tests/kernels/test_deepseek_v41_tilelang_dspark_context.py; do
  MOUNTS+=(-v "$TREE/$f:$R/$f:ro")
done
MOUNTS+=(-v "$W/bench:/bench:ro")
exec docker run --rm --gpus all --ipc=host --ulimit=core=0 -e HF_HUB_OFFLINE=1 -e TILELANG_CACHE_DIR=/c/tilelang -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/bench:/opt/spark-ds41f/candidate/vllm:/opt/spark-ds41f/candidate/b12x \
  "${MOUNTS[@]}" -v $W/cache:/c -w $R --entrypoint python3 "$IMAGE" "$@"
