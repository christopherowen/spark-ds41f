#!/bin/bash
# Run a command in the candidate-37 image with the branch's changed files
# (against candidate-37's patch head) mounted over the image's vLLM tree.
set -euo pipefail
TREE=${TREE:-/tmp/842-final/vllm}
BASE=${BASE:-5b7cfa37f}
DST=/opt/spark-ds41f/candidate/vllm
MOUNTS=()
for f in $(cd "$TREE" && git diff --name-only "$BASE"); do
  MOUNTS+=(-v "$TREE/$f:$DST/$f:ro")
done
exec docker run --rm --gpus all --ipc=host --network none \
  -e HF_HUB_OFFLINE=1 -e TILELANG_CACHE_DIR=/c/tl -e PYTHONPATH=/c/stub \
  -v /tmp/842-final/tlcache:/c/tl -v /tmp/842-final:/c/w \
  -v /tmp/842-final/tblib_stub.py:/c/stub/tblib_stub.py:ro \
  "${MOUNTS[@]}" -w "$DST" --entrypoint bash spark-ds41f:candidate-37 -c "$*"
