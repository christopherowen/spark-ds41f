#!/bin/bash
# Run a command in spark-ds41f:candidate-37 with the overlay tree's files (this
# task's changed vLLM files, at their repository paths) mounted over the image's
# vLLM source (tree af0fb152, the candidate-37 patch head).
set -euo pipefail
ROOT=~/src/markov-fusion
mkdir -p "$ROOT"/cache/tilelang "$ROOT"/cache/triton "$ROOT"/bench
mounts=()
cd "$ROOT/tree"
for f in $(find . -type f | sed 's#^\./##' | sort); do
  mounts+=(-v "$ROOT/tree/$f:/opt/spark-ds41f/candidate/vllm/$f:ro")
done
docker run --rm --gpus all --ipc=host --network=host \
  --name markov-fusion-$$ \
  -e TILELANG_CACHE_DIR=/cache/tilelang -e TRITON_CACHE_DIR=/cache/triton \
  -v "$ROOT/cache:/cache" -v "$ROOT/bench:/bench" "${mounts[@]}" \
  -w /opt/spark-ds41f/candidate/vllm --entrypoint bash spark-ds41f:candidate-37 -c "$*"
