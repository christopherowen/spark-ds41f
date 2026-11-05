#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MAKEFLAGS=-j1 MAX_JOBS=1
export GIT_CONFIG_COUNT=2 GIT_CONFIG_KEY_0=pack.threads GIT_CONFIG_VALUE_0=1 GIT_CONFIG_KEY_1=gc.auto GIT_CONFIG_VALUE_1=0
root=/home/chris/projects/engram-cache-probe
repo=/home/chris/projects/spark-ds41f
if test ! -e "$root/deployment/.git"; then
  git -C "$repo" worktree add --detach "$root/deployment" 2c4b23afcd834ba2adf6a15ca434e59fa4f49f74
fi
cd "$root/deployment"
test "$(git rev-parse HEAD)" = 2c4b23afcd834ba2adf6a15ca434e59fa4f49f74
for f in "$repo"/config/nodes*.json; do
  name=$(basename "$f")
  if ! git ls-files --error-unmatch "config/$name" >/dev/null 2>&1; then cp "$f" "config/$name"; fi
done
cp config/examples/nodes-ring4.json config/nodes-ring4.local.json
cp config/nodes.example.json config/nodes.json
bin/spark doctor > "$root/doctor.log" 2>&1
export GIT_ALTERNATE_OBJECT_DIRECTORIES="$repo/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-9b160482/src/vllm/.git/objects"
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-38/tp3.json build prepare --only vllm
