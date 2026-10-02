#!/usr/bin/env bash
set -euo pipefail
image=${1:-vllm-ds41f-kkref:04c30fa98e79-r5o}
test -z "$(docker ps -q --filter name=^dsv41-karmic-kraken$)"
repo="$HOME/projects/dgx-spark-memory-saver"
"$repo/scripts/status" --json
for test in allocation_stress cuda_smoke; do
  docker run --rm --gpus all --memory=20g --memory-swap=20g --network=none --ipc=host \
    --entrypoint python3 -v "$repo/tests:/tests:ro" "$image" "/tests/$test.py"
done
