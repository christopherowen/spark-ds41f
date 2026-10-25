#!/usr/bin/env bash
# The route probe on dgx4's CPU, in the serving image, beside nothing that serves: cores 0-4 and
# 10-14, a 16 GiB cap, reads throttled to 800 MB/s, no network. Run from this checkout's root.
set -u
OUT=${1:-$HOME/spark-lab/agents/suffix-drafter/out}
SNAP=/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170
IMAGE=spark-ds41f:19f2c20ed4d6-next-6
mkdir -p "$OUT"
docker run --rm --name suffix-drafter-routes --memory 16g --memory-swap 16g --cpus 10 --cpuset-cpus 0-4,10-14 \
  --network none --device-read-bps /dev/nvme0n1:800mb -e HF_HUB_OFFLINE=1 -e REF_THREADS=10 \
  -v "$HOME/.cache/huggingface/hub:/hub:ro" -v "$PWD:/repo:ro" -v "$OUT:/out" \
  --entrypoint nice "$IMAGE" -n 10 python3 -I /repo/experiments/2026-10-26-suffix-drafter/reference/routes_probe.py \
  "$SNAP" /out/routes.json --cases copy,tool,edit,prose --tokens 1000 \
  > "$OUT/routes.log" 2>&1
echo "exit $?" >> "$OUT/routes.status"
