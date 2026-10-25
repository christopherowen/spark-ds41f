#!/usr/bin/env bash
# The route probe on dgx4's CPU, in the serving image: cores 0-4 and 10-14, nice 10, a 16 GiB
# cap, reads throttled to 800 MB/s, no network, the hub read-only. Run from this checkout's root.
# It holds the kernel lab's lock (/tmp/spark-lab-gpu.lock, the flock kernel-local takes) for the
# whole run, so no GPU bundle times its kernels beside these CPU matmuls; it waits its turn
# behind queued bundles. No case starts after DEADLINE (UTC); the case running then finishes.
set -u
OUT=${1:-$HOME/spark-lab/agents/suffix-drafter/out}
DEADLINE=${2:-2026-10-25T22:30:00Z}
SNAP=/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170
IMAGE=spark-ds41f:19f2c20ed4d6-next-6
mkdir -p "$OUT"
echo "waiting for the lab lock $(date -u +%FT%TZ)" >> "$OUT/routes.status"
flock /tmp/spark-lab-gpu.lock bash -c '
  echo "lock held $(date -u +%FT%TZ)" >> "$1/routes.status"
  docker run --rm --name suffix-drafter-routes --memory 16g --memory-swap 16g --cpus 10 --cpuset-cpus 0-4,10-14 \
    --network none --device-read-bps /dev/nvme0n1:800mb -e HF_HUB_OFFLINE=1 -e REF_THREADS=10 \
    -v "$HOME/.cache/huggingface/hub:/hub:ro" -v "$PWD:/repo:ro" -v "$1:/out" \
    --entrypoint nice "$2" -n 10 python3 -I /repo/experiments/2026-10-26-suffix-drafter/reference/routes_probe.py \
    "$3" /out/routes.json --cases copy,prose,tool,edit --tokens 1000 --deadline "$4" \
    > "$1/routes.log" 2>&1
  echo "exit $? $(date -u +%FT%TZ)" >> "$1/routes.status"
' _ "$OUT" "$IMAGE" "$SNAP" "$DEADLINE"
