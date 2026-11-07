#!/usr/bin/env bash
set -euo pipefail
# Run on an isolated CPU host, never on the serving container.
source_dir=$(realpath "${1:?prepared candidate vLLM checkout}")
here=$(cd -- "$(dirname -- "$0")" && pwd)
image=sha256:3795255f485ea5c1a9db54c742921dd9b2917daad0799361e5c6f56a70119713
awk '/MemAvailable:/ {print "host_mem_available_kib", $2}' /proc/meminfo
flock -w 30 /tmp/spark-cpu-analysis.lock docker run --rm --runtime runc   --cpus=1 --memory=4g --memory-swap=4g --pids-limit=128 --network=none   -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void   -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MAX_JOBS=1   -e PYTHONPATH=/src -v "$source_dir:/src:ro" -v "$here:/check:ro"   --entrypoint python3 "$image" /check/cpu.py
awk '/MemAvailable:/ {print "host_mem_available_kib", $2}' /proc/meminfo
