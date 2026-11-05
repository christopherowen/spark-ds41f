#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MAKEFLAGS=-j1 MAX_JOBS=1
export GIT_CONFIG_COUNT=2 GIT_CONFIG_KEY_0=pack.threads GIT_CONFIG_VALUE_0=1 GIT_CONFIG_KEY_1=gc.auto GIT_CONFIG_VALUE_1=0
root=/home/chris/projects/engram-cache-probe
python3 "$root/package.py"
cd "$root/deployment"
export GIT_ALTERNATE_OBJECT_DIRECTORIES=/home/chris/projects/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-9b160482/src/vllm/.git/objects
bin/spark --cluster-config experiments/2026-11-05-engram-cache-probe/tp3.json build prepare --lock --only vllm
bin/spark --cluster-config experiments/2026-11-05-engram-cache-probe/tp3.json doctor
bin/spark --cluster-config experiments/2026-11-05-engram-cache-probe/tp4.json doctor
git diff --check
