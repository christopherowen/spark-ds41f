#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MAKEFLAGS=-j1 MAX_JOBS=1
export GIT_CONFIG_COUNT=4 GIT_CONFIG_KEY_0=pack.threads GIT_CONFIG_VALUE_0=1 GIT_CONFIG_KEY_1=gc.auto GIT_CONFIG_VALUE_1=0 GIT_CONFIG_KEY_2=index.threads GIT_CONFIG_VALUE_2=1 GIT_CONFIG_KEY_3=core.preloadIndex GIT_CONFIG_VALUE_3=false
root=/home/chris/projects/engram-cache-probe
python3 "$root/package-final.py"
cd "$root/deployment"
export GIT_ALTERNATE_OBJECT_DIRECTORIES=/home/chris/projects/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-9b160482/src/vllm/.git/objects
for variant in aligned filter; do
    profile=experiments/2026-11-05-engram-cache-probe/$variant
    bin/spark --cluster-config "$profile/tp3.json" build prepare --lock --only vllm > "$root/final-candidates/$variant-prepare.log" 2>&1
    bin/spark --cluster-config "$profile/tp3.json" doctor > "$root/final-candidates/$variant-doctor-tp3.log" 2>&1
    bin/spark --cluster-config "$profile/tp4.json" doctor > "$root/final-candidates/$variant-doctor-tp4.log" 2>&1
done
git diff --check
