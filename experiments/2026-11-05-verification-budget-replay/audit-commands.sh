#!/bin/sh
# CPU-only audit on sp8, published commit 2a769af2.
# Wrapper: systemd-run --user --scope --quiet -p MemoryMax=2G -p CPUQuota=100%
#   taskset -c 0 flock -w 30 ~/spark-lab/offline-cpu.lock
#   flock -w 30 ~/spark-cpu-work.lock bash -s
set -eu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1
cd "$HOME/spark-lab/verification-budget-replay/checkout"
experiment=experiments/2026-11-05-verification-budget-replay
grid_root=experiments/2026-11-04-topk-live-qualification/runs/2026-11-04-dgx/attempt-2/performance
source_path=.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-a35ced21/src/vllm/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py
/usr/bin/time -v ../env/bin/python -m unittest discover -s "$experiment" -p 'test_*.py' \
  > ../logs/audit-tests.txt 2> ../logs/audit-tests-resource.txt
/usr/bin/time -v ../env/bin/python "$experiment/audit.py" \
  --trace ../inputs/k5.jsonl --source "$source_path" \
  --grid manifests/suites/2026-10-30-spark-ds41f-1030-tp4/grid.json \
  --grid "$grid_root/candidate-40/release-tp3/grid.json" \
  --grid "$grid_root/candidate-40-repeat/release-tp3/grid.json" \
  --output ../logs/alternatives.json \
  > ../logs/alternatives-stdout.txt 2> ../logs/alternatives-resource.txt
