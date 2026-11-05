#!/bin/sh
# Recorded CPU-only commands, run inside the isolated checkout on sp8.
# Scope wrapper: systemd-run --user --scope --quiet -p MemoryMax=4G
#   -p CPUQuota=200% taskset -c 0,1 flock -n /tmp/spark-cpu-analysis.lock bash -s
# Source/environment setup used the same wrapper. No command operates the cluster.
set -eu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1
cd "$HOME/spark-lab/verification-budget-replay/checkout"
experiment=experiments/2026-11-05-verification-budget-replay
profile=experiments/2026-10-26-spark-ds41f-candidate/candidate-40/tp3.json
grid_root=experiments/2026-11-04-topk-live-qualification/runs/2026-11-04-dgx/attempt-2/performance
out=../logs

# Environment: python3 -m venv ../env
# ../env/bin/pip install --only-binary=:all: numpy==2.3.4
# Source: git fetch --no-tags origin feat/verification-budget-replay
#         git checkout --detach 4fccd724
/usr/bin/time -v ../env/bin/python experiments/2026-09-26-dspark-policy/replay.py \
  ../inputs/k5.jsonl ../inputs/costs-r4a.json \
  > "$out/historical-replay-remote.txt" 2> "$out/historical-replay-resource.txt"
/usr/bin/time -v ../env/bin/python "$experiment/replay.py" \
  --trace ../inputs/k5.jsonl \
  --grid manifests/suites/2026-10-30-spark-ds41f-1030-tp4/grid.json \
  --grid "$grid_root/candidate-40/release-tp3/grid.json" \
  --grid "$grid_root/candidate-40-repeat/release-tp3/grid.json" \
  --output "$out/cell-screen.json" \
  > "$out/cell-screen-stdout.txt" 2> "$out/cell-screen-resource.txt"
/usr/bin/time -v ../env/bin/python "$experiment/test_replay.py" \
  > "$out/replay-tests.txt" 2> "$out/replay-tests-resource.txt"
bin/spark --cluster-config "$profile" doctor
bin/spark --cluster-config "$profile" build prepare --only vllm
git diff --check
# Recorded separately: exits 1 because complete image inputs are absent.
bin/spark --cluster-config "$profile" build check
