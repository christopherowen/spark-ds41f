#!/bin/bash
set -eu
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
export PYTHONPATH=/home/swank/spark-lab/collective-pdl/sparknet
cd /home/swank/spark-lab/collective-pdl/deployment
exp=experiments/2026-11-05-collective-pdl
python3 /home/swank/spark-lab/collective-pdl/seed-inputs.py >"$exp/runs/cached-inputs.log" 2>&1
bin/spark --cluster-config "$exp/tp3-triangle.json" build prepare >"$exp/runs/full-prepare.log" 2>&1
printf '0\n' >"$exp/runs/full-prepare.exit"
bin/spark --cluster-config "$exp/tp3-triangle.json" build check >"$exp/runs/full-build-check.log" 2>&1
printf '0\n' >"$exp/runs/full-build-check.exit"
