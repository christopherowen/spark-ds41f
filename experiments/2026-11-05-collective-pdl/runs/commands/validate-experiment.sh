#!/bin/bash
set -u
export PYTHONPATH=/home/swank/spark-lab/collective-pdl/sparknet
export MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2 OMP_NUM_THREADS=2
cd /home/swank/spark-lab/collective-pdl/deployment
exp=experiments/2026-11-05-collective-pdl
for shape in tp3-triangle tp4-ring; do
 bin/spark --cluster-config "$exp/$shape.json" doctor >"$exp/runs/doctor-$shape.log" 2>&1
 echo $? >"$exp/runs/doctor-$shape.exit"
done
bin/spark --cluster-config "$exp/tp3-triangle.json" build prepare --only sparknet >"$exp/runs/prepare-sparknet.log" 2>&1
echo $? >"$exp/runs/prepare-sparknet.exit"
git diff --check >"$exp/runs/diff-check.log" 2>&1
echo $? >"$exp/runs/diff-check.exit"
