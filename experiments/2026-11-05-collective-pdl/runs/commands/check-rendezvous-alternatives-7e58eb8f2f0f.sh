#!/bin/bash
set -eu
export PATH=/home/swank/spark-lab/collective-pdl/venv/bin:$PATH
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
for phase in staged arrived; do
 cd /home/swank/spark-lab/collective-pdl/sparknet-$phase
 make check > /home/swank/spark-lab/collective-pdl/rendezvous-checks/source-$phase-checks.log 2>&1
done
