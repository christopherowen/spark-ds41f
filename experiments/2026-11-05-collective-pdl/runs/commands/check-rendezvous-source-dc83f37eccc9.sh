#!/bin/bash
set -eu
export PATH=/home/swank/spark-lab/collective-pdl/venv/bin:$PATH
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
cd /home/swank/spark-lab/collective-pdl/sparknet
make check > /home/swank/spark-lab/collective-pdl/rendezvous-checks/source-checks.log 2>&1
git diff fb027b4b80e5262416bb6db2b95b671498f6de74 -- sparknet/oneshot > /home/swank/spark-lab/collective-pdl/rendezvous-checks/kernel-delta.diff
test ! -s /home/swank/spark-lab/collective-pdl/rendezvous-checks/kernel-delta.diff
