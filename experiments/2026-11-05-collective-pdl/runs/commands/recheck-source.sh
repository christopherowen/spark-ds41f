#!/bin/bash
set -eu
export PATH=/home/swank/spark-lab/collective-pdl/venv/bin:$PATH
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
cd /home/swank/spark-lab/collective-pdl/sparknet
make check >../runs/source-recheck.log 2>&1
