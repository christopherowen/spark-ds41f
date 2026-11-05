#!/bin/bash
set -eu
export PATH=/home/swank/spark-lab/collective-pdl/venv/bin:$PATH
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
exec python3 /home/swank/spark-lab/collective-pdl/prepare-alternatives.py
