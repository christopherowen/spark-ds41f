#!/bin/bash
set -eu
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
cd /home/swank/spark-lab/collective-pdl/deployment
exp=experiments/2026-11-05-collective-pdl
/home/swank/spark-lab/collective-pdl/venv/bin/python -m pip install --require-hashes -r requirements/bench-client.txt >"$exp/runs/deployment-dependencies.log" 2>&1
PYTHONPATH=/home/swank/spark-lab/collective-pdl/sparknet /home/swank/spark-lab/collective-pdl/venv/bin/python -m unittest discover -s tests >"$exp/runs/deployment-tests.log" 2>&1
printf '0\n' >"$exp/runs/deployment-tests.exit"
python3 scripts/render_docs.py --check >"$exp/runs/generated-docs.log" 2>&1
printf '0\n' >"$exp/runs/generated-docs.exit"
