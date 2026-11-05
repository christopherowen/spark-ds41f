#!/bin/bash
set -eu
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
cd /home/swank/spark-lab/collective-pdl/deployment
out=/home/swank/spark-lab/collective-pdl/published-f474b681
mkdir "$out"
git rev-parse HEAD > "$out/commit.txt"
PYTHONPATH=/home/swank/spark-lab/collective-pdl/sparknet ../venv/bin/python -m unittest discover -s tests > "$out/deployment-tests.log" 2>&1
printf '0\n' > "$out/deployment-tests.exit"
python3 scripts/render_docs.py --check > "$out/generated-docs.log" 2>&1
printf '0\n' > "$out/generated-docs.exit"
git diff --check
