#!/bin/bash
set -eu
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
cd /home/swank/spark-lab/collective-pdl/deployment
out=/home/swank/spark-lab/collective-pdl/final-harness-checks
mkdir "$out"
PYTHONPATH=/home/swank/spark-lab/collective-pdl/sparknet ../venv/bin/python -m unittest discover -s tests > "$out/deployment-tests.log" 2>&1
printf '0\n' > "$out/deployment-tests.exit"
COLLECTIVE_PDL_MAP=config/nodes.json ../venv/bin/python experiments/2026-11-05-collective-pdl/test_source_gate_controls.py > "$out/controls.log" 2>&1
python3 /home/swank/spark-lab/collective-pdl/check-analysis.py > "$out/analysis.log" 2>&1
python3 scripts/render_docs.py --check > "$out/generated-docs.log" 2>&1
python3 - <<'PY'
import ast
from pathlib import Path
for path in Path('experiments/2026-11-05-collective-pdl').glob('*.py'): ast.parse(path.read_text())
PY
git diff --check
