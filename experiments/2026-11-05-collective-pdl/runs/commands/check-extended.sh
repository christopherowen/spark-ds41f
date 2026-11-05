#!/bin/bash
set -eu
export PYTHONPATH=/home/swank/spark-lab/collective-pdl/sparknet
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
cd /home/swank/spark-lab/collective-pdl/deployment
exp=experiments/2026-11-05-collective-pdl
COLLECTIVE_PDL_MAP=config/nodes.json ../venv/bin/python "$exp/test_source_gate_controls.py" > "$exp/runs/source-gate-controls-final.log" 2>&1
for phase in entry staged arrived; do
 python3 "$exp/run-source-gates.py" --map config/nodes.json --out /home/swank/spark-lab/collective-pdl/gpu-unrun --phase "$phase" > "$exp/runs/plan-$phase.json"
done
for profile in tp3-triangle tp4-ring; do
 bin/spark --cluster-config "$exp/$profile.json" doctor > "$exp/runs/doctor-current-$profile.log" 2>&1
done
bin/spark --cluster-config "$exp/tp3-triangle.json" build check > "$exp/runs/build-check-current.log" 2>&1
python3 - <<'PY'
from pathlib import Path
import ast,hashlib,json
root=Path('experiments/2026-11-05-collective-pdl')
for path in root.glob('*.py'): ast.parse(path.read_text(),filename=str(path))
for name,row in json.loads((root/'receipt.json').read_text())['files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==row['sha256'],name
print('Python syntax and all original receipt hashes verified')
PY
git diff --check
