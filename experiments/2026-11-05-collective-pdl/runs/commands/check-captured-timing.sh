#!/bin/bash
set -eu
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
cd /home/swank/spark-lab/collective-pdl/deployment
out=/home/swank/spark-lab/collective-pdl/captured-timing-checks
mkdir "$out"
PYTHONPATH=/home/swank/spark-lab/collective-pdl/sparknet ../venv/bin/python -m unittest discover -s tests > "$out/deployment-tests.log" 2>&1
printf '0\n' > "$out/deployment-tests.exit"
COLLECTIVE_PDL_MAP=config/nodes.json ../venv/bin/python experiments/2026-11-05-collective-pdl/test_source_gate_controls.py > "$out/controls.log" 2>&1
python3 /home/swank/spark-lab/collective-pdl/check-analysis.py > "$out/analysis.log" 2>&1
python3 scripts/render_docs.py --check > "$out/generated-docs.log" 2>&1
python3 - <<'PY'
import ast,subprocess
from pathlib import Path
root=Path('experiments/2026-11-05-collective-pdl')
for path in root.glob('*.py'): ast.parse(path.read_text())
path=root/'benchmark_launch_chain.py'
previous=subprocess.check_output(['git','show','615ccb7ceb481244ca7b4bb40ce46f0ba44ab8c1:'+str(path)],text=True)
def kernel(text): return ast.dump(next(node for node in ast.parse(text).body if isinstance(node,ast.FunctionDef) and node.name=='copy_kernel'))
assert kernel(previous)==kernel(path.read_text())
print('Benchmark CUDA kernels unchanged; host counters only')
PY
for phase in entry staged arrived; do
 python3 experiments/2026-11-05-collective-pdl/run-source-gates.py --map config/nodes.json --out /home/swank/spark-lab/collective-pdl/gpu-unrun --phase "$phase" > "$out/plan-$phase.json"
done
git diff --check
