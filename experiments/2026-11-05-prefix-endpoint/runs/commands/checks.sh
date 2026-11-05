#!/bin/bash
set -uo pipefail
cd "$HOME/spark-lab/agents/prefix-endpoint/deployment"
exp=experiments/2026-11-05-prefix-endpoint
cp config/nodes.example.json config/nodes.json
cp "$HOME/projects/spark-ds41f/config/nodes-ring4.local.json" config/nodes-ring4.local.json 2>/dev/null || true
for shape in tp3 tp4; do
 bin/spark --cluster-config "$exp/$shape.json" doctor > "$exp/runs/doctor-$shape.txt" 2>&1
 echo "$?" > "$exp/runs/doctor-$shape.exit"
done
bin/spark --cluster-config "$exp/tp3.json" build check > "$exp/runs/build-check.txt" 2>&1
echo "$?" > "$exp/runs/build-check.exit"
python3 - <<'PY'
import ast, pathlib
root=pathlib.Path('experiments/2026-11-05-prefix-endpoint')
for path in root.glob('*.py'): ast.parse(path.read_text())
print('Python syntax passed')
PY
git diff --check
