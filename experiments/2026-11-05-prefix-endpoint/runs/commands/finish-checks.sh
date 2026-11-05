#!/bin/bash
set -uo pipefail
area="$HOME/spark-lab/agents/prefix-endpoint"
cd "$area/deployment"
exp=experiments/2026-11-05-prefix-endpoint
for shape in tp3 tp4; do
 bin/spark --cluster-config "$exp/$shape.json" doctor > "$exp/runs/doctor-$shape-final.txt" 2>&1
 echo "$?" > "$exp/runs/doctor-$shape-final.exit"
done
bin/spark --cluster-config "$exp/tp3.json" build check > "$exp/runs/build-check-final.txt" 2>&1
echo "$?" > "$exp/runs/build-check-final.exit"
scripts/lab.py run "$exp/lab-tp3.json" --dry-run > "$exp/runs/lab-dry-run.txt" 2>&1
echo "$?" > "$exp/runs/lab-dry-run.exit"
cd "$area"
git -C vllm worktree add --detach fresh-final 1f33102547e4da5c64712390604695d1abd577c4 > runs/fresh-final.log 2>&1
git -C vllm/fresh-final -c user.name='Christopher Owen' -c user.email=3221756+christopherowen@users.noreply.github.com am --committer-date-is-author-date "$area/deployment/$exp/vllm/0001-retain-encoder-prompt-endpoints.patch" >> runs/fresh-final.log 2>&1
python3 - <<'PY'
import json, subprocess
from pathlib import Path
root=Path.cwd()
def git(ref):
 return subprocess.check_output(['git','-C','vllm/fresh-final','rev-parse',ref],text=True).strip()
r={'head':git('HEAD'),'tree':git('HEAD^{tree}'),'preceding_series_patches':572,'total_patches':573}
r['matches']=r['head']=='0aaebda5cc9fcc941973b43d62fd7b229508a371' and r['tree']=='5328d6b7279c789e824dd5a3800f1165f27ff1f7'
(root/'deployment/experiments/2026-11-05-prefix-endpoint/runs/fresh-final.json').write_text(json.dumps(r,indent=2)+'\n')
assert r['matches'],r
PY
cd "$area/deployment/$exp"
python3 -m unittest -v test_probe > runs/probe-tests-final.txt 2>&1
echo "$?" > runs/probe-tests-final.exit
python3 - <<'PY'
import ast
from pathlib import Path
for p in Path('.').glob('*.py'): ast.parse(p.read_text())
print('Python syntax passed')
PY
cd "$area/deployment"
git add "$exp"
git diff --cached --check > "$exp/runs/diff-check.txt" 2>&1
echo "$?" > "$exp/runs/diff-check.exit"
