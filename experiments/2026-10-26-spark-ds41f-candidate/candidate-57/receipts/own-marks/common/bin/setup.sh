#!/bin/bash
# Round 3 own marks (candidates 55-57) on sp8: arm overlays and tree lists. No GPU.
set -euo pipefail
B=$HOME/spark-lab/agents/round3-own-marks
W=$HOME/spark-lab/agents/candidates-55-57/spark-ds41f/.work/build
T56=$W/vllm-19f2c20ed4d6-b12x-236ddff04a5f-41e7995b/src/vllm
T57=$W/vllm-19f2c20ed4d6-b12x-236ddff04a5f-fd68fd3b/src/vllm
T55=$W/vllm-19f2c20ed4d6-b12x-236ddff04a5f-f7aa2f2d/src/vllm
C54=4e8133ff10a125b90753c2727f162e29fe37f30c
cd $B
mkdir -p ovl lists
docker image inspect spark-ds41f:candidate-54 --format '{{.Id}}' > image-id
echo "image $(cat image-id) (spark-ds41f:candidate-54)"
# arm -> (tree repo, commit holding the patch, its parent)
mk() {  # arm repo commit parent
  local arm=$1 repo=$2 c=$3 p=$4
  mkdir -p ovl/$arm
  git -C $repo diff --name-only $p $c > files-$arm.txt
  while read -r f; do
    # the patch's files must be candidate-54's before the patch (patch applies alone on the base)
    if [ "$(git -C $repo rev-parse $p:$f 2>/dev/null)" != "$(git -C $repo rev-parse $C54:$f 2>/dev/null)" ]; then
      echo "FAIL $arm $f differs from candidate-54 before the patch"; exit 1
    fi
    mkdir -p ovl/$arm/$(dirname $f)
    git -C $repo show $c:$f > ovl/$arm/$f
  done < files-$arm.txt
  echo "$arm: $(wc -l < files-$arm.txt) files from $c (parent $p) in $repo"
}
mk c56p1 $T56 e4b134ab757ebfee9997c84bdfa061756724e9ca 4e8133ff10a125b90753c2727f162e29fe37f30c
mk c57p2 $T57 4fc8de115613ecbd112694260e461d7a612ba49e 4e8133ff10a125b90753c2727f162e29fe37f30c
mk c57p4 $T57 d74e38eefbda4004daaebb50dc7c45337b72df7a 4fc8de115613ecbd112694260e461d7a612ba49e
mk c55dr $T55 410463b3c58362448e6d440503b677e72e87153a 3feb241eaa077d6835b37ec8fd1af5ab617a375b
# expected trees: candidate-54, and candidate-54 + each patch alone
git -C $T56 ls-tree -r $C54 > lists/c54-5f021d9a.txt
for arm in c56p1 c57p2 c57p4 c55dr; do
  repo=$( [ $arm = c56p1 ] && echo $T56 || ( [ $arm = c55dr ] && echo $T55 || echo $T57 ) )
  cp lists/c54-5f021d9a.txt lists/$arm.txt
  python3 - "$repo" lists/$arm.txt files-$arm.txt <<'PY'
import subprocess, sys
repo, lst, files = sys.argv[1:]
commit = {"c56p1":"e4b134ab757ebfee9997c84bdfa061756724e9ca","c57p2":"4fc8de115613ecbd112694260e461d7a612ba49e","c57p4":"d74e38eefbda4004daaebb50dc7c45337b72df7a","c55dr":"410463b3c58362448e6d440503b677e72e87153a"}[lst.split("/")[-1][:-4]]
rows = {}
for line in open(lst):
    meta, path = line.rstrip("\n").split("\t", 1)
    rows[path] = meta
for f in open(files).read().split():
    out = subprocess.run(["git","-C",repo,"ls-tree",commit,f],capture_output=True,text=True,check=True).stdout.strip()
    meta, path = out.split("\t",1)
    rows[path] = meta
with open(lst,"w") as fh:
    for path in sorted(rows):
        fh.write(f"{rows[path]}\t{path}\n")
PY
done
( cd ovl && find . -type f | sort | xargs sha256sum ) > overlay-sha.txt
cat files-*.txt | wc -l
