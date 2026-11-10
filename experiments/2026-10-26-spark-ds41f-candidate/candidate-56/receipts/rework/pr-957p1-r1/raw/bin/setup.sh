#!/bin/bash
# #957 part 1 round 1 rework on sp8: the r1 arm's overlay and the expected trees. No GPU.
#   base = spark-ds41f:candidate-54 as built (tree 5f021d9a);
#   r1   = the same image + 0795-r1's files (candidate-54 + 0795-r1), read-only.
set -euo pipefail
W=$HOME/spark-lab/agents/candidate-56-rework-957p1
R=$W/vllm                       # c54 repo clone, branch rework-957p1-r1 = 4e8133ff + 0795-r1
C54=4e8133ff10a125b90753c2727f162e29fe37f30c
cd $W
mkdir -p ovl lists out
docker image inspect spark-ds41f:candidate-54 --format '{{.Id}}' > image-id
echo "image $(cat image-id) (spark-ds41f:candidate-54)"
c=$(git -C $R rev-parse rework-957p1-r1)
git -C $R diff --name-only $C54 $c > files-r1.txt
rm -rf ovl/r1
while read -r f; do mkdir -p ovl/r1/$(dirname $f); git -C $R show $c:$f > ovl/r1/$f; done < files-r1.txt
git -C $R ls-tree -r $C54 > lists/c54-5f021d9a.txt
git -C $R ls-tree -r $c > lists/r1.txt
echo "r1: $(wc -l < files-r1.txt) files from $c (tree $(git -C $R rev-parse $c^{tree}))"
( cd ovl && find . -type f | sort | xargs sha256sum ) > overlay-sha.txt
# TileLang/Triton caches: start from round 3's own-mark caches (the same kernels), copied.
for p in "base tl-base" "r1 tl-c56p1"; do set -- $p
  [ -d c/tl-$1 ] || { mkdir -p c; cp -a $HOME/spark-lab/agents/round3-own-marks/c/$2 c/tl-$1; }
done
mkdir -p h; cp -a $HOME/spark-lab/agents/round3-own-marks/h/{step957.py,omlib.py,verify_tree.py,tblib_stub.py} h/
sha256sum h/* > h-sha256.txt
