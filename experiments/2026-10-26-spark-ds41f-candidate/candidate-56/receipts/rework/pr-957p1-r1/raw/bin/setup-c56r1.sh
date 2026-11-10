#!/bin/bash
# The testbed overlay (and the c56r1 test arm on sp8): candidate-54 + 0795-r1 + 0796, the
# 8 non-test files 0795-r1 and 0796 change, plus 0796's and 0795-r1's 4 test files for the
# sp8 test run. Mounted over spark-ds41f:candidate-56 they make candidate-56 = candidate-54
# + 0795-r1 + 0796. No GPU.
set -euo pipefail
W=$HOME/spark-lab/agents/candidate-56-rework-957p1
R=$W/vllm
C54=4e8133ff10a125b90753c2727f162e29fe37f30c
am() { git -C $R -c user.name="Christopher Owen" -c user.email=3221756+christopherowen@users.noreply.github.com am --quiet --committer-date-is-author-date "$1"; }
git -C $R branch -f c56r1 rework-957p1-r1
git -C $R checkout -q c56r1
am $W/in/0796-perf-dsv41-sp-ratio2-compressor-and-ced-encoder-only-chunks.patch
c=$(git -C $R rev-parse HEAD)
git -C $R checkout -q rework-957p1-r1
echo "c56r1: $c tree $(git -C $R rev-parse $c^{tree})"
git -C $R diff --name-only $C54 $c > $W/files-c56r1.txt
rm -rf $W/ovl/c56r1
while read -r f; do mkdir -p $W/ovl/c56r1/$(dirname $f); git -C $R show $c:$f > $W/ovl/c56r1/$f; done < $W/files-c56r1.txt
git -C $R ls-tree -r $c > $W/lists/c56r1.txt
docker image inspect spark-ds41f:candidate-56 --format '{{.Id}}' > $W/image-id-c56
( cd $W/ovl && find c56r1 -type f | sort | xargs sha256sum ) > $W/overlay-sha-c56r1.txt
cat $W/files-c56r1.txt
