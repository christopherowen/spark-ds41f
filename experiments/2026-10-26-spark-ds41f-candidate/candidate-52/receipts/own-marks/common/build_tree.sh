#!/bin/bash
# build_tree.sh: candidate-52's vLLM tree on sp8, from its series, as bin/spark build prepare applies it.
# A --shared clone of the selection's base-vllm (it holds the pinned vLLM revision), detached at the pin,
# then every active entry of candidate-52/vllm/series applied in order with git am, committer Christopher
# Owen and committer date = author date (spark_ds41f/build.py PATCH_COMMITTER). The result must equal the
# lock: patch_head 51a16b66d087..., expected_tree 1116740c08aa...
set -euo pipefail
B=$HOME/spark-lab/agents/candidate-52-own-marks
REPO=$B/src/repo                                  # spark-ds41f at origin/main 83e594528e
SERIES=$REPO/experiments/2026-10-26-spark-ds41f-candidate/candidate-52/vllm/series
PIN=19f2c20ed4d64ae22bd009ceacadd85e2a81e354
WANT_HEAD=51a16b66d087686e79496163263311b25d848127
WANT_TREE=1116740c08aa1cf2cdeaa5b1fcf08dc93410ba56
C50_HEAD=ba7400d9b6715ea1be6fa98b1d9ccedf5fcb65f9   # candidate-50's patch_head (tree d5778ab5)
LOG=$B/out/build-tree.log
cd $B
echo "host=$(hostname) utc=$(date -u +%FT%TZ) repo=$(git -C $REPO rev-parse HEAD)" > $LOG
echo "series sha256 $(sha256sum < $SERIES | cut -c1-64)" >> $LOG
rm -rf vllm-c52
git clone -q --shared --no-checkout $HOME/spark-lab/agents/round-51-53-select/base-vllm vllm-c52
cd vllm-c52
git checkout -q --detach $PIN
n=0
while read -r p; do
  case "$p" in ''|'#'*) continue;; esac
  f=$(cd "$(dirname $SERIES)" && realpath "$p")
  git -c user.name="Christopher Owen" -c user.email=3221756+christopherowen@users.noreply.github.com \
      am --quiet --committer-date-is-author-date "$f"
  n=$((n+1))
done < $SERIES
head=$(git rev-parse HEAD); tree=$(git rev-parse HEAD^{tree})
echo "applied $n patches; head $head tree $tree" >> $LOG
[ "$head" = "$WANT_HEAD" ] && echo "patch_head equals the lock ($WANT_HEAD)" >> $LOG || echo "patch_head DIFFERS from the lock $WANT_HEAD" >> $LOG
[ "$tree" = "$WANT_TREE" ] && echo "expected_tree equals the lock ($WANT_TREE)" >> $LOG || echo "expected_tree DIFFERS from the lock $WANT_TREE" >> $LOG
git diff --check $PIN HEAD && echo "git diff --check ${PIN:0:12}..HEAD: clean" >> $LOG
echo "candidate-50 head $C50_HEAD tree $(git rev-parse $C50_HEAD^{tree}); files differing from candidate-52:" >> $LOG
git diff --name-status $C50_HEAD HEAD >> $LOG
git diff --name-only $C50_HEAD HEAD > $B/changed-files.txt
cat $LOG
