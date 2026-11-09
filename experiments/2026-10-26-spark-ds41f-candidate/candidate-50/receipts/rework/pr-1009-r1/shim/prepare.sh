#!/usr/bin/env bash
# Prepare candidate-50 vLLM tree as bin/spark build prepare does: fetch base, git am the series.
set -euo pipefail
B=$HOME/spark-lab/agents/candidate-50-rework-1009
T=$B/vllm-c50
SER=$B/repo/experiments/2026-10-26-spark-ds41f-candidate/candidate-50/vllm/series
rm -rf $T; mkdir -p $T; cd $T
git init --quiet
git remote add upstream https://github.com/local-inference-lab/vllm.git
# reuse local objects from an existing prepared tree (same base) to avoid a full network fetch
REF=$(ls -d $HOME/projects/spark-ds41f/.work/build/*/src/vllm | head -1)
git fetch --quiet --filter=blob:none --no-tags upstream 19f2c20ed4d64ae22bd009ceacadd85e2a81e354
git switch --quiet --create spark/build FETCH_HEAD
n=0
grep -v "^\s*#" $SER | sed "/^\s*$/d" | while read -r p; do
  git -c user.name="Christopher Owen" -c user.email="3221756+christopherowen@users.noreply.github.com" am --quiet --committer-date-is-author-date "$(dirname $SER)/$p"
done
git diff --check 19f2c20ed4d64ae22bd009ceacadd85e2a81e354 HEAD
echo "patch_head $(git rev-parse HEAD)"
echo "tree $(git rev-parse HEAD^{tree})"
git rev-list --count 19f2c20ed4d64ae22bd009ceacadd85e2a81e354..HEAD
