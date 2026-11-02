#!/bin/bash
set -euo pipefail
W=/tmp/c37-ladder-739
S=$W/repo/experiments/2026-10-26-spark-ds41f-candidate/candidate-37/vllm/series
rm -rf $W/vllm
git init --quiet $W/vllm
cd $W/vllm
git remote add upstream https://github.com/local-inference-lab/vllm.git
# Borrow objects from an existing local clone to avoid a network fetch.
git fetch --quiet --no-tags /tmp/vllm-c37-tree 19f2c20ed4d64ae22bd009ceacadd85e2a81e354 || git fetch --quiet --filter=blob:none --no-tags upstream 19f2c20ed4d64ae22bd009ceacadd85e2a81e354
git switch --quiet --create spark/build FETCH_HEAD
while read -r p; do
  case "$p" in ''|'#'*) continue;; esac
  git -c user.name="Christopher Owen" -c user.email="3221756+christopherowen@users.noreply.github.com" am --quiet --committer-date-is-author-date "$(dirname $S)/$p"
done < $S
echo head $(git rev-parse HEAD) tree $(git rev-parse HEAD^{tree}) count $(git rev-list --count 19f2c20ed4d64ae22bd009ceacadd85e2a81e354..HEAD)
git diff --check 19f2c20ed4d64ae22bd009ceacadd85e2a81e354 HEAD && echo diffcheck-ok
