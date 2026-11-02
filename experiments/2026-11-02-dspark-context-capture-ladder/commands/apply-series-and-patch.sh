#!/bin/bash
# Apply candidate-37's series and the experiment patch to the pinned base in a fresh tree.
set -euo pipefail
W=/tmp/c37-ladder-739
S=$W/repo/experiments/2026-10-26-spark-ds41f-candidate/candidate-37/vllm/series
P=$1
rm -rf $W/fresh
git init --quiet $W/fresh
cd $W/fresh
git remote add upstream https://github.com/local-inference-lab/vllm.git
git fetch --quiet --filter=blob:none --no-tags upstream 19f2c20ed4d64ae22bd009ceacadd85e2a81e354
git switch --quiet --create spark/build FETCH_HEAD
am() { git -c user.name="Christopher Owen" -c user.email="3221756+christopherowen@users.noreply.github.com" am --quiet --committer-date-is-author-date "$1"; }
while read -r p; do
  case "$p" in ''|'#'*) continue;; esac
  am "$(dirname $S)/$p"
done < $S
echo candidate-37 head $(git rev-parse HEAD) tree $(git rev-parse HEAD^{tree})
am "$P"
echo patched head $(git rev-parse HEAD) tree $(git rev-parse HEAD^{tree})
git diff --check 19f2c20ed4d64ae22bd009ceacadd85e2a81e354 HEAD && echo diff-check-ok
git diff --check HEAD~1 HEAD && echo patch-diff-check-ok
