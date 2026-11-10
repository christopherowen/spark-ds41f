#!/usr/bin/env bash
# Apply check (CPU only): candidate-54 base alone, then the r4 956 patch by git am with the
# fixed committer and --committer-date-is-author-date as build.py does; git apply --check on
# the bare base; the tree against the tree the GPU runs used; git diff --check.
# usage: apply_check.sh <956 patch>
set -u
W=$HOME/spark-lab/agents/round3-956-r4
P956=$1
BASE=4e8133ff10a125b90753c2727f162e29fe37f30c
D=$W/applycheck
rm -rf $D
git clone -q --no-checkout $HOME/spark-lab/agents/round-55-57-select/c54-vllm $D
cd $D
git config gc.auto 0
git checkout -q -B check $BASE
echo "base: $(git rev-parse HEAD) tree $(git rev-parse HEAD^{tree})"
echo "956 r4 sha256: $(sha256sum $P956 | cut -d' ' -f1)"
git apply --check $P956 && echo "git apply --check on candidate-54: clean" || echo "git apply --check on candidate-54: FAILED"
git -c user.name="Christopher Owen" -c user.email=3221756+christopherowen@users.noreply.github.com am --quiet --committer-date-is-author-date $P956 \
  && echo "git am on candidate-54: ok, patch head $(git rev-parse HEAD), tree $(git rev-parse HEAD^{tree})" || { echo "git am: FAILED"; exit 1; }
git diff --check $BASE HEAD && echo "git diff --check base..HEAD: clean" || echo "git diff --check: NOT clean"
echo "tree of the GPU runs' cand arm: $(git -C $W/tree-cand rev-parse HEAD^{tree})"
git log -1 --format="author: %an <%ae> %ad%ncommitter: %cn <%ce> %cd" HEAD
git diff --stat $BASE HEAD | tail -1
echo APPLY_DONE
