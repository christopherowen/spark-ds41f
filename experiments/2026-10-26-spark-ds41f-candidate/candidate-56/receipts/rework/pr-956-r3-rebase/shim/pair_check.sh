#!/usr/bin/env bash
# Pair check (CPU only): candidate-54 base, 957-p1 (the split as proposed), then the
# rebased 956, both by git am with the fixed committer and --committer-date-is-author-date
# as build.py does; then each order of the pair by git apply --check, as the selection's
# pairs.py checks pairs; and the stack's tree against the tree the GPU runs used.
# usage: pair_check.sh <956 patch> <957-p1 patch>
set -u
W=$HOME/spark-lab/agents/round3-956
P956=$1; P957=$2
BASE=4e8133ff10a125b90753c2727f162e29fe37f30c
D=$W/pair
rm -rf $D
git clone -q --no-checkout $HOME/spark-lab/agents/round-55-57-select/c54-vllm $D
cd $D
git config gc.auto 0
git checkout -q -B pair $BASE
echo "base: $(git rev-parse HEAD) tree $(git rev-parse HEAD^{tree})"
echo "957-p1 sha256: $(sha256sum $P957 | cut -d' ' -f1)"
echo "956 sha256:    $(sha256sum $P956 | cut -d' ' -f1)"
am() { git -c user.name="Christopher Owen" -c user.email=3221756+christopherowen@users.noreply.github.com am --quiet --committer-date-is-author-date "$1"; }
am $P957 && echo "git am 957-p1: ok, tree $(git rev-parse HEAD^{tree})" || { echo "git am 957-p1: FAILED"; exit 1; }
am $P956 && echo "git am 956 after 957-p1: ok, patch head $(git rev-parse HEAD), tree $(git rev-parse HEAD^{tree})" || { echo "git am 956: FAILED"; exit 1; }
git diff --check $BASE HEAD && echo "git diff --check base..HEAD: clean" || echo "git diff --check: NOT clean"
echo "tree of the GPU runs' cand arm: $(git -C $W/tree-cand rev-parse HEAD^{tree})"
git log -1 --format="author: %an <%ae> %ad%ncommitter: %cn <%ce> %cd" HEAD
# pairs.py's two orders, on the base: apply one (git apply --index), check the other.
git reset -q --hard $BASE
git apply --index $P957 && { git apply --check $P956 && echo "956 after 957-p1 (git apply --check): clean" || echo "956 after 957-p1: conflict"; }
git reset -q --hard $BASE
if git apply --check $P956 2>/tmp/r3-956-alone.err; then
  echo "956 alone on candidate-54: applies"
  git apply --index $P956; git apply --check $P957 && echo "957-p1 after 956: clean" || echo "957-p1 after 956: conflict"
else
  echo "956 alone on candidate-54 (without 957-p1): does not apply: $(grep -o 'patch failed: [^ ]*' /tmp/r3-956-alone.err | sort -u | tr '\n' ' ')"
fi
git reset -q --hard $BASE
echo PAIR_DONE
