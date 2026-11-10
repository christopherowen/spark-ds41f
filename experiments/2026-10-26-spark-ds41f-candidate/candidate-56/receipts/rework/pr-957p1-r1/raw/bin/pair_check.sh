#!/usr/bin/env bash
# Pair check (CPU only), as round 3's #956 rebase ran it (round3-956/shim/pair_check.sh):
# candidate-54's head, 0795-r1, then 0796 (#956 round 3, unchanged), each by git am with the
# fixed committer and --committer-date-is-author-date as build.py does; then each order of the
# pair by git apply --check, as the selection's pairs.py checks pairs.
# usage: pair_check.sh <0795-r1 patch> <0796 patch>
set -u
W=$HOME/spark-lab/agents/candidate-56-rework-957p1
P1=$1; P2=$2
BASE=4e8133ff10a125b90753c2727f162e29fe37f30c
D=$W/pair
rm -rf $D
git clone -q --no-checkout $HOME/spark-lab/agents/round-55-57-select/c54-vllm $D
cd $D
git config gc.auto 0
git checkout -q -B pair $BASE
echo "base: $(git rev-parse HEAD) tree $(git rev-parse HEAD^{tree})"
echo "0795-r1 sha256: $(sha256sum $P1 | cut -d' ' -f1)"
echo "0796 sha256:    $(sha256sum $P2 | cut -d' ' -f1)"
am() { git -c user.name="Christopher Owen" -c user.email=3221756+christopherowen@users.noreply.github.com am --quiet --committer-date-is-author-date "$1"; }
am $P1 && echo "git am 0795-r1: ok, patch head $(git rev-parse HEAD), tree $(git rev-parse HEAD^{tree})" || { echo "git am 0795-r1: FAILED"; exit 1; }
git log -1 --format="  author: %an <%ae> %ad" HEAD
am $P2 && echo "git am 0796 after 0795-r1: ok, patch head $(git rev-parse HEAD), tree $(git rev-parse HEAD^{tree})" || { echo "git am 0796: FAILED"; exit 1; }
git diff --check $BASE HEAD && echo "git diff --check base..HEAD: clean" || echo "git diff --check: NOT clean"
echo "files 0795-r1 and 0796 both touch: $(comm -12 <(git diff --name-only $BASE HEAD~1 | sort) <(git diff --name-only HEAD~1 HEAD | sort) | tr '\n' ' ')"
R1=$(git rev-parse HEAD)
# candidate-56 as built: the original 0795, then 0796 (expected tree 0c848613).
git checkout -q -B c56 $BASE
am $W/in/0795-perf-dsv41-replicate-the-BF16-compressor-projection-.patch && am $P2 \
  && echo "candidate-56 (original 0795 + 0796): tree $(git rev-parse HEAD^{tree})"
echo "candidate-56 -> candidate-54 + 0795-r1 + 0796:"
git diff --stat HEAD $R1
git checkout -q pair
# pairs.py's two orders, on the base: apply one (git apply --index), check the other.
git reset -q --hard $BASE
git apply --index $P1 && { git apply --check $P2 && echo "0796 after 0795-r1 (git apply --check): clean" || echo "0796 after 0795-r1: conflict"; }
git reset -q --hard $BASE
if git apply --check $P2 2>/tmp/rw957r1-0796-alone.err; then
  echo "0796 alone on candidate-54: applies"
else
  echo "0796 alone on candidate-54 (without 0795-r1): does not apply, as designed: $(grep -o 'patch failed: [^ ]*' /tmp/rw957r1-0796-alone.err | sort -u | tr '\n' ' ')"
fi
git reset -q --hard $BASE
echo PAIR_DONE
