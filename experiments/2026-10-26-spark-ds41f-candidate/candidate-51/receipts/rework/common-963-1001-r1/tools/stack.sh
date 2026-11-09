#!/usr/bin/env bash
# Candidates 51-53 stack on sp8: 1107's patch head, then candidate-50's patches without 0778
# (the base), then each candidate's 0781.. in series order. Every patch is checked with
# git apply --check --verbose (hunk offsets are listed) and applied with git am, as
# bin/spark build prepare applies it (committer Christopher Owen, committer date = author date).
#
# usage: stack.sh WORK EXP N LOG
#   WORK: directory holding stack-vllm (a --shared clone of the selection's base-vllm)
#   EXP:  a copy of experiments/2026-10-26-spark-ds41f-candidate with candidate-50/vllm and candidate-N/vllm
set -u
WORK=$1; EXP=$2; N=$3; LOG=$4
REPO=$WORK/stack-vllm
BASE_1107=2756478b60b90700b9918b457aeb38657da637b5
VLLM_PIN=19f2c20ed4d64ae22bd009ceacadd85e2a81e354
SELECTION_BASE_TREE=14c5a2dc924541ce4eda44d5b39d811be15f2363
AM=(git -c user.name="Christopher Owen" -c user.email=3221756+christopherowen@users.noreply.github.com am --quiet --committer-date-is-author-date)
SERIES=$EXP/candidate-$N/vllm/series

apply_one() {  # $1: series entry (relative to candidate-N/vllm)
  local p=$1 f sha out rc am files
  f=$(cd "$EXP/candidate-$N/vllm" && realpath "$p")
  sha=$(sha256sum "$f" | cut -c1-64)
  out=$(git apply --check --verbose "$f" 2>&1); rc=$?
  if [ $rc -ne 0 ]; then
    echo "FAIL  $(basename "$p") sha256=$sha  git apply --check rc=$rc" >> "$LOG"; echo "$out" | sed "s/^/      /" >> "$LOG"; return 1
  fi
  am=$("${AM[@]}" "$f" 2>&1); rc=$?
  if [ $rc -ne 0 ]; then
    echo "FAIL  $(basename "$p") sha256=$sha  git am rc=$rc" >> "$LOG"; echo "$am" | sed "s/^/      /" >> "$LOG"; git am --abort; return 1
  fi
  files=$(git diff --name-only HEAD~1 HEAD | wc -l)
  echo "OK    $(basename "$p") sha256=$sha  commit=$(git rev-parse HEAD) tree=$(git rev-parse HEAD^{tree}) files=$files" >> "$LOG"
  echo "$out" | grep -E "offset|fuzz" | sed "s/^/      git apply: /" >> "$LOG"
  [ -n "$am" ] && echo "$am" | sed "s/^/      git am: /" >> "$LOG"
  return 0
}

cd "$REPO" || exit 1
git am --abort >/dev/null 2>&1
git checkout -q --detach "$BASE_1107" || exit 1
git clean -qfdx
{
echo "## 1. The base: 1107's patch head, then candidate-50's 0762..0777, 0779, 0780 (0778, #1009, left out)"
echo
echo "base-1107 patch_head $(git rev-parse HEAD) tree $(git rev-parse HEAD^{tree})"
} >> "$LOG"
grep -v '^#' "$SERIES" | grep '^\.\./\.\./candidate-50/vllm/' | while read -r p; do
  apply_one "$p" || exit 1
done || exit 1
base_head=$(git rev-parse HEAD); base_tree=$(git rev-parse HEAD^{tree})
echo "base head $base_head tree $base_tree patches_after_1107 $(git rev-list --count $BASE_1107..HEAD)" >> "$LOG"
if [ "$base_tree" = "$SELECTION_BASE_TREE" ]; then
  echo "base tree equals the selection's base (candidate-50's prepared tree with 0778 reverted, commit 0bffb886, tree 14c5a2dc)" >> "$LOG"
else
  echo "base tree DIFFERS from the selection's base tree $SELECTION_BASE_TREE" >> "$LOG"; exit 1
fi
{
echo
echo "## 2. candidate-$N: its patches 0781.. in series order"
echo
} >> "$LOG"
grep -v '^#' "$SERIES" | grep -v '^\.\./' | while read -r p; do
  apply_one "$p" || exit 1
done || exit 1
echo "head $(git rev-parse HEAD) tree $(git rev-parse HEAD^{tree}) patches_after_base $(git rev-list --count $base_head..HEAD) patches_after_1107 $(git rev-list --count $BASE_1107..HEAD)" >> "$LOG"
if git diff --check "$VLLM_PIN" HEAD >/dev/null; then
  echo "git diff --check ${VLLM_PIN:0:12}..HEAD: clean" >> "$LOG"
else
  echo "git diff --check ${VLLM_PIN:0:12}..HEAD: PROBLEMS" >> "$LOG"
fi
git branch -f "candidate-$N" HEAD >/dev/null
git branch -f "base-c50-no1009" "$base_head" >/dev/null
exit 0
