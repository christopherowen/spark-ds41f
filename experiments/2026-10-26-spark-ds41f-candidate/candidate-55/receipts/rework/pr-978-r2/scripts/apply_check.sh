#!/usr/bin/env bash
# Apply and pair checks for the round-2 0793 on candidate-55's series (git only, sp8).
# series/0792, series/0794: candidate-55's patch files; patch/0793-*: the round-2 patch;
# r1/0793-*: round 1's corrected patch; series/0793-*: candidate-55's original.
set -euo pipefail
WD=$HOME/spark-lab/agents/candidate-55-rework-978-r2
R1=$HOME/spark-lab/agents/candidate-55-rework-978
P92=$(ls $R1/series/0792-*.patch); P94=$(ls $R1/series/0794-*.patch)
P93_ORIG=$(ls $R1/series/0793-*.patch); P93_R1=$(ls $R1/patch/0793-*.patch)
P93=$(ls $WD/patch/0793-*.patch)
BASE=4e8133ff10a125b90753c2727f162e29fe37f30c   # candidate-54's vLLM head
rm -rf $WD/applycheck && git -C $WD/vllm worktree prune
git -C $WD/vllm worktree add -q --detach $WD/applycheck $BASE
cd $WD/applycheck
am() {  # git am with the stack's fixed committer and the patches' author dates
  git -c user.name="Christopher Owen" -c user.email="3221756+christopherowen@users.noreply.github.com" \
    am -q --committer-date-is-author-date "$@"
}
tree() { git rev-parse HEAD^{tree}; }
echo "patch sha256 $(sha256sum "$P93" | cut -d' ' -f1)"
echo "base $(git rev-parse HEAD) tree $(tree)"
echo "== series order 0792, 0793-r2, 0794"
git checkout -q --detach $BASE; am "$P92"; echo "0792 commit $(git rev-parse HEAD) tree $(tree)"
am "$P93"; echo "0793-r2 commit $(git rev-parse HEAD) tree $(tree)"
git diff --check HEAD~1 HEAD && echo "0793-r2 diff --check clean"
git log -1 --format='0793-r2 author %an <%ae>, date %ad' HEAD
am "$P94"; echo "0794 commit $(git rev-parse HEAD) tree $(tree)"
SERIES_TREE=$(tree); SERIES_HEAD=$(git rev-parse HEAD)
echo "== candidate-55 as locked (0792, 0793, 0794) and with round 1's 0793"
git checkout -q --detach $BASE; am "$P92" "$P93_ORIG" "$P94"; echo "c55 head $(git rev-parse HEAD) tree $(tree)"
C55=$(git rev-parse HEAD)
git checkout -q --detach $BASE; am "$P92" "$P93_R1" "$P94"; echo "c55-r1 head $(git rev-parse HEAD) tree $(tree)"
echo "== 0793-r2 alone on candidate-54"
git checkout -q --detach $BASE; am "$P93"; echo "tree $(tree)"
echo "== pair 0792 / 0793-r2, both orders"
git checkout -q --detach $BASE; am "$P93" "$P92"; T1=$(tree); echo "0793-r2 then 0792: tree $T1"
git checkout -q --detach $BASE; am "$P92" "$P93"; T2=$(tree); echo "0792 then 0793-r2: tree $T2"
[[ $T1 == $T2 ]] && echo "0792/0793-r2: same tree in both orders"
echo "== pair 0793-r2 / 0794, both orders"
git checkout -q --detach $BASE; am "$P93" "$P94"; T3=$(tree)
git checkout -q --detach $BASE; am "$P94" "$P93"; T4=$(tree)
echo "trees $T3 $T4"; [[ $T3 == $T4 ]] && echo "0793-r2/0794: same tree in both orders"
echo "== series head $SERIES_HEAD tree $SERIES_TREE"
echo "== files 0792 and 0793-r2 share"
comm -12 <(git -C $WD/vllm diff --name-only $BASE 7e36cafc16cf07163d8ba5db2aaa65b7ed90ea65 | sort) \
         <(grep '^diff --git' "$P93" | sed 's|diff --git a/\([^ ]*\) .*|\1|' | sort)
echo "== core.py hunks of 0792 and of 0793-r2 (on 0792)"
git checkout -q --detach $BASE; am "$P92"
git diff -U0 $BASE HEAD -- vllm/v1/engine/core.py | grep '^@@' | sed 's/^/  0792 /'
am "$P93"
git diff -U0 HEAD~1 HEAD -- vllm/v1/engine/core.py | grep '^@@' | sed 's/^/  0793-r2 /'
echo "== files the series head changes against candidate-55 (the overlay)"
git diff --name-only $C55 $SERIES_HEAD
cd $WD && git -C $WD/vllm worktree remove --force $WD/applycheck
