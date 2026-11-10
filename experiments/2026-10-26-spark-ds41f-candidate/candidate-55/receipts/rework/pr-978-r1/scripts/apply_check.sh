#!/usr/bin/env bash
# Apply checks for the corrected 0793 on candidate-55's series (git only).
# series/0792, series/0794: candidate-55's patch files on main; patch/0793-r1: the corrected patch.
set -euo pipefail
WD=$HOME/spark-lab/agents/candidate-55-rework-978
P92=$(ls $WD/series/0792-*.patch); P93=$(ls $WD/patch/0793-*.patch); P94=$(ls $WD/series/0794-*.patch)
P93_ORIG=$(ls $WD/series/0793-*.patch)
BASE=4e8133ff10a125b90753c2727f162e29fe37f30c
rm -rf $WD/applycheck && git -C $WD/vllm worktree prune
git -C $WD/vllm worktree add -q --detach $WD/applycheck $BASE
cd $WD/applycheck
am() {  # am <patch>...: git am with the stack's fixed committer and author dates
  git -c user.name="Christopher Owen" -c user.email="3221756+christopherowen@users.noreply.github.com" \
    am -q --committer-date-is-author-date "$@"
}
tree() { git rev-parse HEAD^{tree}; }
echo "base $(git rev-parse HEAD) tree $(tree)"
echo "== series order 0792, 0793-r1, 0794"
git checkout -q --detach $BASE; am "$P92"; echo "0792 commit $(git rev-parse HEAD) tree $(tree)"
am "$P93"; echo "0793-r1 commit $(git rev-parse HEAD) tree $(tree)"; git diff --check HEAD~1 HEAD && echo "0793-r1 diff --check clean"
am "$P94"; echo "0794 commit $(git rev-parse HEAD) tree $(tree)"
SERIES_TREE=$(tree)
echo "== original series order 0792, 0793, 0794 (candidate-55 as locked)"
git checkout -q --detach $BASE; am "$P92" "$P93_ORIG" "$P94"; echo "head $(git rev-parse HEAD) tree $(tree)"
echo "== 0793-r1 alone on the base"
git checkout -q --detach $BASE; am "$P93"; echo "tree $(tree)"
echo "== pair 0793-r1 then 0792"
git checkout -q --detach $BASE; am "$P93" "$P92"; T1=$(tree); echo "tree $T1"
echo "== pair 0792 then 0793-r1"
git checkout -q --detach $BASE; am "$P92" "$P93"; T2=$(tree); echo "tree $T2"
[[ $T1 == $T2 ]] && echo "0792/0793-r1: same tree in both orders"
echo "== pair 0793-r1 then 0794, and 0794 then 0793-r1"
git checkout -q --detach $BASE; am "$P93" "$P94"; T3=$(tree)
git checkout -q --detach $BASE; am "$P94" "$P93"; T4=$(tree)
echo "trees $T3 $T4"; [[ $T3 == $T4 ]] && echo "0793-r1/0794: same tree in both orders"
echo "== series tree $SERIES_TREE"
echo "== files 0792 and 0793-r1 share"
comm -12 <(git -C $WD/vllm diff --name-only $BASE 7e36cafc16cf07163d8ba5db2aaa65b7ed90ea65 | sort) \
         <(grep '^diff --git' "$P93" | sed 's|diff --git a/\([^ ]*\) .*|\1|' | sort)
echo "== core.py functions each touches"
git checkout -q --detach $BASE; am "$P92" "$P93"
git -C $WD/applycheck diff -U0 $BASE HEAD -- vllm/v1/engine/core.py | grep '^@@' | sed 's/^/  /'
cd $WD && git -C $WD/vllm worktree remove --force $WD/applycheck
