#!/bin/bash
# Final patches: replay the rework commits with their round-1 messages (author and author
# date kept), check the tree is the one measured, format-patch them, then apply candidate-51's
# series with the corrected 0782 and 0784 to a fresh tree with stack.sh.
set -e
W=$HOME/spark-lab/agents/candidate-51-rework-963-1001; cd $W/stack-vllm
G=(git -c user.name="Christopher Owen" -c user.email=3221756+christopherowen@users.noreply.github.com)
MEASURED_TREE=35cd2a0e87a6b7a0c61c6c81dbe875ac1e7da005
"${G[@]}" checkout -q -B rw-final 48c7ed80f
"${G[@]}" cherry-pick d5143a355 >/dev/null; "${G[@]}" commit -q --amend -F $W/edit/msg-963-r1.txt
"${G[@]}" cherry-pick 63e2b4750 >/dev/null
"${G[@]}" cherry-pick f9a11e3fc >/dev/null; "${G[@]}" commit -q --amend -F $W/edit/msg-1001-r1.txt
t=$(git rev-parse HEAD^{tree}); echo "final head $(git rev-parse HEAD) tree $t"; [ "$t" = "$MEASURED_TREE" ] || { echo "TREE DIFFERS"; exit 1; }
git log --format='%h %an <%ae> %ad | %s' --date=rfc 48c7ed80f..HEAD
rm -rf $W/patches-r1; mkdir -p $W/patches-r1
# one patch each, as the series carries them ("[PATCH]", file named for the series entry)
git format-patch -q -1 --stdout HEAD~2 > $W/patches-r1/0782-perf-dsv41-unconditional-pack12-plane-loads-and-swar.patch
git format-patch -q -1 --stdout HEAD > $W/patches-r1/0784-Match-128-row-vocab-head-tile-pipeline-stages.patch
V=$W/verify-apply/exp/candidate-51/vllm
cp $W/patches-r1/0782-*.patch $W/patches-r1/0784-*.patch $V/
sha256sum $W/patches-r1/*.patch
rm -f $W/logs/stack-51-r1.log
bash $W/tools/stack.sh $W/verify-apply $W/verify-apply/exp 51 $W/logs/stack-51-r1.log
tail -10 $W/logs/stack-51-r1.log
t2=$(git -C $W/verify-apply/stack-vllm rev-parse HEAD^{tree}); echo "series-applied tree $t2"; [ "$t2" = "$MEASURED_TREE" ] && echo "series-applied tree equals the measured tree"
