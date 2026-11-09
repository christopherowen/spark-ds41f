#!/bin/bash
# Collect the receipts from sp8 into the deliverable tree (run on the Mac; read-only on sp8).
set -e
S=/private/tmp/claude-501/-Users-christopherowen-projects-spark-ds41f/d7a6b002-8e73-4152-857a-2e58be7a2871/scratchpad
D=$S/rework-963-1001-r1; RW=$S/rw; W=spark-lab/agents/candidate-51-rework-963-1001
rsync -a sp8:$W/out/ $RW/pull/out/
mkdir -p $D/pr-963/raw/json $D/pr-963/raw/diag/json $D/pr-1001/raw/json $D/pr-1001/raw/diag/json $D/common/verify $D/common/logs
# pr-963
cp $RW/pull/out/pr-963/pack12-*.txt $RW/pull/out/pr-963/pytest-*.txt $D/pr-963/raw/
cp $RW/pull/out/pr-963/aa-DE-*.txt $D/pr-963/raw/ 2>/dev/null || true
cp $RW/pull/out/pr-963/json/*.json $D/pr-963/raw/json/
cp $RW/pull/out/pr-963/commands.txt $D/pr-963/raw/commands.txt
for s in gemm vocab screen; do cp $RW/pull/out/pr-diag/d963-$s-boot*.txt $D/pr-963/raw/diag/ 2>/dev/null || true; cp $RW/pull/out/pr-diag/json/d963-$s-boot*.json $D/pr-963/raw/diag/json/ 2>/dev/null || true; done
# pr-1001
cp $RW/pull/out/pr-1001/vocab-*.txt $RW/pull/out/pr-1001/pytest-*.txt $D/pr-1001/raw/
cp $RW/pull/out/pr-1001/json/*.json $D/pr-1001/raw/json/
cp $RW/pull/out/pr-1001/commands.txt $D/pr-1001/raw/commands.txt
for s in sconf sab; do cp $RW/pull/out/pr-diag/d963-$s-boot*.txt $D/pr-1001/raw/diag/ 2>/dev/null || true; cp $RW/pull/out/pr-diag/json/d963-$s-boot*.json $D/pr-1001/raw/diag/json/ 2>/dev/null || true; done
cp $RW/pull/out/pr-diag/commands.txt $D/common/logs/diag-commands.txt
# common
cp $RW/pull/out/pr-common/* $D/common/verify/
rsync -a sp8:$W/logs/ $D/common/logs/
rsync -a --exclude 'c51r1-tree.txt' --exclude 'base-14c5a2dc.txt' sp8:$W/h/ $D/common/h/
rsync -a sp8:$W/bin/ $D/common/bin/
cp $RW/h/*.diff $D/common/h/
rsync -a sp8:$W/tools/ $D/common/tools/
for f in files-base.txt files-c51.txt files-c51r1.txt image-id c51r1-head; do scp -q sp8:$W/$f $D/common/; done
mkdir -p $D/common/analyze; cp $RW/analyze/*.py $D/common/analyze/
cp $RW/bin/mkpatches.sh $RW/bin/collect.sh $D/common/bin/
echo collected
