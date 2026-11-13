#!/bin/bash
# snap.sh REF ARM : export the 26 files 0762..0769 touch, as of tree commit REF, to ovl/ARM
set -euo pipefail
W=$HOME/spark-lab/agents/candidate-47-rework-1005
cd $W/tree
rm -rf $W/ovl/$2; mkdir -p $W/ovl/$2
while read -r f; do
  mkdir -p $W/ovl/$2/$(dirname $f)
  git show "$1:$f" > $W/ovl/$2/$f
done < $W/changed-files.txt
echo "$2 = $(git rev-parse $1) tree $(git rev-parse $1^{tree})" | tee -a $W/ovl/arms.txt
(cd $W/ovl/$2 && find . -type f | sort | xargs sha256sum | cut -c1-12,65-) > $W/ovl/$2.sha
