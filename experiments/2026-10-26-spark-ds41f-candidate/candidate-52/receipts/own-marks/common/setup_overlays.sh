#!/bin/bash
# setup_overlays.sh: the two arms' read-only overlays, from the trees build_tree.sh made.
#   ovl/no1009/<path>: candidate-50's no1009/overlay (4 files), as candidate-50/tp4-no1009.json mounts them.
#   ovl/c52/<path>:    every file of candidate-52's tree (51a16b66) that differs from candidate-50's (ba7400d9),
#                      13 files (changed-files.txt), the no1009 files included.
# overlay-sha.txt: per file, SHA-256 prefixes of candidate-50 (the image's tree), no1009 overlay, candidate-52.
set -euo pipefail
B=$HOME/spark-lab/agents/candidate-52-own-marks
REPO=$B/src/repo
NO1009=$REPO/experiments/2026-10-26-spark-ds41f-candidate/candidate-50/no1009/overlay
C50_HEAD=ba7400d9b6715ea1be6fa98b1d9ccedf5fcb65f9
cd $B
rm -rf ovl; mkdir -p ovl/c52 ovl/no1009
git -C vllm-c52 checkout -q --detach 51a16b66d087686e79496163263311b25d848127
(cd $NO1009 && find . -type f | sed 's#^\./##' | sort) > no1009-files.txt
while read -r f; do mkdir -p ovl/no1009/$(dirname $f); cp -a $NO1009/$f ovl/no1009/$f; done < no1009-files.txt
while read -r f; do mkdir -p ovl/c52/$(dirname $f); cp -a vllm-c52/$f ovl/c52/$f; done < changed-files.txt
{
  echo "# c50 = candidate-50 tree ba7400d9 (git), no1009 = candidate-50/no1009/overlay, c52 = candidate-52 tree 51a16b66"
  while read -r f; do
    c50=$(git -C vllm-c52 show $C50_HEAD:$f | sha256sum | cut -c1-12)
    if [ -e ovl/no1009/$f ]; then n=$(sha256sum < ovl/no1009/$f | cut -c1-12); else n=-; fi
    c52=$(sha256sum < ovl/c52/$f | cut -c1-12)
    echo "$c50 $n $c52 $f"
  done < changed-files.txt
} > overlay-sha.txt
cat overlay-sha.txt
# every no1009 file must be one of the candidate files and equal candidate-52's copy
while read -r f; do grep -qx "$f" changed-files.txt || { echo "no1009 file $f not in changed-files"; exit 1; }
  cmp -s ovl/no1009/$f ovl/c52/$f && echo "no1009 == c52: $f" || echo "no1009 != c52: $f"; done < no1009-files.txt
