#!/bin/bash
# verify_image_tree.sh: the image's /opt/spark-ds41f/candidate/vllm against candidate-50's git tree.
# docker cp the image's tree out, then compare every tracked path of ba7400d9 (tree d5778ab5) with the
# copy through a temporary index (git diff-index over the copy as work tree). Untracked files in the copy
# (build outputs) are listed by count only. Then the 13 candidate paths are checked one by one.
set -euo pipefail
B=$HOME/spark-lab/agents/candidate-52-own-marks
IMG=$(cat $B/image-id)
C50=ba7400d9b6715ea1be6fa98b1d9ccedf5fcb65f9
cd $B
if [ ! -d vllm-c50img ]; then
  cid=$(docker create $IMG)
  docker cp $cid:/opt/spark-ds41f/candidate/vllm $B/vllm-c50img >/dev/null
  docker rm $cid >/dev/null
fi
echo "host=$(hostname) image=$IMG utc=$(date -u +%FT%TZ)"
export GIT_DIR=$B/vllm-c52/.git GIT_WORK_TREE=$B/vllm-c50img GIT_INDEX_FILE=$B/.idx-c50
git read-tree $C50
git update-index -q --refresh >/dev/null || true
echo "tracked paths in $C50: $(git ls-files | wc -l)"
echo "paths whose content in the image differs from $C50 (git diff-index):"
git diff-index --name-status $C50 | tee $B/out/image-vs-c50.txt
echo "count: $(wc -l < $B/out/image-vs-c50.txt)"
echo "untracked files in the image copy: $(git ls-files --others | wc -l) (e.g. $(git ls-files --others | grep -v __pycache__ | head -3 | tr '\n' ' '))"
rm -f $B/.idx-c50
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE
echo "candidate paths, image SHA-256 vs candidate-50 git blob vs candidate-52:"
while read -r f; do
  i=$(sha256sum < vllm-c50img/$f | cut -c1-12)
  g=$(git -C vllm-c52 show $C50:$f | sha256sum | cut -c1-12)
  c=$(sha256sum < ovl/c52/$f | cut -c1-12)
  echo "$i $g $c $f"
done < changed-files.txt
echo "__pycache__ for candidate paths in the image:"
while read -r f; do d=$(dirname $f); b=$(basename $f .py); ls vllm-c50img/$d/__pycache__/$b.* 2>/dev/null || true; done < changed-files.txt
echo "prebuilt Engram reader in the image:"
docker run --rm --entrypoint bash $IMG -c 'ls -la /opt/spark-ds41f/native/engram_rows/*/'
