#!/bin/bash
# setup.sh (sp8, CPU only): the two arms' overlays over spark-ds41f:candidate-50.
#   base = candidate-54's tree (5f021d9a): every file that differs from the image's tree
#          (candidate-50, d5778ab5), mounted read-only at its tree path.
#   cand = the same file list from the guard commit's tree (candidate-54 + 0781 guard).
# Both changed files of the guard (tilelang_copy.py, test_copy_drafts.py) are in that list,
# so the two arms mount the same 36 paths and differ only in those two files' contents.
set -euo pipefail
W=$HOME/spark-lab/agents/round3-995-guard
cd $W
IMG=$(docker image inspect spark-ds41f:candidate-50 --format '{{.Id}}')
echo "$IMG" > image-id
echo "host=$(hostname) image=$IMG created=$(docker image inspect $IMG --format '{{.Created}}') utc=$(date -u +%FT%TZ)"
docker image inspect $IMG --format '{{json .Config.Labels}}' | tr ',' '\n' | grep -E 'vllm\.(tree|patch-head)'
C50=d5778ab56647a20d94710ecb94cfd9a4de27bee9
C54=5f021d9a7f01c03e2389dd3d8021759cfa19ea34
CAND=$(git -C vllm rev-parse guard-995^{tree})
echo "c50 tree $C50; c54 tree $C54; cand tree $CAND (commit $(git -C vllm rev-parse guard-995))"
git -C vllm diff --name-status $C50 $C54 > c50-c54-name-status.txt
if grep -v '^M' c50-c54-name-status.txt; then echo "non-modify entries: overlay cannot represent them"; exit 1; fi
cut -f2 c50-c54-name-status.txt > files.txt
git -C vllm diff --name-only $C54 $CAND > guard-files.txt
while read -r f; do grep -qxF "$f" files.txt || { echo "guard file $f not in the overlay list"; exit 1; }; done < guard-files.txt
rm -rf ovl; mkdir -p ovl/base ovl/cand
while read -r f; do
  mkdir -p ovl/base/$(dirname $f) ovl/cand/$(dirname $f)
  git -C vllm show $C54:$f > ovl/base/$f
  git -C vllm show $CAND:$f > ovl/cand/$f
done < files.txt
{
  echo "# columns: image (candidate-50 d5778ab5) | base (candidate-54 5f021d9a) | cand (guard $CAND) | path; SHA-256 prefixes"
  while read -r f; do
    s0=$(git -C vllm show $C50:$f | sha256sum | cut -c1-12)
    echo "$s0 $(sha256sum < ovl/base/$f | cut -c1-12) $(sha256sum < ovl/cand/$f | cut -c1-12) $f"
  done < files.txt
} > overlay-sha.txt
cat overlay-sha.txt
wc -l files.txt guard-files.txt
