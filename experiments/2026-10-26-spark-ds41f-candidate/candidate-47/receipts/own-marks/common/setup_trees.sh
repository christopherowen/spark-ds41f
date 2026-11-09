#!/bin/bash
set -euo pipefail
B=$HOME/spark-lab/agents/candidate-47-own-marks
cd $B
IMG=$(docker image inspect spark-ds41f:1107 --format '{{.Id}}')
echo "host=$(hostname) image=$IMG utc=$(date -u +%FT%TZ)"
if [ ! -d vllm-1107 ]; then
  cid=$(docker create $IMG)
  docker cp $cid:/opt/spark-ds41f/candidate/vllm $B/vllm-1107 >/dev/null
  docker rm $cid >/dev/null
fi
rm -rf vllm-c47; cp -a vllm-1107 vllm-c47
cd vllm-c47
for p in $B/patches/076*.patch; do
  git apply --check "$p" && git apply "$p" && echo "applied $(basename $p)"
done
cd $B
# changed files list (from patch headers), ordered, unique
grep -h '^diff --git' patches/076*.patch | awk '{print substr($4,3)}' | sort -u > changed-files.txt
wc -l < changed-files.txt
# build overlay dir with only candidate files
rm -rf ovl; mkdir -p ovl/c47
while read f; do
  if [ -e vllm-c47/$f ]; then mkdir -p ovl/c47/$(dirname $f); cp -a vllm-c47/$f ovl/c47/$f; else echo "DELETED $f"; fi
  if [ -e vllm-1107/$f ]; then s1=$(sha256sum < vllm-1107/$f | cut -c1-12); else s1=new; fi
  s2=$(sha256sum < vllm-c47/$f | cut -c1-12)
  echo "$s1 $s2 $f"
done < changed-files.txt > overlay-sha.txt
cat overlay-sha.txt
du -sh vllm-1107 vllm-c47 ovl
