#!/bin/bash
# setup_r1.sh ARM...  build overlay ARM from the 1107 tree's 26 candidate files.
#   no1016: 0762..0767 + 0769 (candidate-47 without #1016)
#   c47:    0762..0769 (candidate-47 as built)
#   r1:     0762..0767 + r1/0768 (corrected) + 0769
set -euo pipefail
B=$HOME/spark-lab/agents/candidate-47-rework-1016
cd $B
grep -h '^diff --git' patches/076*.patch | awk '{print substr($4,3)}' | sort -u > changed-files.txt
for arm in "$@"; do
  case $arm in
    no1016) series="0762 0763 0764 0765 0766 0767 0769" ;;
    c47) series="0762 0763 0764 0765 0766 0767 0768 0769" ;;
    r1) series="0762 0763 0764 0765 0766 0767 r1 0769" ;;
    *) echo "bad arm $arm"; exit 2 ;;
  esac
  rm -rf ovl/$arm; mkdir -p ovl/$arm
  while read -r f; do mkdir -p ovl/$arm/$(dirname $f); cp -a vllm-1107/$f ovl/$arm/$f; done < changed-files.txt
  ( cd ovl/$arm
    for s in $series; do
      if [ $s = r1 ]; then p=$B/r1/0768-perf-dsv41-align-qb-and-indexer-qb-tile-contiguous-decode.patch
      else p=$(ls $B/patches/$s-*.patch); fi
      git apply --check "$p" && git apply "$p" && echo "$arm: applied $(basename $p) sha256=$(sha256sum < $p | cut -c1-12)"
    done )
  while read -r f; do
    echo "$(sha256sum < vllm-1107/$f | cut -c1-12) $(sha256sum < ovl/$arm/$f | cut -c1-12) $f"
  done < changed-files.txt > overlay-sha-$arm.txt
  echo "$arm gemm.py: $(grep -E '^\s+\((16384|6144|2048), 1280\),?$' ovl/$arm/vllm/models/deepseek_v4_1/tilelang/gemm.py | wc -l) new-shape lines"
done
