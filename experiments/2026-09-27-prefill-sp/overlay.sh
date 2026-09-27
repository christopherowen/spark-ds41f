#!/bin/bash
# usage: overlay.sh   (on dgx1)
# Writes the files of the prefill sequence-parallelism commits (branch
# spark3/r5f-sp, on the r5f series) to ~/spark3-overlay/r5f-sp on every node.
set -eu
V=~/projects/spark3-vllm-ds41f/.work/upstreams/vllm
O=~/spark3-overlay/r5f-sp
FILES=$(git -C $V diff --name-only spark3/r5f-check spark3/r5f-sp)
rm -rf $O
for f in $FILES; do
  mkdir -p "$O/$(dirname $f)"
  git -C $V show "spark3/r5f-sp:$f" > "$O/$f"
done
for host in dgx2 dgx3; do
  ssh "$host" "rm -rf $O && mkdir -p ~/spark3-overlay"
  tar cf - -C ~/spark3-overlay r5f-sp | ssh "$host" "tar xf - -C ~/spark3-overlay"
done
