#!/bin/bash
# usage: overlay.sh   (on dgx1, deployment checkout)
# Prepares vLLM with patches 0001-0021 and writes patch 0021's runtime files to
# ~/spark3-overlay/display-kv on every node for mounting over the r5j image.
set -eu
cd ~/projects/spark3-vllm-ds41f
V=.work/upstreams/vllm
O=~/spark3-overlay/display-kv
bin/spark3 upstream prepare vllm >/dev/null
# prepare applies the series to the working tree without committing it.
git -C $V add -A
[ "$(git -C $V write-tree)" = 52e9d1a1f1ceec1a6e25ec41664d526737100b04 ]
rm -rf $O
for f in vllm/v1/worker/utils.py vllm/v1/worker/display_carveout.py; do
  mkdir -p "$O/$(dirname $f)"
  cp "$V/$f" "$O/$f"
done
for host in dgx2 dgx3; do
  ssh "$host" "rm -rf $O && mkdir -p ~/spark3-overlay"
  tar cf - -C ~/spark3-overlay display-kv | ssh "$host" "tar xf - -C ~/spark3-overlay"
done
echo "overlay written from tree 52e9d1a1 to $O on dgx1-3"
