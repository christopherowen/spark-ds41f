#!/bin/bash
# usage: overlay.sh   (on dgx1, deployment checkout)
# Applies vLLM patches 0001-0021 to the base in a throwaway worktree, the way
# bin/spark3 build prepare does, checks the tree, and copies patch 0021 runtime
# files to ~/spark3-overlay/display-kv on every node for mounting over r5j.
set -eu
cd ~/projects/spark3-vllm-ds41f
V=$PWD/.work/upstreams/vllm
O=~/spark3-overlay/display-kv
BASE=04c30fa98e7917fee0a24c739ea503ce1e22538d
T=$(mktemp -d /tmp/display-kv.XXXX)
trap "git -C $V worktree remove --force $T" EXIT
git -C "$V" worktree add -q --detach "$T" "$BASE"
for p in $(grep -v "^#" patches/vllm/series | grep -v "^$"); do
  git -C "$T" -c user.name="Christopher Owen" \
    -c user.email="3221756+christopherowen@users.noreply.github.com" \
    am --quiet --committer-date-is-author-date "$PWD/patches/vllm/$p"
done
[ "$(git -C "$T" rev-parse HEAD^{tree})" = 19270e209a0831efd8273379d0a77ae7134fb7af ]
rm -rf "$O"
for f in vllm/v1/worker/display_carveout.py vllm/v1/worker/gpu/model_runner.py; do
  mkdir -p "$O/$(dirname $f)"
  cp "$T/$f" "$O/$f"
done
for host in dgx2 dgx3; do
  ssh "$host" "rm -rf $O && mkdir -p ~/spark3-overlay"
  tar cf - -C ~/spark3-overlay display-kv | ssh "$host" "tar xf - -C ~/spark3-overlay"
done
echo "overlay written from tree 19270e20 to $O on dgx1-3"
