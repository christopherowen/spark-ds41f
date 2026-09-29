#!/bin/bash
# usage: overlay.sh   (on dgx1, deployment checkout)
# Applies vLLM patches 0001-0024 and this experiment's 0025 to the base in a
# throwaway worktree, checks both trees, and copies the patched attention
# module to ~/spark3-overlay/indexer-split on every node for mounting over r5k.
set -eu
cd ~/projects/spark3-vllm-ds41f
E=$PWD/experiments/2026-09-29-indexer-split
V=$PWD/.work/upstreams/vllm
O=~/spark3-overlay/indexer-split
F=vllm/models/deepseek_v4_1/attention.py
BASE=04c30fa98e7917fee0a24c739ea503ce1e22538d
T=$(mktemp -d /tmp/indexer-split.XXXX)
trap "git -C $V worktree remove --force $T" EXIT
git -C "$V" worktree add -q --detach "$T" "$BASE"
apply() {
  git -C "$T" -c user.name="Christopher Owen" \
    -c user.email="3221756+christopherowen@users.noreply.github.com" \
    am --quiet --committer-date-is-author-date "$1"
}
for p in $(grep -v "^#" patches/vllm/series | grep -v "^$"); do
  apply "$PWD/patches/vllm/$p"
done
[ "$(git -C "$T" rev-parse HEAD^{tree})" = 9ba14ba1c4b77fcfdb727784b3b869bd8e382c18 ]
apply "$E/0025-deepseek-v41-indexer-sp-split.patch"
[ "$(git -C "$T" rev-parse HEAD^{tree})" = d82a42e22239d34b6af71d61f1f8d05073aeda47 ]
rm -rf "$O"
mkdir -p "$O/$(dirname $F)"
cp "$T/$F" "$O/$F"
for host in dgx2 dgx3; do
  ssh "$host" "rm -rf $O && mkdir -p ~/spark3-overlay"
  tar cf - -C ~/spark3-overlay indexer-split | ssh "$host" "tar xf - -C ~/spark3-overlay"
done
for host in dgx1 dgx2 dgx3; do
  ssh "$host" "sha256sum $O/$F"
done
echo "overlay written from tree d82a42e2 to $O on dgx1-3"
