#!/bin/bash
# usage: overlay.sh   (on dgx1, deployment checkout)
# Applies B12X patches 0001-0002 and this experiment's 0003 to the pinned
# revision in a throwaway worktree, checks both trees, and copies the patched
# top-k module to ~/spark3-overlay/topk-ties on every node for mounting over r5l.
set -eu
cd ~/projects/spark3-vllm-ds41f
E=$PWD/experiments/2026-09-29-topk-ties
B=$PWD/.work/upstreams/b12x
O=~/spark3-overlay/topk-ties
F=b12x/attention/dsa_indexer/tiled_topk.py
BASE=f8069b2c0be1311df3b112591c6b8876a843f8be
T=$(mktemp -d /tmp/topk-ties.XXXX)
trap "git -C $B worktree remove --force $T" EXIT
git -C "$B" worktree add -q --detach "$T" "$BASE"
apply() {
  git -C "$T" -c user.name="Christopher Owen" \
    -c user.email="3221756+christopherowen@users.noreply.github.com" \
    am --quiet --committer-date-is-author-date "$1"
}
for p in $(grep -v "^#" patches/b12x/series | grep -v "^$"); do
  apply "$PWD/patches/b12x/$p"
done
[ "$(git -C "$T" rev-parse HEAD^{tree})" = 640c8544b3242e858c962ad6de98febd8f74e8e3 ]
apply "$E/0003-dsa-topk-position-ties.patch"
[ "$(git -C "$T" rev-parse HEAD^{tree})" = 35299956ce4954b71a7bc3b0529e3c6417fa975f ]
rm -rf "$O"
mkdir -p "$O/$(dirname $F)"
cp "$T/$F" "$O/$F"
for host in dgx2 dgx3; do
  ssh "$host" "rm -rf $O && mkdir -p ~/spark3-overlay"
  tar cf - -C ~/spark3-overlay topk-ties | ssh "$host" "tar xf - -C ~/spark3-overlay"
done
for host in dgx1 dgx2 dgx3; do
  ssh "$host" "sha256sum $O/$F"
done
echo "overlay written from B12X tree 35299956 to $O on dgx1-3"
