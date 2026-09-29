#!/bin/bash
# usage: overlay.sh   (on dgx1, deployment checkout)
# Applies the B12X series (r5m tree 35299956) and this experiment's 0004 in a
# throwaway worktree, checks both trees, and copies the three patched MoE
# planning modules to ~/spark3-overlay/det-planning on every node.
set -eu
cd ~/projects/spark3-vllm-ds41f
E=$PWD/experiments/2026-09-29-determinism
B=$PWD/.work/upstreams/b12x
O=~/spark3-overlay/det-planning
FILES="b12x/moe/fused_moe/_impl.py b12x/moe/fused_moe/_preparation.py b12x/moe/fused_moe/_tuning.py"
BASE=f8069b2c0be1311df3b112591c6b8876a843f8be
T=$(mktemp -d /tmp/det-planning.XXXX)
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
[ "$(git -C "$T" rev-parse HEAD^{tree})" = 35299956ce4954b71a7bc3b0529e3c6417fa975f ]
apply "$E/0004-moe-deterministic-planning.patch"
[ "$(git -C "$T" rev-parse HEAD^{tree})" = 66d62dc5f8894717540803cf34153c5ec48e26ab ]
rm -rf "$O"
for f in $FILES; do
  mkdir -p "$O/$(dirname $f)"
  cp "$T/$f" "$O/$f"
done
for host in dgx2 dgx3; do
  ssh "$host" "rm -rf $O && mkdir -p ~/spark3-overlay"
  tar cf - -C ~/spark3-overlay det-planning | ssh "$host" "tar xf - -C ~/spark3-overlay"
done
for host in dgx1 dgx2 dgx3; do
  ssh "$host" "cd $O && sha256sum $FILES"
done
echo "overlay written from B12X tree 66d62dc5 to $O on dgx1-3"
