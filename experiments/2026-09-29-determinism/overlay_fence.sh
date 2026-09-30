#!/bin/bash
# usage: overlay_fence.sh   (on dgx1, deployment checkout)
# Applies the B12X series (r5m tree 35299956) and this experiment's 0007 in a
# throwaway worktree, checks the tree, and copies the patched dense GEMM to
# every node: ~/spark3-overlay/gemm-fence/b12x/_lib/dense_gemm.py. 0004-0006
# do not touch that file, so it composes with the det-* overlays.
set -eu
cd ~/projects/spark3-vllm-ds41f
E=$PWD/experiments/2026-09-29-determinism
B=$PWD/.work/upstreams/b12x
O=~/spark3-overlay/gemm-fence
F=b12x/_lib/dense_gemm.py
T=$(mktemp -d /tmp/gemm-fence.XXXX)
trap "git -C $B worktree remove --force $T" EXIT
git -C "$B" worktree add -q --detach "$T" f8069b2c0be1311df3b112591c6b8876a843f8be
apply() {
  git -C "$T" -c user.name="Christopher Owen" \
    -c user.email="3221756+christopherowen@users.noreply.github.com" \
    am --quiet --committer-date-is-author-date "$1"
}
for p in $(grep -v "^#" patches/b12x/series | grep -v "^$"); do
  apply "$PWD/patches/b12x/$p"
done
[ "$(git -C "$T" rev-parse HEAD^{tree})" = 35299956ce4954b71a7bc3b0529e3c6417fa975f ]
apply "$E/0007-gemm-fence-stage-reads-before-tma-refill.patch"
[ "$(git -C "$T" rev-parse HEAD^{tree})" = 693aaed550673dcd86655ad4b2c3589882f0bd29 ]
rm -rf "$O" && mkdir -p "$O/$(dirname $F)" && cp "$T/$F" "$O/$F"
for host in dgx2 dgx3; do
  ssh "$host" "rm -rf $O && mkdir -p ~/spark3-overlay"
  tar cf - -C ~/spark3-overlay gemm-fence | ssh "$host" "tar xf - -C ~/spark3-overlay"
done
for host in dgx1 dgx2 dgx3; do
  ssh "$host" "cd $O && sha256sum $F"
done
echo "gemm-fence overlay written from B12X tree 693aaed5 on dgx1-3"
