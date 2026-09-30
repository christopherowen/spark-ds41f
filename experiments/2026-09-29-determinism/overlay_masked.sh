#!/bin/bash
# usage: overlay_masked.sh   (on dgx1, deployment checkout)
# Applies the B12X series (r5m tree 35299956), then this experiment's 0004,
# 0005, 0006 and 0008 in a throwaway worktree, checks the tree, and copies the
# patched MoE modules and tests to every node: ~/spark3-overlay/det-masked.
set -eu
cd ~/projects/spark3-vllm-ds41f
E=$PWD/experiments/2026-09-29-determinism
B=$PWD/.work/upstreams/b12x
O=~/spark3-overlay/det-masked
FILES="b12x/moe/fused_moe/_impl.py b12x/moe/fused_moe/_preparation.py b12x/moe/fused_moe/_tuning.py \
b12x/moe/_shared/kernels/dynamic.py b12x/moe/_shared/kernels/silu.py b12x/moe/_shared/kernels/w4a16/kernel.py \
tests/moe/test_w4a8_dynamic_kernel.py tests/preparation/test_tuning_predicates.py"
T=$(mktemp -d /tmp/det-masked.XXXX)
trap "git -C $B worktree remove --force $T" EXIT
git -C "$B" worktree add -q --detach "$T" f8069b2c0be1311df3b112591c6b8876a843f8be
apply() {
  git -C "$T" -c user.name="Christopher Owen" \
    -c user.email="3221756+christopherowen@users.noreply.github.com" \
    am --quiet --committer-date-is-author-date "$1"
}
for p in $(grep -v "^#" patches/b12x/series | grep -v "^$"); do
  case $p in 0004-gemm-fence*) continue ;; esac   # r5n's dense GEMM fence is not in this stack
  apply "$PWD/patches/b12x/$p"
done
[ "$(git -C "$T" rev-parse HEAD^{tree})" = 35299956ce4954b71a7bc3b0529e3c6417fa975f ]
for p in 0004-moe-deterministic-planning 0005-moe-deterministic-decode 0006-moe-deterministic-slice-partials 0008-moe-masked-slice-topk-sum; do
  apply "$E/$p.patch"
done
[ "$(git -C "$T" rev-parse HEAD^{tree})" = c28e70fd2475c183f31e8d9f23e494a8d676a326 ]
rm -rf "$O"
for f in $FILES; do
  mkdir -p "$O/$(dirname $f)"
  cp "$T/$f" "$O/$f"
done
for host in dgx2 dgx3; do
  ssh "$host" "rm -rf $O && mkdir -p ~/spark3-overlay"
  tar cf - -C ~/spark3-overlay det-masked | ssh "$host" "tar xf - -C ~/spark3-overlay"
done
for host in dgx1 dgx2 dgx3; do
  ssh "$host" "cd $O && sha256sum $FILES | sha256sum | sed \"s/^/\$(hostname) /\""
done
echo "det-masked overlay written from B12X tree c28e70fd on dgx1-3"
