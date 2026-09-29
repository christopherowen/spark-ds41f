#!/bin/bash
# usage: overlay.sh   (on dgx1, deployment checkout)
# Applies the B12X series (r5m tree 35299956), then this experiment's 0004,
# 0005 and 0006, in a throwaway worktree, checks each tree, and copies the
# patched MoE modules to every node: ~/spark3-overlay/det-planning (0004),
# det-decode (0004-0005) and det-slices (0004-0006, with the kernel, its SiLU
# wrapper and the two patched test files).
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
write() {
  rm -rf "$1"
  for f in $FILES; do
    mkdir -p "$1/$(dirname $f)"
    cp "$T/$f" "$1/$f"
  done
  for host in dgx2 dgx3; do
    ssh "$host" "rm -rf $1 && mkdir -p ~/spark3-overlay"
    tar cf - -C ~/spark3-overlay "$(basename $1)" | ssh "$host" "tar xf - -C ~/spark3-overlay"
  done
  for host in dgx1 dgx2 dgx3; do
    ssh "$host" "cd $1 && sha256sum $FILES"
  done
}
write "$O"
apply "$E/0005-moe-deterministic-decode.patch"
[ "$(git -C "$T" rev-parse HEAD^{tree})" = f422b9110ed29c900e8c9e7254fed65ec3773024 ]
write ~/spark3-overlay/det-decode
apply "$E/0006-moe-deterministic-slice-partials.patch"
[ "$(git -C "$T" rev-parse HEAD^{tree})" = 44709de4b96dd6134cf14e9a9bcc838aa6c22d83 ]
FILES="$FILES b12x/moe/_shared/kernels/dynamic.py b12x/moe/_shared/kernels/silu.py
  tests/moe/test_w4a8_migration_corpus.py tests/preparation/test_tuning_predicates.py"
write ~/spark3-overlay/det-slices
echo "overlays written from B12X trees 66d62dc5 (det-planning), f422b911 (det-decode) and 44709de4 (det-slices) on dgx1-3"
