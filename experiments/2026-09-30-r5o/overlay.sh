#!/bin/bash
# usage: overlay.sh   (on dgx1, deployment checkout)
# Applies the B12X series with 0005 (r5n's 0001-0004 plus the three-kernel
# fence) in a throwaway worktree, checks tree 1a8b9401, and copies the three
# files 0005 changes to every node: ~/spark3-overlay/r5o-fence.
set -eu
cd ~/projects/spark3-vllm-ds41f
B=$PWD/.work/upstreams/b12x
O=~/spark3-overlay/r5o-fence
FILES="b12x/attention/_shared/contiguous/forward.py b12x/gemm/bf16_gemv/_prefill.py b12x/norm/mhc/_kernels.py"
T=$(mktemp -d /tmp/r5o-fence.XXXX)
trap "git -C $B worktree remove --force $T" EXIT
git -C "$B" worktree add -q --detach "$T" f8069b2c0be1311df3b112591c6b8876a843f8be
for p in $(grep -v "^#" patches/b12x/series | grep -v "^$"); do
  git -C "$T" -c user.name="Christopher Owen" \
    -c user.email="3221756+christopherowen@users.noreply.github.com" \
    am --quiet --committer-date-is-author-date "$PWD/patches/b12x/$p"
done
[ "$(git -C "$T" rev-parse HEAD^{tree})" = 1a8b9401584ada0372939df49e658c3dbeae7658 ]
rm -rf "$O"
for f in $FILES; do mkdir -p "$O/$(dirname $f)"; cp "$T/$f" "$O/$f"; done
for host in dgx2 dgx3; do
  ssh "$host" "rm -rf $O && mkdir -p ~/spark3-overlay"
  tar cf - -C ~/spark3-overlay r5o-fence | ssh "$host" "tar xf - -C ~/spark3-overlay"
done
for host in dgx1 dgx2 dgx3; do ssh "$host" "cd $O && sha256sum $FILES | sed \"s/^/\$(hostname) /\""; done
echo "r5o-fence overlay written from B12X tree 1a8b9401 on dgx1-3"
