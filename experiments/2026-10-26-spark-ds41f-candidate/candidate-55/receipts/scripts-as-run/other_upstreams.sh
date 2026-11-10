#!/bin/bash
# Run on sp8 in ~/spark-lab/agents/round-55-57-select. candidate-54 pins TileLang and sparknet exactly as 1107
# (its lock's tilelang and sparknet entries equal 1107's). Local clones of round 2's checked trees: TileLang
# 2123e412 + 1107/tilelang/series (13 patches, tip 3b0c7814, tree e2b00182) and sparknet 1fd6a1fb (no series,
# tree 8cc28a5c). Then each pool PR's TileLang / sparknet patch alone (git apply --check, then git am with a
# fixed committer to record the tree it gives); the trees are reset to the pins afterwards.
set -u
W=~/spark-lab/agents/round-55-57-select; cd $W
TL_TIP=3b0c78142cd09d5addc69c2e523b6071fe012aab; SN_TIP=1fd6a1fbd8dc4884057046d1ebd93d585c9e4bcf
[ -d tilelang ] || git clone -q --no-hardlinks ../round-51-53-select/tilelang tilelang
[ -d sparknet ] || git clone -q --no-hardlinks ../round-51-53-select/sparknet sparknet
git -C tilelang checkout -q -B tl1107 $TL_TIP; git -C sparknet checkout -q -B base $SN_TIP
echo "tilelang 1107 tip $(git -C tilelang rev-parse HEAD) tree $(git -C tilelang rev-parse HEAD^{tree})"
echo "sparknet 1107 tip $(git -C sparknet rev-parse HEAD) tree $(git -C sparknet rev-parse HEAD^{tree})"
check() { # repo pin patch
  git -C $1 reset -q --hard $2; git -C $1 clean -qfd
  if git -C $1 apply --check $W/$3 2>/tmp/r555757-ap.err; then
    git -C $1 -c user.name=b -c user.email=b@b am -q --committer-date-is-author-date $W/$3
    echo "OK $3 -> tree $(git -C $1 rev-parse HEAD^{tree})"
  else echo "FAIL $3 :: $(tr '\n' ' ' < /tmp/r555757-ap.err)"; fi
  git -C $1 reset -q --hard $2; git -C $1 clean -qfd
}
echo -n "#967 tilelang: "; check tilelang $TL_TIP v/967/tilelang/0014-transform-vectorize-reinterpret-and-merge-readstate-ifs.patch
echo -n "#1008 sparknet: "; check sparknet $SN_TIP v/1008/sparknet/0001-perf-oneshot-pin-RoCE-proxy-thread-to-dedicated-fast.patch
echo "after: tilelang $(git -C tilelang rev-parse HEAD), sparknet $(git -C sparknet rev-parse HEAD)"
