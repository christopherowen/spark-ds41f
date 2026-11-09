#!/bin/bash
# Run on sp8 in ~/spark-lab/agents/round-51-53-select. TileLang at 1107's pin with 1107's TileLang series
# (git am, fixed committer), and sparknet at 1107's pin (no series); then each PR's TileLang / sparknet
# patches alone on those trees (git apply --check, then 3-way when that fails).
set -u
W=~/spark-lab/agents/round-51-53-select; cd $W
export GIT_COMMITTER_NAME=b GIT_COMMITTER_EMAIL=b@b GIT_COMMITTER_DATE=2026-01-01T00:00:00Z
if [ ! -d tilelang ]; then
  git clone -q --no-checkout --filter=blob:none https://github.com/tile-ai/tilelang.git tilelang
  git -C tilelang checkout -q -B base 2123e412ce2217b324c50c4ec967213906456ac5
  git -C tilelang config gc.auto 0
  for p in $(grep -v '^#' tl-series/experiments/2026-10-26-spark-ds41f-candidate/1107/tilelang/series | sed '/^$/d'); do
    git -C tilelang -c user.name=b -c user.email=b@b am -q --keep-cr $W/tl-series/experiments/2026-10-26-spark-ds41f-candidate/1107/tilelang/$p || { echo "TL SERIES FAIL $p"; git -C tilelang am --abort; break; }
  done
  git -C tilelang branch -f tl1107 HEAD
fi
echo "tilelang 1107 tip $(git -C tilelang rev-parse tl1107) tree $(git -C tilelang rev-parse tl1107^{tree})"
if [ ! -d sparknet ]; then
  git clone -q --no-checkout https://github.com/christopherowen/dgx-spark-networking.git sparknet
  git -C sparknet checkout -q -B base 1fd6a1fbd8dc4884057046d1ebd93d585c9e4bcf
fi
echo "sparknet tip $(git -C sparknet rev-parse HEAD) tree $(git -C sparknet rev-parse HEAD^{tree})"
check() { # repo ref patch
  git -C $1 reset -q --hard $2; git -C $1 clean -qfd
  if git -C $1 apply --check $3 2>/tmp/ap.err; then echo "OK $3"
  else
    if git -C $1 apply -3 --index $3 >/dev/null 2>&1; then echo "3WAY-CLEAN $3 :: $(grep -o 'patch failed: [^:]*' /tmp/ap.err | sort -u | tr '\n' ' ')"
    else echo "FAIL $3 :: $(grep -o 'patch failed: [^:]*\|does not exist in index\|No such file' /tmp/ap.err | sort -u | tr '\n' ' ')"; fi
  fi
  git -C $1 reset -q --hard $2; git -C $1 clean -qfd
}
for n in 967 994 1015; do for p in $(ls v/$n/tilelang/*.patch | sort); do echo -n "#$n tilelang: "; check tilelang tl1107 $W/$p; done; done
for n in 962 982 989 1008; do for p in $(ls v/$n/sparknet/*.patch | sort); do echo -n "#$n sparknet: "; check sparknet base $W/$p; done; done
