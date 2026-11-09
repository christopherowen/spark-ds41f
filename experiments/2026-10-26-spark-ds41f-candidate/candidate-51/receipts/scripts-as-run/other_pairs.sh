#!/bin/bash
# Pairs on the TileLang and sparknet trees: apply A's patches, check B's (both orders).
W=~/spark-lab/agents/round-51-53-select; cd $W
pair() { # repo ref "A patches" "B patches" label
  git -C $1 reset -q --hard $2; git -C $1 clean -qfd
  for p in $3; do git -C $1 apply --index $W/$p || echo "  (A failed: $p)"; done
  r=clean; for p in $4; do git -C $1 apply --check $W/$p 2>/tmp/pp.err || r="CONFLICT $(grep -o 'patch failed: [^:]*' /tmp/pp.err | sort -u | tr '\n' ' ')"; done
  echo "$5: $r"; git -C $1 reset -q --hard $2; git -C $1 clean -qfd
}
TL967="v/967/tilelang/0014-transform-vectorize-reinterpret-and-merge-readstate-ifs.patch"
TL994="v/994/tilelang/0014-cuda-fp4-ptx92-cvt-bf16x2-e2m1x2.patch"
TL1015="$(ls v/1015/tilelang/*.patch | sort | tr '\n' ' ')"
pair tilelang tl1107 "$TL967" "$TL1015" "tilelang 1015 after 967"
pair tilelang tl1107 "$TL1015" "$TL967" "tilelang 967 after 1015"
pair tilelang tl1107 "$TL994" "$TL1015" "tilelang 1015 after 994"
pair tilelang tl1107 "$TL1015" "$TL994" "tilelang 994 after 1015"
pair tilelang tl1107 "$TL967" "$TL994" "tilelang 994 after 967"
S962="v/962/sparknet/0001-oneshot-trace-all-collectives-split-cta-wait-and-exp.patch"; S982="v/982/sparknet/sparknet-pdl-direct-stage.patch"
S989="v/989/sparknet/0001-oneshot-add-single-kernel-multi-stage-pipelined-RoCE.patch"; S1008="v/1008/sparknet/0001-perf-oneshot-pin-RoCE-proxy-thread-to-dedicated-fast.patch"
for a in 962 982 989 1008; do for b in 962 982 989 1008; do [ $a = $b ] && continue; A=S$a; B=S$b; pair sparknet base "${!A}" "${!B}" "sparknet $b after $a"; done; done
