#!/bin/bash
# sp4 queue A: PR #966 verbatim bench, then PR #1005 verbatim bench; ABBA per PR.
B=$HOME/spark-lab/agents/candidate-47-own-marks; cd $B
E966=$B/src/pr-966/experiments/2026-11-08-l2-eviction-policy
for step in "1107 1" "c47 1" "c47 2" "1107 2"; do
  set -- $step; arm=$1; boot=$2
  ./om.sh 966 bench-$arm-boot$boot $arm -v $E966/bench_moe_l2_evict.py:/tmp/bench_moe_evict.py:ro -- python3 /tmp/bench_moe_evict.py
done
E1005=$B/src/pr-1005/experiments/2026-11-08-moe-swiglu-and-dual-stream-graph
for step in "1107 1" "c47 1" "c47 2" "1107 2"; do
  set -- $step; arm=$1; boot=$2
  ./om.sh 1005 bench-$arm-boot$boot $arm -v $E1005/bench_moe_swiglu_and_dual_stream.py:/workspace/bench_moe_swiglu_and_dual_stream.py:ro -- \
    python3 /workspace/bench_moe_swiglu_and_dual_stream.py --boot $boot --out /out/bench-$arm-boot$boot.json
done
echo done > $B/out/q-sp4-a.done
