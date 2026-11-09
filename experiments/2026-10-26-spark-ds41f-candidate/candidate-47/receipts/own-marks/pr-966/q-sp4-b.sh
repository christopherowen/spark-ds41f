#!/bin/bash
# sp4 queue B (after queue A): our graph-replay harness for #966; ABBA.
B=$HOME/spark-lab/agents/candidate-47-own-marks; cd $B
while [ ! -e out/q-sp4-a.done ]; do sleep 20; done
E966=$B/src/pr-966/experiments/2026-11-08-l2-eviction-policy
for step in "1107 1" "c47 1" "c47 2" "1107 2"; do
  set -- $step; arm=$1; boot=$2
  ./om.sh 966 h966-$arm-boot$boot $arm -v $E966/bench_moe_l2_evict.py:/bench/bench_moe_l2_evict.py:ro \
    -v $B/h966.py:/h/h966.py:ro -- python3 /h/h966.py $arm-boot$boot
done
echo done > out/q-sp4-b.done
