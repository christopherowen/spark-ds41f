#!/bin/bash
# sp3 queue B (after queue A): our graph-replay harness for #1016; ABBA.
B=$HOME/spark-lab/agents/candidate-47-own-marks; cd $B
while [ ! -e out/q-sp3-a.done ]; do sleep 20; done
E=$B/src/pr-1016/experiments/2026-11-08-qb-indexer-qb-tile-contiguous-weights
for step in "1107 1" "c47 1" "c47 2" "1107 2"; do
  set -- $step; arm=$1; boot=$2
  ./om.sh 1016 h1016-$arm-boot$boot $arm -v $E/bench_qb_tile_contiguous.py:/bench/bench_qb_tile_contiguous.py:ro \
    -v $B/h1016.py:/h/h1016.py:ro -- python3 /h/h1016.py $arm-boot$boot
done
echo done > out/q-sp3-b.done
