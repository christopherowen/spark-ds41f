#!/bin/bash
# sp3 queue A: PR #1016 verbatim bench, ABBA.
B=$HOME/spark-lab/agents/candidate-47-own-marks; cd $B
E=$B/src/pr-1016/experiments/2026-11-08-qb-indexer-qb-tile-contiguous-weights
for step in "1107 1" "c47 1" "c47 2" "1107 2"; do
  set -- $step; arm=$1; boot=$2
  ./om.sh 1016 bench-$arm-boot$boot $arm -v $E/bench_qb_tile_contiguous.py:/bench/bench_qb_tile_contiguous.py:ro -- \
    python3 /bench/bench_qb_tile_contiguous.py --output /out/bench-$arm-boot$boot.json
done
echo done > $B/out/q-sp3-a.done
