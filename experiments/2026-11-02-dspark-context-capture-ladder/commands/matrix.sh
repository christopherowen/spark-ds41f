#!/bin/bash
# The measurement matrix: each (shape, first arm) twice, every run a fresh process.
set -u
W=/tmp/c37-ladder-739
mkdir -p $W/cache/out $W/logs
for rep in 1 2; do
  for shape in tp3 tp4; do
    for first in base ladder; do
      name=$shape-$first-first-$rep
      echo "$(date -Is) start $name" >> $W/logs/matrix.log
      nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader >> $W/logs/matrix.log
      bash $W/runimg.sh spark-ds41f:candidate-37 $W/vllm /bench/bench_context_ladder.py \
        --shape $shape --first $first --hot-iters 300 --cold-iters 100 --rounds 3 \
        --out /c/out/$name.json > $W/logs/$name.log 2>&1
      echo "$(date -Is) end $name rc=$?" >> $W/logs/matrix.log
    done
  done
done
echo "$(date -Is) matrix done" >> $W/logs/matrix.log
