#!/bin/bash
# Capture memory, amplified: 20 copies of one ladder per fresh process.
set -u
W=/tmp/c37-ladder-739
for rep in 1 2; do
  for shape in tp3 tp4; do
    for first in base ladder; do
      name=memory-$shape-$first-$rep
      echo "$(date -Is) start $name" >> $W/logs/memory.log
      bash $W/runimg.sh spark-ds41f:candidate-37 $W/vllm /bench/bench_context_ladder.py \
        --shape $shape --first $first --memory-copies 20 \
        --out /c/out/$name.json > $W/logs/$name.log 2>&1
      echo "$(date -Is) end $name rc=$?" >> $W/logs/memory.log
    done
  done
done
echo "$(date -Is) memory done" >> $W/logs/memory.log
