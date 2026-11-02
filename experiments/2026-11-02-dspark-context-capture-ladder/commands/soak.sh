#!/bin/bash
# Poisoned-buffer soak of both ladders, TP3 and TP4 shapes, on whichever dev node runs it.
set -u
W=/tmp/c37-ladder-739
mkdir -p $W/cache/out $W/logs
for shape in tp3 tp4; do
  name=soak-$shape
  echo "$(date -Is) start $name $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ' ')" >> $W/logs/soak.log
  bash $W/runimg.sh spark-ds41f:candidate-37 $W/vllm /bench/bench_context_ladder.py \
    --shape $shape --first base --soak 3000 --out /c/out/$name.json > $W/logs/$name.log 2>&1
  echo "$(date -Is) end $name rc=$?" >> $W/logs/soak.log
done
