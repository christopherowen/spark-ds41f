#!/bin/bash
# Repeat of tp4-ladder-first-2, which another session's GPU job overlapped
# (17:04:41-17:06:10): the same command, with the GPU's compute processes sampled
# every 2 s for the whole run.
set -u
W=/tmp/c37-ladder-739
name=tp4-ladder-first-3
( while true; do echo "$(date -Is) $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ' ')"; sleep 2; done ) > $W/logs/$name.gpu.log &
sampler=$!
echo "$(date -Is) start $name" >> $W/logs/matrix.log
bash $W/runimg.sh spark-ds41f:candidate-37 $W/vllm /bench/bench_context_ladder.py \
  --shape tp4 --first ladder --hot-iters 300 --cold-iters 100 --rounds 3 \
  --out /c/out/$name.json > $W/logs/$name.log 2>&1
echo "$(date -Is) end $name rc=$?" >> $W/logs/matrix.log
kill $sampler
