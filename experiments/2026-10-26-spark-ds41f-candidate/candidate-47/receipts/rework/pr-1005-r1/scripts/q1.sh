#!/bin/bash
# Session 1 (sp3): four arms, palindromic order, two boots each (a boot = a fresh container).
# no1005 = candidate-47 without #1005; kern = + 0765's moe.py only; hook = + 0765's moe_runner.py
# and shared_experts.py only; c47 = candidate-47 (0765 whole).
W=$HOME/spark-lab/agents/candidate-47-rework-1005
MS=1,2,3,4,6,8,12,16,20,24,28,32,40,48
for run in no1005:1 kern:1 hook:1 c47:1 c47:2 hook:2 kern:2 no1005:2; do
  arm=${run%%:*}; boot=${run##*:}
  $W/scripts/rw.sh s1-$arm-b$boot $arm -- python3 /h/ffnbench.py s1-$arm-b$boot --windows 15 --replays 10 --layers 6 --shapes tp4,tp3 --ms $MS
  echo "$(date -u +%FT%TZ) s1-$arm-b$boot rc=$?" >> $W/out/q1.log
done
echo "$(date -u +%FT%TZ) q1 done" >> $W/out/q1.log
