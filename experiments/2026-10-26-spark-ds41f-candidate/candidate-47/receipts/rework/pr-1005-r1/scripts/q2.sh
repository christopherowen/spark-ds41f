#!/bin/bash
# Session 2 (sp3): paired in-process screen, two boots. Live tree: candidate-47 without #1005
# (ovl/no1005); candidate-47's moe.py, moe_runner.py, shared_experts.py at /alt (ro).
W=$HOME/spark-lab/agents/candidate-47-rework-1005
A=$W/ovl/c47
for boot in 1 2; do
  $W/scripts/rw.sh s2-pair-b$boot no1005 \
    -v $A/vllm/models/deepseek_v4_1/tilelang/moe.py:/alt/moe.py:ro \
    -v $A/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:/alt/moe_runner.py:ro \
    -v $A/vllm/model_executor/layers/fused_moe/runner/shared_experts.py:/alt/shared_experts.py:ro \
    -- python3 /h/pairbench.py s2-pair-b$boot --windows 12 --replays 8 --layers 6 --shapes tp4,tp3 \
       --ms 1,2,3,4,6,8,12,16,24,32,48 --variants base,kern,hook,c47,post,postk
  echo "$(date -u +%FT%TZ) s2-pair-b$boot rc=$?" >> $W/out/q2.log
done
echo "$(date -u +%FT%TZ) q2 done" >> $W/out/q2.log
