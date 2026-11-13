#!/bin/bash
# In-process A/B/C step cost (two boots) and the warmup-token check (two boots per arm).
cd $HOME/spark-lab/agents/candidate-48-rework-980
./run.sh stepcost-ab-boot1 -- ./stepcost_ab.sh boot1 --iters 1500 --warmup 30
for arm in 1107 c48 r1; do ./run.sh warmup-tokens-$arm-boot1 -- ./warmup_tokens.sh $arm boot1; done
./run.sh stepcost-ab-boot2 -- ./stepcost_ab.sh boot2 --iters 1500 --warmup 30
for arm in r1 c48 1107; do ./run.sh warmup-tokens-$arm-boot2 -- ./warmup_tokens.sh $arm boot2; done
echo batch-ab-done
