#!/bin/bash
# Step-cost bench: three arms (1107 files, candidate-48 files = original #980, r1 corrected),
# two boots each, one fresh container per arm-boot, boot 2 in reverse order.
cd $HOME/spark-lab/agents/candidate-48-rework-980
A="--rounds 9 --iters 300 --warmup 50"
for arm in 1107 c48 r1; do ./run.sh stepcost-$arm-boot1 -- ./stepcost.sh $arm boot1 $A --trace-out /out/trace-$arm-boot1.json; done
for arm in r1 c48 1107; do ./run.sh stepcost-$arm-boot2 -- ./stepcost.sh $arm boot2 $A; done
echo batch-stepcost-done
