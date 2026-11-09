#!/bin/bash
D=$HOME/spark-lab/agents/candidate-53-rework-998
cd $D
# wait for chain 1 to finish
until [ -e $D/out/chain.done ]; do sleep 10; done
BARGS=(-v $D/scripts:/s:ro -v $D/out/json:/o -e SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=1
       -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6 -e VLLM_DS41_HEAD_DTYPE=float32)
./run.sh fix control-inprocess-fix-boot1 "${BARGS[@]}" -- python3 /s/control_inprocess.py fix-boot1 /o/control-inprocess-fix-boot1.json
./run.sh base control-inprocess-base-boot1 "${BARGS[@]}" -- python3 /s/control_inprocess.py base-boot1 /o/control-inprocess-base-boot1.json
./run.sh fix control-inprocess-fix-boot2 "${BARGS[@]}" -- python3 /s/control_inprocess.py fix-boot2 /o/control-inprocess-fix-boot2.json
echo CHAIN2-DONE > $D/out/chain2.done
