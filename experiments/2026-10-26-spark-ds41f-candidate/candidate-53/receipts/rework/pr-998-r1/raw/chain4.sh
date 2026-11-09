#!/bin/bash
D=$HOME/spark-lab/agents/candidate-53-rework-998
cd $D
until [ -e $D/out/chain3.done ]; do sleep 10; done
BARGS=(-v $D/scripts:/s:ro -v $D/out/json:/o -e SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=1
       -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6 -e VLLM_DS41_HEAD_DTYPE=float32)
./run.sh base control-inprocess-base-boot1b "${BARGS[@]}" -- python3 /s/control_inprocess.py base-boot1b /o/control-inprocess-base-boot1b.json
echo CHAIN4-DONE > $D/out/chain4.done
