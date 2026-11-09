#!/bin/bash
D=$HOME/spark-lab/agents/candidate-53-rework-998
cd $D
until [ -e $D/out/chain2.done ]; do sleep 10; done
BARGS=(-v $D/scripts:/s:ro -v $D/out/json:/o -e SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=1
       -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6 -e VLLM_DS41_HEAD_DTYPE=float32)
./run.sh fix control-inprocess-fix-boot1b "${BARGS[@]}" -- python3 /s/control_inprocess.py fix-boot1b /o/control-inprocess-fix-boot1b.json
echo CHAIN3-DONE > $D/out/chain3.done
