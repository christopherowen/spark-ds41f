#!/bin/bash
# sp1 queue B (after queue A): our graph-replay harnesses for #997 (carve-out 1) and #964; ABBA per PR.
B=$HOME/spark-lab/agents/candidate-47-own-marks; cd $B
while [ ! -e out/q-sp1-a.done ]; do sleep 20; done
E997=$B/src/pr-997/experiments/2026-11-08-certified-lm-head-screen
BLOB=/home/chris/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/blobs/4ae12b941c3facf8616ce62dd45bc6868076c2aeb82106f98123ea6c5bb64182
for step in "1107 1" "c47 1" "c47 2" "1107 2"; do
  set -- $step; arm=$1; boot=$2
  ./om.sh 997 h997-carveout1-$arm-boot$boot $arm --privileged -v $E997/bench_two_boots.py:/tmp/bench_two_boots.py:ro \
    -v $B/h997.py:/h/h997.py:ro -v $BLOB:/weights/head.safetensors:ro -e SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=1 -- \
    python3 /h/h997.py carveout1-$arm-boot$boot
done
E964=$B/src/pr-964/experiments/2026-11-08-sparse-mla-kv-pipeline
for step in "1107 1" "c47 1" "c47 2" "1107 2"; do
  set -- $step; arm=$1; boot=$2
  ./om.sh 964 h964-$arm-boot$boot $arm -v $E964/bench_sparse_mla_pipeline.py:/bench/bench_sparse_mla_pipeline.py:ro \
    -v $B/h964.py:/h/h964.py:ro -- python3 /h/h964.py $arm-boot$boot
done
echo done > out/q-sp1-b.done
