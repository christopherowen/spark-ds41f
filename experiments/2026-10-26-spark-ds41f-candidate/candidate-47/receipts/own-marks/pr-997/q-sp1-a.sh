#!/bin/bash
# sp1 queue A: PR #997 verbatim bench (carve-out 1, then 0), then PR #964 verbatim bench; ABBA per PR.
B=$HOME/spark-lab/agents/candidate-47-own-marks; cd $B
E997=$B/src/pr-997/experiments/2026-11-08-certified-lm-head-screen
BLOB=/home/chris/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/blobs/4ae12b941c3facf8616ce62dd45bc6868076c2aeb82106f98123ea6c5bb64182
for CO in 1 0; do
  for step in "1107 baseline 1" "c47 candidate 1" "c47 candidate 2" "1107 baseline 2"; do
    set -- $step; arm=$1; mode=$2; boot=$3
    ./om.sh 997 bench-carveout$CO-$arm-boot$boot $arm --privileged -v $E997/bench_two_boots.py:/tmp/bench_two_boots.py:ro \
      -v $BLOB:/weights/head.safetensors:ro -e SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=$CO -- python3 /tmp/bench_two_boots.py $mode $boot
  done
done
E964=$B/src/pr-964/experiments/2026-11-08-sparse-mla-kv-pipeline
for step in "1107 1" "c47 1" "c47 2" "1107 2"; do
  set -- $step; arm=$1; boot=$2
  ./om.sh 964 bench-$arm-boot$boot $arm -v $E964/bench_sparse_mla_pipeline.py:/bench/bench_sparse_mla_pipeline.py:ro -- \
    python3 /bench/bench_sparse_mla_pipeline.py --output /out/bench-$arm-boot$boot.json --boot $boot
done
echo done > $B/out/q-sp1-a.done
