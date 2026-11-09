#!/usr/bin/env bash
# usage: rerun_warm.sh <kind> <boot-label>   kind: bench2boot | served-gaps | short-sweep
# As rerun.sh, through warm_wrap.py (a GPU clock warm-up before every timed arm).
set -u
B=$HOME/spark-lab/agents/candidate-49-rework-999
C=sha256:9f51f2e2c5ab728e96490d794bff7b83a147d6d3d61333c5b60b2cff662049cd
L=/tmp/spark-lab-gpu.lock
V=/opt/spark-ds41f/candidate/vllm
IDX=$V/vllm/models/deepseek_v4_1/tilelang/indexer.py
J=$B/out/json
MOUNT_IDX="-v $B/corrected/indexer.py:$IDX:ro"
BENCH="-v $B/src/bench_2boot.py:/tmp/bench_2boot.py:ro -v $B/src/indexer_baseline_1107.py:/tmp_host/indexer_baseline.py:ro -v $B/src/warm_wrap.py:/tmp/warm_wrap.py:ro"
kind=$1; boot=$2
n=c49rw999-r1w-$kind-$boot
case $kind in
  bench2boot) $B/run.sh r1w-bench2boot-$boot $C -- flock $L $B/quiet2.sh $n docker run --rm --gpus all --ipc=host --name $n --entrypoint "" -w $V $MOUNT_IDX $BENCH -v $J:/out $C bash -lc "sha256sum $IDX; python3 /tmp/warm_wrap.py /tmp/bench_2boot.py /out/r1w-bench2boot-$boot.json" ;;
  served-gaps) $B/run.sh r1w-served-gaps-$boot $C -- flock $L $B/quiet2.sh $n docker run --rm --gpus all --ipc=host --name $n --entrypoint "" -w $V $MOUNT_IDX $BENCH -v $B/src/bench_served_gaps.py:/tmp/bench_served_gaps.py:ro -v $J:/out $C bash -lc "sha256sum $IDX; python3 /tmp/warm_wrap.py /tmp/bench_served_gaps.py /out/r1w-served-gaps-$boot.json" ;;
  short-sweep) $B/run.sh r1w-short-sweep-$boot $C -- flock $L $B/quiet2.sh $n docker run --rm --gpus all --ipc=host --name $n --entrypoint "" -w $V $MOUNT_IDX $BENCH -v $B/src/bench_short_sweep.py:/tmp/bench_short_sweep.py:ro -v $B/c49-orig/indexer.py:/tmp_host/c49/indexer.py:ro -v $J:/out $C bash -lc "sha256sum $IDX /tmp_host/c49/indexer.py; python3 /tmp/warm_wrap.py /tmp/bench_short_sweep.py /out/r1w-short-sweep-$boot.json c49=/tmp_host/c49/indexer.py" ;;
esac
case $kind in
  pair-c49) $B/run.sh r1w-pair-c49-$boot $C -- flock $L $B/quiet2.sh $n docker run --rm --gpus all --ipc=host --name $n --entrypoint "" -w $V $MOUNT_IDX $BENCH -v $B/src/bench_pair_c49.py:/tmp/bench_pair_c49.py:ro -v $B/c49-orig/indexer.py:/tmp_host/c49/indexer.py:ro -v $J:/out $C bash -lc "sha256sum $IDX /tmp_host/c49/indexer.py; python3 /tmp/warm_wrap.py /tmp/bench_pair_c49.py /out/r1w-pair-c49-$boot.json c49=/tmp_host/c49/indexer.py" ;;
esac
