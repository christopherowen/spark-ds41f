#!/usr/bin/env bash
# usage: rerun.sh <kind> <boot-label>   kind: bench2boot | served-gaps | short-sweep
# Repeat boot of one r1 bench (same command as batch2.sh) under quiet2.sh, which
# reports any foreign GPU container that shared the run.
set -u
B=$HOME/spark-lab/agents/candidate-49-rework-999
C=sha256:9f51f2e2c5ab728e96490d794bff7b83a147d6d3d61333c5b60b2cff662049cd
L=/tmp/spark-lab-gpu.lock
V=/opt/spark-ds41f/candidate/vllm
IDX=$V/vllm/models/deepseek_v4_1/tilelang/indexer.py
J=$B/out/json
MOUNT_IDX="-v $B/corrected/indexer.py:$IDX:ro"
BENCH="-v $B/src/bench_2boot.py:/tmp/bench_2boot.py:ro -v $B/src/indexer_baseline_1107.py:/tmp_host/indexer_baseline.py:ro"
kind=$1; boot=$2
n=c49rw999-r1-$kind-$boot
case $kind in
  bench2boot) $B/run.sh r1-bench2boot-$boot $C -- flock $L $B/quiet2.sh $n docker run --rm --gpus all --ipc=host --name $n --entrypoint "" -w $V $MOUNT_IDX $BENCH -v $J:/out $C bash -lc "sha256sum $IDX; python3 /tmp/bench_2boot.py /out/r1-bench2boot-$boot.json" ;;
  served-gaps) $B/run.sh r1-served-gaps-$boot $C -- flock $L $B/quiet2.sh $n docker run --rm --gpus all --ipc=host --name $n --entrypoint "" -w $V $MOUNT_IDX $BENCH -v $B/src/bench_served_gaps.py:/tmp/bench_served_gaps.py:ro -v $J:/out $C bash -lc "sha256sum $IDX; python3 /tmp/bench_served_gaps.py /out/r1-served-gaps-$boot.json" ;;
  short-sweep) $B/run.sh r1-short-sweep-$boot $C -- flock $L $B/quiet2.sh $n docker run --rm --gpus all --ipc=host --name $n --entrypoint "" -w $V $MOUNT_IDX $BENCH -v $B/src/bench_short_sweep.py:/tmp/bench_short_sweep.py:ro -v $B/c49-orig/indexer.py:/tmp_host/c49/indexer.py:ro -v $J:/out $C bash -lc "sha256sum $IDX /tmp_host/c49/indexer.py; python3 /tmp/bench_short_sweep.py /out/r1-short-sweep-$boot.json c49=/tmp_host/c49/indexer.py" ;;
esac
