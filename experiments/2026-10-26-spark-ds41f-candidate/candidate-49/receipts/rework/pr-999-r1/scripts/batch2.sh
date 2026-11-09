#!/usr/bin/env bash
# PR #999 rework r1: the corrected indexer.py (and its test file) bind-mounted over
# spark-ds41f:candidate-49, two boots of: the PR's pytest + reads check, the PR's
# bench_2boot.py, the served-gap bench, and the short-extent sweep (with the
# candidate-49 indexer as an extra arm). Every GPU command runs under the lab lock
# and quiet.sh (waits for no other container; reports foreign containers seen).
set -u
B=$HOME/spark-lab/agents/candidate-49-rework-999
C=sha256:9f51f2e2c5ab728e96490d794bff7b83a147d6d3d61333c5b60b2cff662049cd
L=/tmp/spark-lab-gpu.lock
V=/opt/spark-ds41f/candidate/vllm
IDX=$V/vllm/models/deepseek_v4_1/tilelang/indexer.py
TST=$V/tests/kernels/attention/test_deepseek_v41_tilelang.py
J=$B/out/json; mkdir -p $J
MOUNT_IDX="-v $B/corrected/indexer.py:$IDX:ro"
MOUNT_TST="-v $B/corrected/test_deepseek_v41_tilelang.py:$TST:ro"
BENCH="-v $B/src/bench_2boot.py:/tmp/bench_2boot.py:ro -v $B/src/indexer_baseline_1107.py:/tmp_host/indexer_baseline.py:ro"
for boot in 1 2; do
  n=c49rw999-r1-pyt-b$boot
  $B/run.sh r1-pytest-boot$boot $C -- flock $L $B/quiet.sh $n docker run --rm --gpus all --ipc=host --name $n --entrypoint "" -w $V $MOUNT_IDX $MOUNT_TST $C bash -lc "sha256sum $IDX $TST; pytest --noconftest -q tests/kernels/attention/test_deepseek_v41_tilelang.py tests/models/test_deepseek_v4_1_lsa.py tests/models/test_deepseek_v4_1_attention.py; a=\$?; python3 tests/kernels/attention/deepseek_v41_indexer_reads.py; b=\$?; echo pytest_rc=\$a reads_rc=\$b; exit \$((a|b))"
  n=c49rw999-r1-bench-b$boot
  $B/run.sh r1-bench2boot-boot$boot $C -- flock $L $B/quiet.sh $n docker run --rm --gpus all --ipc=host --name $n --entrypoint "" -w $V $MOUNT_IDX $BENCH -v $J:/out $C bash -lc "sha256sum $IDX; python3 /tmp/bench_2boot.py /out/r1-bench2boot-boot$boot.json"
  n=c49rw999-r1-gaps-b$boot
  $B/run.sh r1-served-gaps-boot$boot $C -- flock $L $B/quiet.sh $n docker run --rm --gpus all --ipc=host --name $n --entrypoint "" -w $V $MOUNT_IDX $BENCH -v $B/src/bench_served_gaps.py:/tmp/bench_served_gaps.py:ro -v $J:/out $C bash -lc "sha256sum $IDX; python3 /tmp/bench_served_gaps.py /out/r1-served-gaps-boot$boot.json"
  n=c49rw999-r1-sweep-b$boot
  $B/run.sh r1-short-sweep-boot$boot $C -- flock $L $B/quiet.sh $n docker run --rm --gpus all --ipc=host --name $n --entrypoint "" -w $V $MOUNT_IDX $BENCH -v $B/src/bench_short_sweep.py:/tmp/bench_short_sweep.py:ro -v $B/c49-orig/indexer.py:/tmp_host/c49/indexer.py:ro -v $J:/out $C bash -lc "sha256sum $IDX /tmp_host/c49/indexer.py; python3 /tmp/bench_short_sweep.py /out/r1-short-sweep-boot$boot.json c49=/tmp_host/c49/indexer.py"
done
