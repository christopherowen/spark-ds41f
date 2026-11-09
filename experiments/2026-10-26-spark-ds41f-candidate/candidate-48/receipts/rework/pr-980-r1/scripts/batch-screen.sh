#!/bin/bash
# PR #980 rework r1 on sp2's single GB10: the PR's tests, its 1,000-step lockstep
# screen at TP3/TP4 (two boots, the PR orchestrator's order), and the TP2 diagnosis.
# Layout as the sp8 own-mark run: every rank a separate process in ONE container,
# NCCL_MULTI_RANK_GPU_ENABLE=1, otherwise the screen worker's own docker flags and env.
W=$HOME/spark-lab/agents/candidate-48-rework-980
cd $W
C=sha256:d6ffd842978849651e756ebd39e9bf2da7ddb03e746113013533029878485a24   # spark-ds41f:candidate-48
R=sha256:3f32a962ce04b7c8d85337cd427a8946b8b6bcaadd97cc8d8a8cd14e8d68a6ce   # spark-ds41f:1107
L=/tmp/spark-lab-gpu.lock
T=/opt/spark-ds41f/candidate/vllm
FILES="sample/sampler.py sample/states.py spec_decode/acceptance_estimator.py spec_decode/adaptive_verification.py spec_decode/dflash/speculator.py spec_decode/speculator.py"
ovmounts() {  # ovmounts <arm>: the arm's six files over the image's
  local m="" f
  for f in $FILES; do m="$m -v $W/ov/$1/vllm/v1/worker/gpu/$f:$T/vllm/v1/worker/gpu/$f:ro"; done
  echo "$m"
}
ENVF="-e NCCL_MULTI_RANK_GPU_ENABLE=1 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e TP_SOCKET_IFNAME=enp1s0f0np0 -e NCCL_IB_HCA==rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_MERGE_NICS=1 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_CROSS_NIC=1 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none -e NCCL_NVLS_ENABLE=0 -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=1 -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
DFLAGS="--network=host --ipc=host --gpus=all --device=/dev/infiniband:/dev/infiniband:rwm --cap-add=IPC_LOCK --ulimit=memlock=-1:-1 --entrypoint bash"

screen() {  # screen <label> <arm|none> <script> <world> <port> <boot>
  local label=$1 arm=$2 script=$3 world=$4 port=$5 boot=$6
  local MNT="-v $W/$script:/tmp/screen.py:ro"
  [ "$arm" != none ] && MNT="$MNT $(ovmounts $arm)"
  local Wk="python3 /tmp/screen.py --worker --boot $boot --world-size $world --master-addr 10.30.0.10 --master-port $port --verify-steps 1000 --bench-iters 1000"
  local inner="" r
  for r in $(seq 1 $((world - 1))); do inner="$inner $Wk --rank $r > /tmp/rank$r.log 2>&1 &"; done
  inner="$inner $Wk --rank 0; rc=\$?; wait; for r in \$(seq 1 $((world - 1))); do echo \"--- rank\$r log\"; cat /tmp/rank\$r.log; done; exit \$rc"
  ./run.sh $label -- flock $L timeout 1200 docker run --rm --name c48rw980-$label $DFLAGS $ENVF $MNT $C -c "$inner"
  grep -h "^JSON_RECORD:" out/$label.txt | sed "s/^JSON_RECORD://" > out/$label.jsonl
  docker rm -f c48rw980-$label > /dev/null 2>&1
}

# 1. The PR's tests with the corrected files and test file, two boots.
for boot in 1 2; do
  ./run.sh pytest-r1-boot$boot -- flock $L timeout 1200 docker run --rm --name c48rw980-pyt-b$boot --gpus all --ipc=host --network=host \
    $(ovmounts r1) -v $W/ov/r1-tests/tests/v1/spec_decode/test_adaptive_verification.py:$T/tests/v1/spec_decode/test_adaptive_verification.py:ro \
    --entrypoint pytest $C --noconftest -v $T/tests/v1/spec_decode/test_adaptive_verification.py $T/tests/v1/spec_decode/test_acceptance_estimator.py \
    $T/tests/v1/spec_decode/test_copy_drafts.py $T/tests/v1/worker/test_dspark_greedy_markov.py $T/tests/distributed/test_roce_policy_no_fallback.py \
    $T/tests/kernels/test_deepseek_v41_tilelang_confidence.py
done

# 2. The mark: the PR's own screen (unchanged) with the corrected files, TP3 and TP4, two boots.
screen screen-r1-boot1-tp3 r1 screen_zero_broadcast_sync.py 3 29941 boot1
screen screen-r1-boot1-tp4 r1 screen_zero_broadcast_sync.py 4 29942 boot1
screen screen-r1-boot2-tp4 r1 screen_zero_broadcast_sync.py 4 29943 boot2
screen screen-r1-boot2-tp3 r1 screen_zero_broadcast_sync.py 3 29944 boot2

# 3. TP2 diagnosis.
# 3a. Deterministic reproduction, no TP and none of #980's code on the call path, on both images.
./run.sh tp2-oob-repro-1107 -- flock $L timeout 900 docker run --rm --name c48rw980-repro-1107 --gpus all --ipc=host --network=host \
  -v $W/tp2_oob_repro.py:/tmp/tp2_oob_repro.py:ro --entrypoint python3 $R /tmp/tp2_oob_repro.py
./run.sh tp2-oob-repro-c48 -- flock $L timeout 900 docker run --rm --name c48rw980-repro-c48 --gpus all --ipc=host --network=host \
  -v $W/tp2_oob_repro.py:/tmp/tp2_oob_repro.py:ro --entrypoint python3 $C /tmp/tp2_oob_repro.py
# 3b. The PR's screen at TP2 as sp8 ran it (candidate-48's own files), then with the corrected files.
screen screen-tp2-c48-orig none screen_zero_broadcast_sync.py 2 29951 diag
screen screen-tp2-r1-orig r1 screen_zero_broadcast_sync.py 2 29952 diag
# 3c. The screen with draft_sampled laid out as [num_logits] (screen-rowsfix.diff), TP2 then TP3/TP4.
screen screen-tp2-c48-rowsfix none screen_zero_broadcast_sync_rowsfix.py 2 29953 diag
screen screen-tp2-r1-rowsfix r1 screen_zero_broadcast_sync_rowsfix.py 2 29954 diag
screen screen-tp3-r1-rowsfix r1 screen_zero_broadcast_sync_rowsfix.py 3 29955 diag
screen screen-tp4-r1-rowsfix r1 screen_zero_broadcast_sync_rowsfix.py 4 29956 diag
echo batch-screen-done
