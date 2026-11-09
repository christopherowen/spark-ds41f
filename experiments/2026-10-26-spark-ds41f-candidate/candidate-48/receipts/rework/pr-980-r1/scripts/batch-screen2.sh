#!/bin/bash
# PR #980 rework r1, follow-up: the TP2 diagnosis with the screen comparing only the sampled tokens rejection_sample wrote.
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

# 3d. rowsfix + only written sampled entries compared (screen-rowsfix-valid.diff); raw counts reported beside.
screen screen-tp2-r1-valid r1 screen_zero_broadcast_sync_rowsfix_valid.py 2 29961 diag
screen screen-tp2-c48-valid none screen_zero_broadcast_sync_rowsfix_valid.py 2 29962 diag
screen screen-tp3-r1-valid r1 screen_zero_broadcast_sync_rowsfix_valid.py 3 29963 diag
screen screen-tp4-r1-valid r1 screen_zero_broadcast_sync_rowsfix_valid.py 4 29964 diag
screen screen-tp2-r1-valid-again r1 screen_zero_broadcast_sync_rowsfix_valid.py 2 29965 diag2
echo batch-screen2-done
