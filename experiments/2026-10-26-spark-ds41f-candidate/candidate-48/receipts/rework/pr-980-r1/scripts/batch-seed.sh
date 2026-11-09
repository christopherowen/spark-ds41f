#!/bin/bash
# PR #980 rework r1, seed follow-up: _seed_rng keyed by ModelConfig.seed (ov/r1seed: seven files incl. model_runner.py).
# The PR tests (two boots) and the harness-fixed screen (rowsfix + written entries only) with --sampling-seed 1234,
# TP3 and TP4, two boots in the PR orchestrator order; every rank a process in one container on sp2s GB10.
# NCCL_MULTI_RANK_GPU_ENABLE=1, otherwise the screen worker's own docker flags and env.
W=$HOME/spark-lab/agents/candidate-48-rework-980
cd $W
C=sha256:d6ffd842978849651e756ebd39e9bf2da7ddb03e746113013533029878485a24   # spark-ds41f:candidate-48
R=sha256:3f32a962ce04b7c8d85337cd427a8946b8b6bcaadd97cc8d8a8cd14e8d68a6ce   # spark-ds41f:1107
L=/tmp/spark-lab-gpu.lock
T=/opt/spark-ds41f/candidate/vllm
FILES="model_runner.py sample/sampler.py sample/states.py spec_decode/acceptance_estimator.py spec_decode/adaptive_verification.py spec_decode/dflash/speculator.py spec_decode/speculator.py"
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

for boot in 1 2; do
  ./run.sh pytest-r1seed-boot$boot -- flock $L timeout 1200 docker run --rm --name c48rw980-pyt-seed-b$boot --gpus all --ipc=host --network=host \
    $(ovmounts r1seed) -v $W/ov/r1seed-tests/tests/v1/spec_decode/test_adaptive_verification.py:$T/tests/v1/spec_decode/test_adaptive_verification.py:ro \
    --entrypoint pytest $C --noconftest -v $T/tests/v1/spec_decode/test_adaptive_verification.py $T/tests/v1/spec_decode/test_acceptance_estimator.py \
    $T/tests/v1/spec_decode/test_copy_drafts.py $T/tests/v1/worker/test_dspark_greedy_markov.py $T/tests/distributed/test_roce_policy_no_fallback.py \
    $T/tests/kernels/test_deepseek_v41_tilelang_confidence.py
done
screen() {  # screen <label> <arm> <script> <world> <port> <boot> <sampling-seed>
  local label=$1 arm=$2 script=$3 world=$4 port=$5 boot=$6 sseed=$7
  local MNT="-v $W/$script:/tmp/screen.py:ro $(ovmounts $arm)"
  local Wk="python3 /tmp/screen.py --worker --boot $boot --world-size $world --master-addr 10.30.0.10 --master-port $port --verify-steps 1000 --bench-iters 1000 --sampling-seed $sseed"
  local inner="" r
  for r in $(seq 1 $((world - 1))); do inner="$inner $Wk --rank $r > /tmp/rank$r.log 2>&1 &"; done
  inner="$inner $Wk --rank 0; rc=\$?; wait; for r in \$(seq 1 $((world - 1))); do echo \"--- rank\$r log\"; cat /tmp/rank\$r.log; done; exit \$rc"
  ./run.sh $label -- flock $L timeout 1200 docker run --rm --name c48rw980-$label $DFLAGS $ENVF $MNT $C -c "$inner"
  grep -h "^JSON_RECORD:" out/$label.txt | sed "s/^JSON_RECORD://" > out/$label.jsonl
  docker rm -f c48rw980-$label > /dev/null 2>&1
}
SC=screen_zero_broadcast_sync_rowsfix_valid_seeded.py
screen screen-r1seed-boot1-tp3 r1seed $SC 3 29971 boot1 1234
screen screen-r1seed-boot1-tp4 r1seed $SC 4 29972 boot1 1234
screen screen-r1seed-boot2-tp4 r1seed $SC 4 29973 boot2 1234
screen screen-r1seed-boot2-tp3 r1seed $SC 3 29974 boot2 1234
./run.sh trace-r1seed -- ./stepcost.sh r1seed trace --rounds 1 --iters 5 --warmup 3 --trace-out /out/trace-r1seed.json
echo batch-seed-done
