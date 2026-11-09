. $HOME/spark-lab/agents/candidate-48-own-marks/batches/env.sh
# Diagnostic (ours): the per-rank-container layout fails with 'invalid device context' even with
# NCCL_MULTI_RANK_GPU_ENABLE=1. One try with every rank as a separate process in one container,
# NCCL_DEBUG=WARN, TP2. If this fails too, the screen is not runnable on one GB10.
EXP=$B/src/pr-980/experiments/2026-11-08-spec-decode-zero-broadcast-sync
ENVF="-e NCCL_DEBUG=WARN -e NCCL_MULTI_RANK_GPU_ENABLE=1 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e TP_SOCKET_IFNAME=enp1s0f0np0 -e NCCL_IB_HCA==rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_MERGE_NICS=1 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_CROSS_NIC=1 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none -e NCCL_NVLS_ENABLE=0 -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=1 -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
DFLAGS="--network=host --ipc=host --gpus=all --device=/dev/infiniband:/dev/infiniband:rwm --cap-add=IPC_LOCK --ulimit=memlock=-1:-1 --entrypoint bash"
MNT="-v $EXP/screen_zero_broadcast_sync.py:/tmp/screen_zero_broadcast_sync.py:ro"
W="python3 /tmp/screen_zero_broadcast_sync.py --worker --boot diag --world-size 2 --master-addr 10.30.0.16 --master-port 29901 --verify-steps 1000 --bench-iters 1000"
$B/run.sh 980 screen-diag-tp2-one-container $C -- flock $L timeout 400 docker run --rm --name c48om-980-diag $DFLAGS $ENVF $MNT $C -c "$W --rank 1 > /tmp/rank1.log 2>&1 & $W --rank 0; rc=\$?; wait; echo '--- rank1 log'; cat /tmp/rank1.log; exit \$rc"
