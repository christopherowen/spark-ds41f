. $HOME/spark-lab/agents/candidate-48-own-marks/batches/env.sh
# PR #980's screen orchestrator ssh-es to sp5..sp8 and takes their locks, so it is not run.
# Only its --worker entry point is run here, on sp8 alone, with the worker's own docker flags.
EXP=$B/src/pr-980/experiments/2026-11-08-spec-decode-zero-broadcast-sync
ENVF="-e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e TP_SOCKET_IFNAME=enp1s0f0np0 -e NCCL_IB_HCA==rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_MERGE_NICS=1 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_CROSS_NIC=1 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none -e NCCL_NVLS_ENABLE=0 -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=1 -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
DFLAGS="--network=host --ipc=host --gpus=all --device=/dev/infiniband:/dev/infiniband:rwm --cap-add=IPC_LOCK --ulimit=memlock=-1:-1 --entrypoint python3"
MNT="-v $EXP/screen_zero_broadcast_sync.py:/tmp/screen_zero_broadcast_sync.py:ro"

# 1. One attempt at world_size=2 with both ranks on sp8's single GB10 (two containers).
tp2() {
  timeout 300 docker run --rm --name c48om-980-tp2-r1 $DFLAGS $ENVF $MNT $C /tmp/screen_zero_broadcast_sync.py --worker --boot tp2-sp8 --rank 1 --world-size 2 --master-addr 10.30.0.16 --master-port 29861 --verify-steps 1000 --bench-iters 1000 > $B/out/pr-980/screen-tp2-one-gpu-rank1.txt 2>&1 &
  timeout 300 docker run --rm --name c48om-980-tp2-r0 $DFLAGS $ENVF $MNT $C /tmp/screen_zero_broadcast_sync.py --worker --boot tp2-sp8 --rank 0 --world-size 2 --master-addr 10.30.0.16 --master-port 29861 --verify-steps 1000 --bench-iters 1000
  rc0=$?; wait; echo "rank0_rc=$rc0 (rank1 output in screen-tp2-one-gpu-rank1.txt)"; return $rc0
}
export -f tp2; export B C DFLAGS ENVF MNT
$B/run.sh 980 screen-tp2-one-gpu-rank0 $C -- flock $L bash -c tp2
docker rm -f c48om-980-tp2-r0 c48om-980-tp2-r1 >/dev/null 2>&1

# 2. world_size=1 on sp8 (degenerate: no peer rank, so mismatch counts are trivially 0), two boots.
for b in 1 2; do
  $B/run.sh 980 screen-tp1-smoke-candidate48-boot$b $C -- flock $L timeout 600 docker run --rm --name c48om-980-tp1-b$b $DFLAGS $ENVF $MNT $C /tmp/screen_zero_broadcast_sync.py --worker --boot tp1-boot$b --rank 0 --world-size 1 --master-addr 10.30.0.16 --master-port 2987$b --verify-steps 1000 --bench-iters 1000
done
