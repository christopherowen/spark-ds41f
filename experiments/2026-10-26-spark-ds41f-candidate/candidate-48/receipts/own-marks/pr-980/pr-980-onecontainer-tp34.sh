. $HOME/spark-lab/agents/candidate-48-own-marks/batches/env.sh
# Adapted (ours): PR #980's screen workers at the PR's world sizes (TP3, TP4), every rank a separate
# process in ONE container on sp8's single GB10, NCCL_MULTI_RANK_GPU_ENABLE=1, otherwise the worker's
# own docker flags and env. Boot order as the PR's orchestrator: boot1 TP3 then TP4, boot2 TP4 then TP3.
# Mismatch counts are comparable to the PR's; the latency benchmark is not (no network between ranks).
EXP=$B/src/pr-980/experiments/2026-11-08-spec-decode-zero-broadcast-sync
ENVF="-e NCCL_MULTI_RANK_GPU_ENABLE=1 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e TP_SOCKET_IFNAME=enp1s0f0np0 -e NCCL_IB_HCA==rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_MERGE_NICS=1 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_CROSS_NIC=1 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none -e NCCL_NVLS_ENABLE=0 -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=1 -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
DFLAGS="--network=host --ipc=host --gpus=all --device=/dev/infiniband:/dev/infiniband:rwm --cap-add=IPC_LOCK --ulimit=memlock=-1:-1 --entrypoint bash"
MNT="-v $EXP/screen_zero_broadcast_sync.py:/tmp/screen_zero_broadcast_sync.py:ro"
one() {  # one <boot> <world> <port>
  local boot=$1 world=$2 port=$3
  local W="python3 /tmp/screen_zero_broadcast_sync.py --worker --boot $boot --world-size $world --master-addr 10.30.0.16 --master-port $port --verify-steps 1000 --bench-iters 1000"
  local inner="" r
  for r in $(seq 1 $((world - 1))); do inner="$inner $W --rank $r > /tmp/rank$r.log 2>&1 &"; done
  inner="$inner $W --rank 0; rc=\$?; wait; for r in \$(seq 1 $((world - 1))); do echo \"--- rank\$r log\"; cat /tmp/rank\$r.log; done; exit \$rc"
  $B/run.sh 980 screen-onecontainer-$boot-tp$world $C -- flock $L timeout 900 docker run --rm --name c48om-980-oc-$boot-tp$world $DFLAGS $ENVF $MNT $C -c "$inner"
  grep -h "^JSON_RECORD:" $B/out/pr-980/screen-onecontainer-$boot-tp$world.txt | sed "s/^JSON_RECORD://" >> $B/out/pr-980/screen-onecontainer-$boot.jsonl
  docker rm -f c48om-980-oc-$boot-tp$world >/dev/null 2>&1
}
one boot1 3 29921
one boot1 4 29922
one boot2 4 29931
one boot2 3 29932
