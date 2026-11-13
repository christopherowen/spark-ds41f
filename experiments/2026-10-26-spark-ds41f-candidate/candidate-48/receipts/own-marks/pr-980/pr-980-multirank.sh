. $HOME/spark-lab/agents/candidate-48-own-marks/batches/env.sh
# Adapted (ours): PR #980's screen workers, all ranks on sp8's single GB10, one container per rank,
# with the worker's own docker flags and env, plus NCCL_MULTI_RANK_GPU_ENABLE=1 (NCCL 2.30.7 refuses
# several ranks on one GPU without it: screen-tp2-one-gpu-rank0.txt). Rank processes are separate, so
# each has its own process-global np.random state, which is what the screen's control arm desyncs.
# Latency numbers from this layout are not comparable to the PR's multi-node ones; mismatch counts are.
EXP=$B/src/pr-980/experiments/2026-11-08-spec-decode-zero-broadcast-sync
ENVF="-e NCCL_MULTI_RANK_GPU_ENABLE=1 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e TP_SOCKET_IFNAME=enp1s0f0np0 -e NCCL_IB_HCA==rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_MERGE_NICS=1 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_CROSS_NIC=1 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none -e NCCL_NVLS_ENABLE=0 -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=1 -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
DFLAGS="--network=host --ipc=host --gpus=all --device=/dev/infiniband:/dev/infiniband:rwm --cap-add=IPC_LOCK --ulimit=memlock=-1:-1 --entrypoint python3"
MNT="-v $EXP/screen_zero_broadcast_sync.py:/tmp/screen_zero_broadcast_sync.py:ro"

mr() {  # mr <image> <tag> <boot> <world> <port>
  local I=$1 tag=$2 boot=$3 world=$4 port=$5 r
  for r in $(seq 1 $((world - 1))); do
    timeout 600 docker run --rm --name c48om-980-$tag-$boot-tp$world-r$r $DFLAGS $ENVF $MNT $I /tmp/screen_zero_broadcast_sync.py --worker --boot $boot --rank $r --world-size $world --master-addr 10.30.0.16 --master-port $port --verify-steps 1000 --bench-iters 1000 > $B/out/pr-980/screen-$tag-$boot-tp$world-onegpu-rank$r.txt 2>&1 &
  done
  timeout 600 docker run --rm --name c48om-980-$tag-$boot-tp$world-r0 $DFLAGS $ENVF $MNT $I /tmp/screen_zero_broadcast_sync.py --worker --boot $boot --rank 0 --world-size $world --master-addr 10.30.0.16 --master-port $port --verify-steps 1000 --bench-iters 1000
  local rc0=$?; wait
  echo "rank0_rc=$rc0 (other ranks: screen-$tag-$boot-tp$world-onegpu-rank<N>.txt)"; return $rc0
}
export -f mr; export B DFLAGS ENVF MNT

run_mr() {  # run_mr <image> <tag> <boot> <world> <port>
  $B/run.sh 980 screen-$2-$3-tp$4-onegpu-rank0 $1 -- flock $L bash -c "mr $1 $2 $3 $4 $5"
  grep -h "^JSON_RECORD:" $B/out/pr-980/screen-$2-$3-tp$4-onegpu-rank0.txt | sed "s/^JSON_RECORD://" >> $B/out/pr-980/screen-$2-$3.jsonl
  docker ps -a --format "{{.Names}}" | grep "^c48om-980-$2-$3-tp$4-" | xargs -r docker rm -f >/dev/null 2>&1
}
# Boot order as in the PR's orchestrator: boot1 TP3 then TP4, boot2 TP4 then TP3.
run_mr $C candidate48 boot1 3 29881
run_mr $C candidate48 boot1 4 29882
run_mr $C candidate48 boot2 4 29891
run_mr $C candidate48 boot2 3 29892
