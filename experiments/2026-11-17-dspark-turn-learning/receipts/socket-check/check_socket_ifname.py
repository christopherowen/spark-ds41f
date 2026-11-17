"""Does a Socket communicator made after an IB one pick up NCCL_SOCKET_IFNAME set for its init alone?
Order as in serving: a default communicator first (NCCL_NET=IB), then the learner's Socket one."""
import os
import torch

torch.cuda.set_device(0)
from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary  # noqa: E402
import vllm.v1.worker.gpu.spec_decode.dspark.speculator as sp  # noqa: E402

nccl = NCCLLibrary()
print("== default first (NCCL_NET=IB, NCCL_SOCKET_IFNAME=enP7s7)", flush=True)
nccl.ncclCommInitRank(1, nccl.ncclGetUniqueId(), 0)
print("default ok", flush=True)
print("== socket, NCCL_SOCKET_IFNAME=enp1s0f0np0 for its init", flush=True)
saved = None
os.environ["VLLM_DSPARK_TURN_LEARNING_SOCKET_IFNAME"] = "enp1s0f0np0"
try:
    sp._socket_init_rank(nccl, 1, nccl.ncclGetUniqueId(), 0)
finally:
    print("NCCL_SOCKET_IFNAME after:", os.environ.get("NCCL_SOCKET_IFNAME"), "NCCL_NET after:", os.environ.get("NCCL_NET"), flush=True)
print("socket ok", flush=True)
