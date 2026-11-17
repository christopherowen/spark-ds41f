"""One-rank check of the Socket exchange's ncclCommInitRankConfig: the config is accepted and selects
the Socket network while NCCL_NET names IB; then the subclass's _init_comm path with its warm-up."""
import torch  # loads libnccl.so.2 first

torch.cuda.set_device(0)
from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary  # noqa: E402
import vllm.v1.worker.gpu.spec_decode.dspark.speculator as sp  # noqa: E402

nccl = NCCLLibrary()
print("runtime NCCL", nccl.ncclGetRawVersion(), "config bytes", __import__("ctypes").sizeof(sp._NcclConfig), flush=True)
import os
print("== socket config", flush=True)
socket = sp._socket_init_rank(nccl, 1, nccl.ncclGetUniqueId(), 0)
print("socket comm", socket, "NCCL_NET restored:", os.environ.get("NCCL_NET"), flush=True)
print("== default after it (NCCL_NET=IB)", flush=True)
try:
    nccl.ncclCommInitRank(1, nccl.ncclGetUniqueId(), 0)
    print("default init ok", flush=True)
except RuntimeError as error:
    print("default init failed (no IB device in this container):", error, flush=True)
print("== subclass _init_comm with warm-up", flush=True)
cls = sp._socket_pynccl_class()
obj = cls.__new__(cls)
obj.nccl, obj.rank, obj.world_size = NCCLLibrary(), 0, 1
obj.available, obj.disabled, obj._suspended, obj.group = True, False, False, None
obj.nccl_version = obj.nccl.ncclGetRawVersion()
obj.unique_id = obj.nccl.ncclGetUniqueId()
obj._init_comm(torch.device("cuda:0"))
print("subclass comm", obj.comm, "method restored:", "ncclCommInitRank" not in vars(obj.nccl), flush=True)
x = torch.ones(1280 * 256, device="cuda")
obj.all_reduce(x, out_tensor=x, stream=torch.cuda.current_stream())
torch.cuda.synchronize()
print("all_reduce ok", float(x[0]), flush=True)
