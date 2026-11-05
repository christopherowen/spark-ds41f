"""Compile the integer row map for SM121 without opening a GPU device."""
import resource

import tilelang
import tvm
from vllm.models.deepseek_v4_1.tilelang.ced import sp_indices_kernel

target = tvm.target.Target({"kind": "cuda", "arch": "sm_121"})
with target, tvm.transform.PassContext(config=sp_indices_kernel.pass_configs):
    artifact = tilelang.lower(
        sp_indices_kernel.get_tir(), target=target, enable_device_compile=True
    )
print("CUDA device compilation passed without GPU execution")
print("artifact_fields", [field for field in dir(artifact) if not field.startswith("_")])
print("process_maxrss_kib", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
