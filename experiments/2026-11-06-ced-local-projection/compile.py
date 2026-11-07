"""Compile the preserved five-part BF16 prefill projection without a GPU."""
import resource
import tilelang
import tvm
from vllm.models.deepseek_v4_1.tilelang.gemm import bf16_gemm

target = tvm.target.Target({"kind": "cuda", "arch": "sm_121a"})
for columns in (128, 512):
    with target, tvm.transform.PassContext(config=bf16_gemm.pass_configs):
        tilelang.lower(
            bf16_gemm.get_tir(columns, 5120, shards=5, blocked=True,
                              out_dtype="bfloat16"),
            target=target, enable_device_compile=True,
        )
    print("device_compile_passed", {"columns": columns, "K": 5120,
                                   "shards": 5, "blocked": True}, flush=True)
print("process_maxrss_kib", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
