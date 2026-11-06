"""Compile the SP WO local-exchange TileLang kernels for sm_121a without a GPU."""
import resource
import tilelang
import tilelang.language as T
import tvm
import vllm.tilelang_utils as tl
from vllm.platforms import current_platform

current_platform.device_type = "cpu"
tl.tilelang, tl.T = tilelang, T

from vllm.models.deepseek_v4_1.tilelang.collectives import rank_order_sum_kernel
from vllm.models.deepseek_v4_1.tilelang.columns import zero_packs_kernel
from vllm.models.deepseek_v4_1.tilelang.gemm import (
    fp8_prefill_config,
    fp8w_bf16_gemm_cast,
    mxfp8_gemm,
    splitk_reduce_cast,
)
from vllm.models.deepseek_v4_1.tilelang.wo import _WO_A_COMMON, _WO_A_SHARDS

target = tvm.target.Target({"kind": "cuda", "arch": "sm_121a"})

cases = [
    (
        "wo_a_prefill_g3",
        fp8w_bf16_gemm_cast,
        (1024, 4096, 3),
        dict(shards=_WO_A_SHARDS, block_M=96, **_WO_A_COMMON),
    ),
    (
        "wo_a_prefill_g2",
        fp8w_bf16_gemm_cast,
        (1024, 4096, 2),
        dict(shards=_WO_A_SHARDS, block_M=96, **_WO_A_COMMON),
    ),
    (
        "wo_b_prefill_g3_3072",
        mxfp8_gemm,
        (5120, 3072),
        dict(groups=1, **fp8_prefill_config(5120, 3072, 1)),
    ),
    (
        "wo_b_prefill_g2_2048",
        mxfp8_gemm,
        (5120, 2048),
        dict(groups=1, **fp8_prefill_config(5120, 2048, 1)),
    ),
    (
        "splitk_reduce_cast_g3",
        splitk_reduce_cast,
        (3072, _WO_A_SHARDS),
        {},
    ),
    (
        "splitk_reduce_cast_g2",
        splitk_reduce_cast,
        (2048, _WO_A_SHARDS),
        {},
    ),
    ("zero_packs", zero_packs_kernel, (), {}),
    ("rank_order_sum_tp3", rank_order_sum_kernel, (3,), {}),
    ("rank_order_sum_tp4", rank_order_sum_kernel, (4,), {}),
]

for name, jit_fn, args, kwargs in cases:
    with target, tvm.transform.PassContext(config=jit_fn.pass_configs):
        art = tilelang.lower(
            jit_fn.get_tir(*args, **kwargs),
            target=target,
            enable_device_compile=True,
        )
    assert art.kernel_source and "extern "C"" in art.kernel_source
    print("device_compile_passed", {"kernel": name, "source_chars": len(art.kernel_source)}, flush=True)
print("process_maxrss_kib", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
