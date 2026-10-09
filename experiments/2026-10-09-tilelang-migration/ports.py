"""The ports of the TileLang migration: for each, its commit on the vLLM migration
branch, the vLLM files it changes, its kernel bundle (the vLLM tests and bench scripts
it runs), the environment its arm sets, and any package it mounts.

An arm is the r6c TP4 recipe plus exactly one port, taken at the port's own commit, so
a later port's edits to a shared file never reach an earlier port's arm. Ports are
measured against the control, never against each other.
"""

# vLLM migration branch (~/projects/vllm-ds41-tilelang-migration, from r6c's 125c404e4).
VLLM_BRANCH = "tilelang-migration"

PORTS = {
    # 1. The CuTe DSL L2 weight prefetch -> TileLang (same work split and PTX).
    "prefetch": {
        "commit": "44c44e6e3",
        "files": (
            "models/glm5next/nvidia/l2_prefetch.py",
            "models/glm5next/nvidia/l2_prefetch_tilelang.py",
        ),
        "bundle": {
            "description": "L2 prefetch port: the TileLang family's GPU tests, then a "
                           "read-after-prefetch comparison with the CuTe family.",
            "tests": ("tests/models/test_glm5next_l2_prefetch_tilelang.py",),
            "select": None,
            "scripts": ("bench_prefetch.py",),
        },
        "environment": {"VLLM_L2_PREFETCH_KERNELS": "tilelang"},
    },
    # 3. B5: the compressor's wkv/wgate projection, B12X bf16_gemv -> TileLang BF16 GEMM.
    "compressor-projection": {
        "commit": "1d3752533",
        "files": (
            "models/deepseek_v4_1/compressor.py",
            "models/deepseek_v4_1/tilelang/linear.py",
        ),
        "bundle": {
            "description": "B5 compressor projection: the split BF16 GEMM unit tests, then numerics, "
                           "batch invariance and warm/cold CUDA-graph timing against B12X bf16_gemv at "
                           "the TP4 serving shapes.",
            "tests": ("tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py",),
            "select": "split_linear or bf16_gemm",
            "scripts": ("bench_compressor_projection.py",),
        },
        "environment": {},
    },
    # 3. B6: the DSpark drafter's context KV, B12X block_fp8_linear -> TileLang MXFP8 GEMM
    # over the fused Q-A/KV weight's KV rows.
    "context-kv": {
        "commit": "e45271941",
        "files": (
            "models/deepseek_v4_1/nvidia/dspark.py",
            "models/deepseek_v4_1/tilelang/linear.py",
        ),
        "bundle": {
            "description": "B6 DSpark context KV: the block-32 row-slice unit tests, numerics, batch "
                           "invariance and warm/cold CUDA-graph timing against B12X block_fp8_linear, "
                           "then a decode-tile sweep at 512 x 5120.",
            "tests": ("tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py",),
            "select": "block32_rows or mxfp8",
            "scripts": ("bench_context_kv.py", "sweep_context_kv.py"),
        },
        "environment": {},
    },
    # 4. B7: the Engram gate, B12X run_engram_mix -> a TileLang kernel with DeepSeek's
    # arithmetic; the model passes the image-token mask instead of its complement.
    "engram-gate": {
        "commit": "00fd5e6aa",
        "files": (
            "models/deepseek_v4_1/tilelang/engram.py",
            "models/deepseek_v4_1/common/engram.py",
            "models/deepseek_v4_1/nvidia/model.py",
        ),
        "bundle": {
            "description": "B7 Engram gate: unit tests against FP64 and TileKernels, then numerics, "
                           "batch invariance and warm/cold CUDA-graph timing against B12X "
                           "run_engram_mix, with TileKernels' kernel and other block shapes.",
            "tests": ("tests/kernels/test_deepseek_v41_tilelang_engram.py",),
            "select": "engram_gate",
            "scripts": ("bench_engram_gate.py",),
        },
        "environment": {},
    },
    # 4. B2: RoPE, B12X rotary.rotate (out of place) -> a TileLang kernel with TileKernels'
    # arithmetic, in place on the last 64 columns, for all five attention roles.
    "rope": {
        "commit": "4e4af513b",
        "files": (
            "models/deepseek_v4_1/tilelang/rope.py",
            "models/deepseek_v4_1/attention.py",
        ),
        "bundle": {
            "description": "B2 RoPE: unit tests against TileKernels' apply_rotary and FP64, then "
                           "numerics, row independence and warm/cold CUDA-graph timing against B12X "
                           "rotary.rotate at the five TP4 attention roles.",
            "tests": ("tests/kernels/test_deepseek_v41_tilelang_rope.py",),
            "select": "rope",
            "scripts": ("bench_rope.py",),
        },
        "environment": {},
    },
    # 4. B9: the Engram hash, vLLM's metadata copy plus B12X's three Triton launches and
    # a copy per layer -> one TileLang launch for every layer.
    "engram-hash": {
        "commit": "004bf569c",
        "files": (
            "models/deepseek_v4_1/tilelang/engram_hash.py",
            "models/deepseek_v4_1/common/engram.py",
            "models/deepseek_v4_1/nvidia/model.py",
        ),
        "bundle": {
            "description": "B9 Engram hash: unit tests against B12X's integer oracle, then exact "
                           "equality with B12X's hash path and eager and CUDA-graph timing per step, "
                           "decode and prefill.",
            "tests": ("tests/kernels/test_deepseek_v41_tilelang_engram_hash.py",),
            "select": "engram_hash",
            "scripts": ("bench_engram_hash.py",),
        },
        "environment": {},
    },
}

# sparknet's one-shot collectives: the image pins 0.2.0, which predates the TileLang
# family, so both arms mount sparknet main (b61660f) and differ only in the family.
SPARKNET_MAIN = {
    "revision": "b61660f",
    "mount": ["{home}/sparknet-main/sparknet", "/usr/local/lib/python3.12/dist-packages/sparknet", "ro"],
}
