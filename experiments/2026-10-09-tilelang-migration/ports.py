"""The ports of the TileLang migration.

The vLLM migration branch (~/projects/vllm-ds41-tilelang-migration, from r6c's tree
BASE) holds module commits, then one switch commit per port:

- module commits add kernels, refactors that compile to the same code, and imports,
  and change nothing the model runs (the L2 prefetch family is chosen by environment);
- a switch commit routes one part of the model to its TileLang kernel.

Every arm mounts the modules; a port's arm adds its switch (or its environment) and
nothing else, so the switch is the only variable. The control mounts the modules
alone. Each switch applies to the modules on its own, without the others.

For each port: its switch commit, the environment its arm sets, and its kernel bundle
(the module files its tests and benches import, the vLLM tests and the bench scripts).
"""

VLLM_BRANCH = "tilelang-migration"
BASE = "125c404e4"  # r6c's vLLM tree, as the image ships it
MODULES = "69beaf0d2"  # the last module commit

PORTS = {
    # 1. C1: the CuTe DSL L2 weight prefetch -> TileLang (same work split and PTX).
    "prefetch": {
        "switch": None,
        "environment": {"VLLM_L2_PREFETCH_KERNELS": "tilelang"},
        "bundle": {
            "modules": (
                "vllm/models/glm5next/nvidia/l2_prefetch.py",
                "vllm/models/glm5next/nvidia/l2_prefetch_tilelang.py",
            ),
            "description": "L2 prefetch port: the TileLang family's GPU tests, then a "
            "read-after-prefetch comparison with the CuTe family.",
            "tests": ("tests/models/test_glm5next_l2_prefetch_tilelang.py",),
            "select": None,
            "scripts": ("bench_prefetch.py",),
        },
    },
    # 3. B5: the compressor's wkv/wgate projection, B12X bf16_gemv -> TileLang BF16 GEMM.
    "compressor-projection": {
        "switch": "47eaa1e78",
        "environment": {},
        "bundle": {
            "modules": (
                "vllm/models/deepseek_v4_1/tilelang/linear.py",
                "vllm/models/deepseek_v4_1/tilelang/gemm.py",
            ),
            "description": "B5 compressor projection: the split BF16 GEMM unit tests, then numerics, "
            "batch invariance and warm/cold CUDA-graph timing against B12X bf16_gemv at "
            "the TP4 serving shapes.",
            "tests": (
                "tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py",
            ),
            "select": "split_linear or bf16_gemm",
            "scripts": ("bench_compressor_projection.py",),
        },
    },
    # 3. B6: the DSpark drafter's context KV, B12X block_fp8_linear -> TileLang MXFP8 GEMM
    # over the fused Q-A/KV weight's KV rows.
    "context-kv": {
        "switch": "0ebb3001a",
        "environment": {},
        "bundle": {
            "modules": (
                "vllm/models/deepseek_v4_1/tilelang/linear.py",
                "vllm/models/deepseek_v4_1/tilelang/gemm.py",
            ),
            "description": "B6 DSpark context KV: the block-32 row-slice unit tests, numerics, batch "
            "invariance and warm/cold CUDA-graph timing against B12X block_fp8_linear, "
            "then a split-K sweep (shards, decode and prefill tiles) at 512 x 5120.",
            "tests": (
                "tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py",
            ),
            "select": "block32_rows or mxfp8",
            "scripts": ("bench_context_kv.py", "sweep_context_kv.py"),
        },
    },
    # 4. B7: the Engram gate, B12X run_engram_mix -> a TileLang kernel with DeepSeek's
    # arithmetic; the model passes the image-token mask instead of its complement.
    "engram-gate": {
        "switch": "0b503d01b",
        "environment": {},
        "bundle": {
            "modules": ("vllm/models/deepseek_v4_1/tilelang/engram.py",),
            "description": "B7 Engram gate: unit tests against FP64 and TileKernels, then numerics, "
            "batch invariance and warm/cold CUDA-graph timing against B12X "
            "run_engram_mix, with TileKernels' kernel and other block shapes.",
            "tests": ("tests/kernels/test_deepseek_v41_tilelang_engram.py",),
            "select": "engram_gate",
            "scripts": ("bench_engram_gate.py",),
        },
    },
    # 4. B2: RoPE, B12X rotary.rotate (out of place) -> a TileLang kernel with TileKernels'
    # arithmetic, in place on the last 64 columns, for all five attention roles.
    "rope": {
        "switch": "bb48517f1",
        "environment": {},
        "bundle": {
            "modules": ("vllm/models/deepseek_v4_1/tilelang/rope.py",),
            "description": "B2 RoPE: unit tests against TileKernels' apply_rotary and FP64, then "
            "numerics, row independence and warm/cold CUDA-graph timing against B12X "
            "rotary.rotate at the five TP4 attention roles.",
            "tests": ("tests/kernels/test_deepseek_v41_tilelang_rope.py",),
            "select": "rope",
            "scripts": ("bench_rope.py",),
        },
    },
    # 4. B9: the Engram hash, vLLM's metadata copy plus B12X's three Triton launches and
    # a copy per layer -> one TileLang launch for every layer.
    "engram-hash": {
        "switch": "6ae570a01",
        "environment": {},
        "bundle": {
            "modules": ("vllm/models/deepseek_v4_1/tilelang/engram_hash.py",),
            "description": "B9 Engram hash: unit tests against B12X's integer oracle, then exact "
            "equality with B12X's hash path and eager and CUDA-graph timing per step, "
            "decode and prefill.",
            "tests": ("tests/kernels/test_deepseek_v41_tilelang_engram_hash.py",),
            "select": "engram_hash",
            "scripts": ("bench_engram_hash.py",),
        },
    },
    # 5. B1: the attention's WO projection, B12X's fused wo_projection -> inverse RoPE in
    # place, grouped WO-A and WO-B as TileLang block-32 GEMMs.
    "wo-projection": {
        "switch": "96a684bae",
        "environment": {},
        "bundle": {
            "modules": (
                "vllm/models/deepseek_v4_1/tilelang/wo.py",
                "vllm/models/deepseek_v4_1/tilelang/linear.py",
                "vllm/models/deepseek_v4_1/tilelang/gemm.py",
                "vllm/models/deepseek_v4_1/tilelang/rope.py",
            ),
            "description": "B1 WO projection: unit tests (grouped GEMM equals per-group GEMMs, FP64), "
            "then numerics, batch invariance and warm/cold CUDA-graph timing against "
            "B12X's fused wo_projection, then decode-tile sweeps for WO-A and WO-B.",
            "tests": ("tests/kernels/quantization/test_deepseek_v41_tilelang_wo.py",),
            "select": None,
            "scripts": ("bench_wo.py", "sweep_wo.py", "breakdown_wo.py"),
        },
    },
    # 5. B8: the indexer's head-weight scale, B12X scale_index_weights -> folded exactly
    # into the TileLang projection's weight (no kernel).
    "index-weights": {
        "switch": "f4ae61cac",
        "environment": {},
        "bundle": {
            "modules": (
                "vllm/models/deepseek_v4_1/tilelang/linear.py",
                "vllm/models/deepseek_v4_1/tilelang/gemm.py",
            ),
            "description": "B8 index head weights: the scaled-projection unit test, then exact equality "
            "with the TileLang projection plus B12X's scale pass, and warm/cold CUDA-graph "
            "timing per step.",
            "tests": (
                "tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py",
            ),
            "select": "scaled_linear",
            "scripts": ("bench_index_weights.py",),
        },
    },
    # 9. D1: rows 65-128 (16-stream steps) on the production TileLang projections: decode
    # tiles for the fused Q-A/KV and shared experts, split-K for the router and the
    # indexer head weights; same bits, speed only.
    "decode-rows": {
        "switch": "f24c69936",
        "environment": {},
        "bundle": {
            "modules": (
                "vllm/models/deepseek_v4_1/tilelang/gemm.py",
                "vllm/models/deepseek_v4_1/tilelang/linear.py",
                "vllm/models/deepseek_v4_1/tilelang/router.py",
            ),
            "description": "Rows 65-128 on the production TileLang projections: prefill tiles "
            "against 64- and 32-row decode tiles (warm, cold, after their own prefetch), and the "
            "router's shard GEMM against split-K partials; bits and graph-replay repeatability "
            "checked, after the TileLang linear unit tests.",
            "tests": (
                "tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py",
                # The fused gate router takes the wider split rows (the r6d gate's
                # 65-row router test found a mask sized for 64).
                "tests/kernels/moe/test_deepseek_v41_tilelang_router.py",
            ),
            "select": None,
            "scripts": ("bench_decode_rows.py",),
        },
    },
    # 10. sparknet's one-shot all-reduce up to its 2 MiB capacity (every step below
    # 205 rows, where sequence parallelism starts) instead of NCCL above 1 MiB: the
    # environment half of C3, alone, to price it.
    "oneshot-dispatch": {
        "switch": None,
        "environment": {"SPARKNET_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES": "2097152"},
    },
    # 11. C3: one arithmetic for every hidden-state reduction. The one-shot all-reduce
    # to 2 MiB, and sequence parallelism's reduce-scatter with its arithmetic (FP32,
    # rank order, one rounding) instead of NCCL's ring sums.
    "collectives": {
        "switch": "f195fe2ef",
        "environment": {"SPARKNET_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES": "2097152"},
        "bundle": {
            "modules": ("vllm/models/deepseek_v4_1/tilelang/collectives.py",),
            "description": "C3 rank-order reduce-scatter: the exchange (simulated ranks, ring4 "
            "neighbour relay) and the TileLang sum's GPU tests, then the sum's bits, "
            "repeatability and time at TP4 sequence-parallel chunk sizes.",
            "tests": ("tests/kernels/test_deepseek_v41_tilelang_collectives.py",),
            "select": None,
            "scripts": ("bench_rank_order_sum.py",),
        },
    },
    # 12. Prefill chunks that end at absolute multiples of the long-prefill threshold
    # (8096: the 8192-token budget less 16 streams of 6 decode rows), so a long prompt
    # splits at the same tokens whatever else is served.
    "chunk-align": {
        "switch": "8cc9ed19c",
        "environment": {},
        "serve_args": {"--long-prefill-token-threshold": "8096"},
        "bundle": {
            "modules": ("vllm/v1/core/sched/scheduler.py",),
            "description": "Aligned prefill chunks: the scheduler's unit tests.",
            "tests": ("tests/v1/core/test_aligned_prefill_chunks.py",),
            "select": None,
            "scripts": (),
        },
    },
    # 13. B3: the KV-cache record writers, B12X write_cache -> TileLang (the same bytes).
    "cache-writer": {
        "switch": "f4da408a4",
        "environment": {},
        "bundle": {
            "modules": ("vllm/models/deepseek_v4_1/tilelang/cache_writer.py",),
            "description": "B3 cache writers: the record unit tests, then bytes against B12X's "
            "write_cache, repeatability and warm/cold CUDA-graph timing for the SWA and "
            "indexed records.",
            "tests": ("tests/kernels/test_deepseek_v41_tilelang_cache_writer.py",),
            "select": None,
            "scripts": ("bench_cache_writer.py",),
        },
    },
    # 14. NCCL point-to-point tuning for C3's exchange (send/recv only; NCCL's
    # collectives keep their settings): four channels per network peer and 512 KiB
    # chunks instead of two and 128 KiB.
    "p2p-channels": {
        "switch": None,
        "environment": {"NCCL_NCHANNELS_PER_NET_PEER": "4", "NCCL_P2P_NET_CHUNKSIZE": "524288"},
    },
}

# Arms of several ports together (their switches touch different files): whether the
# determinism fixes together make the outputs independent of the company they keep.
COMBOS = {
    "deterministic": ("collectives", "compressor-projection", "chunk-align"),
    "collectives-p2p": ("collectives", "p2p-channels"),
}

# Kernel bundles that are not ports (no switch, no arm): they measure the modules.
BUNDLES = {}

# sparknet's one-shot collectives: the image pins 0.2.0, which predates the TileLang
# family, so both arms mount sparknet main (b61660f) and differ only in the family.
SPARKNET_MAIN = {
    "revision": "b61660f",
    "mount": [
        "{home}/sparknet-main/sparknet",
        "/usr/local/lib/python3.12/dist-packages/sparknet",
        "ro",
    ],
}
