#!/usr/bin/env python3
"""The PDL chain kernels' CUDA sources, lowered without a GPU, checked as the pytest
plugin checks them (tests/kernels/deepseek_v41_pdl_sources.py).

usage (in the serving image, the PDL branch's package files and the plugin mounted over
the tree; bundles/pdl-sources): python3 sources_cpu.py

Each call is a chain kernel at a TP4 decode shape (and the warp-specialized GEMMs, which
must have no PDL). TileLang lowers it for sm_121 with the factory's own pass configs and
without device compilation (tilelang.lower(enable_device_compile=False), as
tilelang.tools.compile_only does). Some factories need a GPU to choose their arguments,
so the bundle runs with one. One JSON line per call, then a summary; exit status 1 if any
call fails to lower or any source has a violation.
"""

import json
import sys

import tilelang
import torch
from tvm.target import Target

from tests.kernels.deepseek_v41_pdl_sources import WAIT, violations
from vllm.models.deepseek_v4_1.tilelang import (
    activation,
    cache_writer,
    compress,
    gemm,
    indexer,
    metadata,
    mhc,
    moe,
    rope,
    router,
)

TARGET = Target({"kind": "cuda", "arch": "sm_121"})
H = 5120


def calls():
    out = []

    def add(label, function, *args, **kwargs):
        out.append((label, function, args, kwargs))

    def add_calls(label, table):
        for key, value in table.items():
            if isinstance(value, dict):
                add_calls(f"{label} {key}", value)
            else:
                add(f"{label} {key}", value[0], *value[1], **value[2])

    add("mhc post_pre", mhc.project_streams, H, "post_pre", block_M=16, world=1)
    add("mhc pre", mhc.project_streams, H, "pre", block_M=16)
    add("mhc broadcast", mhc.project_streams, H, "broadcast", block_M=16)
    add(
        "mhc post_pre of 4 parts",
        mhc.project_streams,
        H,
        "post_pre",
        block_M=16,
        world=4,
    )
    add("mhc finalize", mhc.finalize, H, 4 * H, 1e-20, 1e-6, 20, 1e-20)
    add("mhc post", mhc.post_streams, H)
    add("mhc collapse", mhc.collapse_streams, H)
    add("mhc collapse column", mhc.collapse_streams, H, column=0)
    for n, k in ((1792, H), (8192, 1280), (4096, 1280), (1152, H), (H, 576), (H, 2048)):
        for rows in gemm.DECODE_TILE_ROWS:
            config = gemm.fp8_decode_config(n, k, rows)
            add(
                f"mxfp8 decode {n}x{k} @{rows}",
                gemm.mxfp8_gemm_decode,
                n,
                k,
                **config,
                padded_rows=True,
            )
    add("mxfp8 prefill tile (warp-specialized)", gemm.mxfp8_gemm, 1792, H, block_M=128, block_N=128,
        block_K=128, num_stages=2)  # fmt: skip
    add("mxfp8 split-K partials", gemm.mxfp8_gemm_partials, 512, H, 5, block_M=16, block_N=32, block_K=256,
        num_stages=3, threads=128)  # fmt: skip
    add("split-K reduce", gemm.splitk_reduce, 512, 5)
    add("split-K reduce and cast", gemm.splitk_reduce_cast, 2048, 4)
    add(
        "WO-A partials",
        gemm.fp8w_bf16_gemm_partials,
        1024,
        4096,
        2,
        4,
        block_M=16,
        block_N=64,
        num_stages=2,
    )
    add("WO-A with its cast", gemm.fp8w_bf16_gemm_cast, 1024, 4096, 2, shards=4, block_M=96, block_N=64,
        num_stages=2)  # fmt: skip
    add("BF16 partials (warp-specialized)", gemm.bf16_gemm_partials, 384, H, 8)
    add("BF16 split-K", gemm.bf16_gemm_splitk, 1024, H, 5, blocked=True)
    add("RoPE", rope.rope_kernel, 16, 512, 64, 1, False)
    add("RoPE inverse", rope.rope_kernel, 16, 512, 64, 1, True)
    for kind in ("swa", "indexed"):
        add_calls(
            f"cache writer {kind}",
            {"": cache_writer.writer_call(kind, 64, torch.int64)},
        )
    add("metadata builder", metadata.builder_kernel, 16, 64, 1, False)
    add("metadata builder circular", metadata.builder_kernel, 16, 64, 4, True)
    add("metadata chunk", metadata.chunk_kernel, 64, 128, 512, True, False)
    add(
        "metadata decode",
        metadata.decode_metadata_kernel,
        64,
        128,
        512,
        True,
        False,
        2048,
        64,
    )
    add("metadata pages", metadata.pages_kernel, 512, 64)
    add("metadata pages of 256", metadata.pages_kernel, 512, 256)
    add_calls("indexer", {"quantize": indexer.quantize_call(), "quantize paged": indexer.quantize_call(64),
                          "score": indexer._score_call(8192, 64, False)})  # fmt: skip
    add_calls("indexer top-k", dict(enumerate(indexer.IndexerTopK.calls(8192))))
    # The re-index: scores and selection over candidate block ids.
    add_calls("indexer candidates", dict(enumerate(indexer.IndexerV41Plan.calls(
        8192, 256, candidates=True, candidate_blocks=True))))  # fmt: skip
    add("compress gather", compress.gather_partial_state_kernel, 16, 64)
    for ratio in (1, 2):
        add(f"compress at ratio {ratio}", compress.compress_kernel, ratio, 16, 64)
    add("SwiGLU clamp", activation.swiglu_clamp_kernel, 576, 10.0)
    add("router partials", router.route_partials, 6, 384, 8, False, False)
    add("routing masks", router.routing_masks_kernel, None, True)
    for block_M in (32, 64):  # decode and prefill tiles
        add_calls(
            f"MoE dispatch @{block_M}",
            moe.MoeDispatchV41Plan(
                384, 6, block_M, H, torch.int64, torch.float32
            ).kernels(),
        )
    add_calls(
        "MoE experts",
        moe.MoeExpertsV41Plan(384, 6, H, 576, swiglu_limit=10.0).kernels(),
    )
    return out


def lower(function, args, kwargs):
    configs = {
        str(getattr(key, "value", key)): value
        for key, value in (function.pass_configs or {}).items()
    }
    prim = function.get_tir(*args, **kwargs)
    with tilelang.transform.PassContext(opt_level=3, config=configs), TARGET:
        artifact = tilelang.lower(prim, target=TARGET, enable_device_compile=False)
    warp_specialized = not any(
        "disable_warp_specialized" in key and value for key, value in configs.items()
    )
    return str(artifact.kernel_source), warp_specialized


def main() -> int:
    checked = waiting = failed = unlowered = 0
    for label, function, args, kwargs in calls():
        try:
            source, warp_specialized = lower(function, args, kwargs)
        except Exception as error:  # reported, and the run fails
            unlowered += 1
            print(
                json.dumps(
                    {
                        "call": label,
                        "unlowered": f"{type(error).__name__}: {error}"[:300],
                    }
                ),
                flush=True,
            )
            continue
        problems = violations(source, warp_specialized)
        kernels = source.count("__global__")
        checked += 1
        waiting += WAIT in source
        failed += bool(problems)
        print(json.dumps({"call": label, "kernels": kernels, "waits": source.count(WAIT),
                          "warp_specialized": warp_specialized, "violations": problems}), flush=True)  # fmt: skip
    print(
        json.dumps(
            {
                "checked": checked,
                "with_pdl": waiting,
                "with_violations": failed,
                "unlowered": unlowered,
            }
        )
    )
    return 1 if failed or unlowered else 0


if __name__ == "__main__":
    sys.exit(main())
