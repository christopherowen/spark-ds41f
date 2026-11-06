#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isolated GPU bit-exactness gate and 2-boot microbenchmark screen for SP Engram prefill.

Verifies 100% bit-identical outputs (int16 view equality across finite, signed-zero,
non-finite NaN/+Inf/-Inf, IMAGE_SENTINEL, and RankOrderParts inputs) between the
baseline full-row SP Engram path and the candidate SP Engram path (interleaved_sp_rows
+ local_sp_product, as well as sliced project_exchange_rows) on real sm_121a TileLang
kernels, and measures per-rank GPU kernel latencies under warm-L2 and 128 MiB L2-wash
modes across TP3 (2048, 4096, 6144) and TP4 (4096, 8192, 16384) prefill shapes.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from types import SimpleNamespace

import torch
from torch import nn

import vllm  # noqa: F401
from vllm.models.deepseek_v4_1 import sp_prefill
from vllm.models.deepseek_v4_1.common import engram as engram_ops
from vllm.models.deepseek_v4_1.tilelang import collectives, columns, mhc, pdl
from vllm.models.deepseek_v4_1.tilelang import engram as tl_engram
from vllm.models.deepseek_v4_1.tilelang import linear as tl_linear
from vllm.models.deepseek_v4_1.tilelang.engram_hash import IMAGE_SENTINEL
from vllm.models.deepseek_v4_1.tilelang.gemm import (
    DECODE_TILE_ROWS,
    SF_BLOCK,
    fp8_decode_config,
    fp8_prefill_config,
    mxfp8_gemm,
    mxfp8_gemm_decode,
    pack_scale_words,
    scale_words,
)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--boot", required=True, help="Boot identifier (e.g. boot1, boot2)")
p.add_argument("--repeats", type=int, default=15, help="Timed graph replays per case")
p.add_argument("--output", default="", help="Optional clean JSONL output path")
a = p.parse_args()
assert a.repeats >= 3
_out_fh = open(a.output, "w", encoding="utf-8") if a.output else None


def emit(**record) -> None:
    line = json.dumps(
        {
            "boot": a.boot,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **record,
        },
        sort_keys=True,
    )
    print(line, flush=True)
    if _out_fh is not None:
        _out_fh.write(line + "\n")
        _out_fh.flush()


def timings(call, wash_buf: torch.Tensor | None) -> list[float]:
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        if wash_buf is not None:
            wash_buf.add_(1.0)
        begin.record()
        call()
        end.record()
    samples: list[float] = []
    for _ in range(a.repeats):
        graph.replay()
        torch.cuda.synchronize()
        samples.append(float(begin.elapsed_time(end)))
    return samples


class _NativeBlock32WKV(engram_ops._PaddedColumnParallelLinear):
    """Real sm_121a TileLang block-32 FP8 _PaddedColumnParallelLinear for single-GPU
    multi-rank simulation."""

    def __init__(
        self,
        rank: int,
        world: int,
        block_cols: int,
        logical_cols: int,
        k: int,
        weight: torch.Tensor,
        scales_u8: torch.Tensor,
        prefill_kernel,
        decode_kernels: dict,
    ):
        nn.Module.__init__(self)
        self.tp_rank = rank
        self.tp_size = world
        self.output_size_per_partition = block_cols
        self.logical_output_size = logical_cols
        self.padded_output_size = world * block_cols
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight_scale_inv = nn.Parameter(scales_u8, requires_grad=False)
        self.tilelang_weight = self.weight.data
        self.tilelang_weight_sf = pack_scale_words(scales_u8)[: block_cols // 32]
        self.tilelang_groups = 1
        self.tilelang_mxfp8_input = False
        self.tilelang_shards = 1
        self.tilelang_split_rows = 0
        self.tilelang_wide = {}
        self.tilelang_decode_rows = 64
        self.tilelang_prefill = prefill_kernel
        self.tilelang_decode = decode_kernels
        self.quant_method = tl_linear.TileLangFP8LinearMethod(
            SimpleNamespace(weight_block_size=[32, 32], scale_fmt="ue8m0")
        )
        self.sp_shards = None


def main() -> int:
    device = torch.device("cuda")
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    tl_linear.current_preallocated_workspace = lambda: ws
    flush = torch.zeros(128 * 1024 * 1024 // 4, device=device, dtype=torch.float32)

    hidden = 5120  # DeepSeek-V4.1-Flash hidden_size = 5120 -> logical_cols = 25600
    hc_mult = 4
    logical_cols = (hc_mult + 1) * hidden  # 12800
    k_engram = 24 * 256  # 6144
    eps = 1e-6

    emit(
        kind="identity",
        gpu=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        seed=6144,
        repeats=a.repeats,
        cache_modes=["warm", "128MiB_wash"],
        scope="per-rank Engram prefill column interleave, block-32 FP8 wkv projection, and fused mHC-post gate on sm_121a TileLang kernels",
        timer="external CUDA event nodes inside CUDAGraph replay",
        geometry={
            "hidden_size": hidden,
            "hc_mult": hc_mult,
            "logical_cols": logical_cols,
            "engram_k": k_engram,
            "engram_layers": [1, 14],
        },
    )

    gen = torch.Generator(device=device).manual_seed(6144)

    shapes_by_world = {
        3: (2048, 4096, 6144),
        4: (4096, 8192, 16384),
    }

    for world, token_shapes in shapes_by_world.items():
        quantum = world * 32
        padded_cols = -(-logical_cols // quantum) * quantum
        block_cols = padded_cols // world  # 4288 for TP3, 3200 for TP4
        max_cols = (24 + world - 1) // world  # 8 for TP3, 6 for TP4

        prefill_kernel = mxfp8_gemm(
            block_cols,
            k_engram,
            **fp8_prefill_config(block_cols, k_engram, 1),
            groups=1,
        )
        decode_kernels = {
            bm: mxfp8_gemm_decode(
                block_cols,
                k_engram,
                **fp8_decode_config(block_cols, k_engram, bm, 1),
                padded_rows=True,
                groups=1,
            )
            for bm in DECODE_TILE_ROWS
        }

        wkvs: list[_NativeBlock32WKV] = []
        for r in range(world):
            w_r = torch.randint(
                0,
                120,
                (block_cols, k_engram),
                generator=gen,
                device=device,
                dtype=torch.uint8,
            ).view(torch.float8_e4m3fn)
            sf_r = torch.randint(
                124,
                129,
                (block_cols // 32, k_engram // 32),
                generator=gen,
                device=device,
                dtype=torch.uint8,
            )
            # Zero-fill padded output rows on the last rank (matching allow_tp_padding)
            valid_rows = max(0, min(block_cols, logical_cols - r * block_cols))
            if valid_rows < block_cols:
                w_r[valid_rows:].zero_()
                sf_r[valid_rows // 32 :].zero_()
            wkvs.append(
                _NativeBlock32WKV(
                    r,
                    world,
                    block_cols,
                    logical_cols,
                    k_engram,
                    w_r,
                    sf_r,
                    prefill_kernel,
                    decode_kernels,
                )
            )

        for r, wkv_r in enumerate(wkvs):
            wkv_r.sp_shards = tuple(
                (wkv_r.tilelang_weight, wkv_r.tilelang_weight_sf)
                if s == r
                else (
                    wkvs[s].tilelang_weight.clone(),
                    wkvs[s].tilelang_weight_sf.clone(),
                )
                for s in range(world)
            )
            assert (
                wkv_r.sp_shards[r][0].data_ptr()
                == wkv_r.tilelang_weight.data_ptr()
            )
            assert (
                wkv_r.sp_shards[r][1].data_ptr()
                == wkv_r.tilelang_weight_sf.data_ptr()
            )

        peer_shard_bytes = sum(
            w.nbytes + sf.nbytes
            for s, (w, sf) in enumerate(wkvs[0].sp_shards)
            if s != 0
        )
        emit(
            kind="memory",
            world=world,
            block_cols=block_cols,
            per_layer_peer_sp_shard_bytes=peer_shard_bytes,
            two_layers_peer_sp_shard_mib=round(
                2 * peer_shard_bytes / (1024 * 1024), 4
            ),
            own_shard_aliased=True,
        )

        norm_weights = (
            torch.randn((hc_mult, hidden), generator=gen, device=device, dtype=torch.bfloat16).float()
            * 0.25
        ).flatten()

        # 1. Bit-exactness gate across unaligned/non-finite probe and all production shapes
        gate_probes = [
            (206 if world == 3 else 205, True),
            *((t, False) for t in token_shapes),
        ]
        for probe_tokens, nonfinite in gate_probes:
            rank_staged = [
                (
                    torch.randn(
                        (probe_tokens, max_cols * 256),
                        generator=gen,
                        device=device,
                    )
                    * 0.5
                ).bfloat16()
                for _ in range(world)
            ]
            rank_staged[0][0, 0] = -0.0
            if nonfinite:
                rank_staged[0][1, 0] = float("nan")
                rank_staged[0][2, 1] = float("inf")
                rank_staged[0][3, 2] = -float("inf")

            # Simulate dim=-1 all_gather + concat_rank_shards for baseline interleaved_rows
            flat_shards = [s.view(-1, 256) for s in rank_staged]
            gathered_rank_major = torch.cat(flat_shards, dim=0).contiguous()
            full_concat = collectives.concat_rank_shards(
                gathered_rank_major, world, tuple(flat_shards[0].shape)
            )
            assert full_concat is not None
            full_rows = full_concat.view(probe_tokens, -1)[:, :k_engram].contiguous()

            # Baseline per-rank local_product on full_rows
            full_kv_by_rank = [
                wkvs[s].local_product(full_rows) for s in range(world)
            ]

            residual_full = (
                torch.randn(
                    (probe_tokens, hc_mult, hidden), generator=gen, device=device
                )
                * 0.25
            ).bfloat16()
            prev_a_full = (
                torch.randn((probe_tokens, hidden), generator=gen, device=device)
                * 0.25
            ).bfloat16()
            prev_b_full = (
                torch.randn((probe_tokens, hidden), generator=gen, device=device)
                * 0.25
            ).bfloat16()
            post_mix_full = torch.rand(
                (probe_tokens, hc_mult),
                generator=gen,
                device=device,
                dtype=torch.float32,
            )
            res_mix_full = torch.rand(
                (probe_tokens, hc_mult, hc_mult),
                generator=gen,
                device=device,
                dtype=torch.float32,
            )
            token_ids = torch.full(
                (probe_tokens,), 11, dtype=torch.int32, device=device
            )
            token_ids[min(5, probe_tokens - 1)] = IMAGE_SENTINEL

            for r in range(world):
                layout = sp_prefill.SPRows(probe_tokens, world, r)
                real = layout.real_rows
                local_rows = layout.local_rows
                start = layout.start

                # Exchange staged_rows for rank r
                def exchange_staged(
                    _tensor: torch.Tensor,
                    r_idx: int = r,
                    l_rows: int = local_rows,
                    rr: int = real,
                    st: int = start,
                ) -> torch.Tensor:
                    blocks = torch.empty(
                        (world, l_rows, max_cols * 256),
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    if rr > 0:
                        for s in range(world):
                            blocks[s, :rr].copy_(rank_staged[s][st : st + rr])
                    return blocks

                sp_prefill.SPRows.exchange_rows = (
                    lambda self_sp, tensor, _fn=exchange_staged: _fn(tensor)
                )
                rows_sp = engram_ops.interleaved_sp_rows(rank_staged[r], layout)
                rows_ref = full_rows[start : start + real]
                rows_equal = bool(
                    torch.equal(rows_sp.view(torch.int16), rows_ref.view(torch.int16))
                )
                assert rows_equal, f"interleaved_sp_rows mismatch on world={world} rank={r} T={probe_tokens}"

                kv_sp = wkvs[r].local_sp_product(rows_sp, local_rows)
                kv_ref = torch.empty(
                    (world, local_rows, block_cols),
                    dtype=torch.bfloat16,
                    device=device,
                )
                for s in range(world):
                    kv_ref[s, :real].copy_(full_kv_by_rank[s][start : start + real])
                kv_equal = bool(
                    torch.equal(
                        kv_sp[:, :real].view(torch.int16),
                        kv_ref[:, :real].view(torch.int16),
                    )
                )
                assert kv_equal, f"local_sp_product mismatch on world={world} rank={r} T={probe_tokens}"

                # End-to-end Engram.forward (sp_local_wkv) vs baseline full-row blocked gate
                local_res = layout.local_slice(residual_full)
                local_post = layout.local_slice(post_mix_full)
                local_comb = layout.local_slice(res_mix_full)
                local_parts_base = collectives.RankOrderParts(
                    torch.stack(
                        (
                            layout.local_slice(prev_a_full),
                            layout.local_slice(prev_b_full),
                        )
                    )
                )
                local_parts_cand = collectives.RankOrderParts(
                    torch.stack(
                        (
                            layout.local_slice(prev_a_full),
                            layout.local_slice(prev_b_full),
                        )
                    )
                )

                out_base = torch.empty_like(local_res)
                prev_sum_base = local_parts_base.sum()
                if real < local_rows:
                    mhc._mhc_post(
                        prev_sum_base[real:],
                        local_res[real:],
                        local_post[real:],
                        local_comb[real:],
                        out_base[real:],
                    )
                if real > 0:
                    tl_engram.engram_post_gate_blocked(
                        prev_sum_base[:real],
                        local_res[:real],
                        local_post[:real],
                        local_comb[:real],
                        kv_ref,
                        norm_weights,
                        token_ids[start : start + real],
                        out_base[:real],
                        eps,
                        hc_mult,
                    )

                engram_mod = SimpleNamespace(
                    embed_tokens=SimpleNamespace(
                        table_memory="device",
                        columns=tuple(range(r, 24, world)),
                    ),
                    staged_rows=rank_staged[r],
                    wkv=wkvs[r],
                    l2_prefetch_after_projection=None,
                    norm_weights=norm_weights,
                    eps=eps,
                    hc_mult=hc_mult,
                )
                out_cand = engram_ops.Engram.forward(
                    engram_mod,
                    local_res,
                    torch.zeros((probe_tokens, 24), dtype=torch.int64, device=device),
                    token_ids,
                    sp_rows=layout,
                    previous_output=local_parts_cand,
                    post_mix=local_post,
                    res_mix=local_comb,
                )
                out_equal = bool(
                    torch.equal(
                        out_cand.view(torch.int16), out_base.view(torch.int16)
                    )
                )
                finite_mask = torch.isfinite(out_base.float()) & torch.isfinite(
                    out_cand.float()
                )
                max_abs_diff = (
                    float(
                        (
                            out_cand.float()[finite_mask]
                            - out_base.float()[finite_mask]
                        )
                        .abs()
                        .max()
                        .item()
                    )
                    if finite_mask.any()
                    else 0.0
                )
                assert out_equal and max_abs_diff == 0.0
                emit(
                    kind="bit_exact",
                    world=world,
                    rank=r,
                    tokens=probe_tokens,
                    local_rows=local_rows,
                    real_rows=real,
                    nonfinite_probe=nonfinite,
                    interleaved_rows_int16_equal=rows_equal,
                    wkv_local_sp_product_int16_equal=kv_equal,
                    engram_forward_int16_equal=out_equal,
                    max_abs_diff=max_abs_diff,
                )

        # 2. Timed CUDAGraph kernel benchmarks across production prefill chunk sizes
        for tokens in token_shapes:
            r = 0
            layout = sp_prefill.SPRows(tokens, world, r)
            local_rows = layout.local_rows
            real = layout.real_rows

            # Allocate pre-staged buffers for CUDAGraph capture
            gathered_full = torch.randn(
                (world * tokens * max_cols, 256),
                generator=gen,
                device=device,
                dtype=torch.bfloat16,
            )
            exchanged_blocks = torch.randn(
                (world * real * max_cols, 256),
                generator=gen,
                device=device,
                dtype=torch.bfloat16,
            )
            full_rows_buf = torch.randn(
                (tokens, k_engram),
                generator=gen,
                device=device,
                dtype=torch.bfloat16,
            )
            local_rows_buf = full_rows_buf[:real].contiguous()
            kv_full_out = torch.empty(
                (tokens, block_cols), dtype=torch.bfloat16, device=device
            )
            kv_sp_blocks = torch.empty(
                (world, local_rows, block_cols),
                dtype=torch.bfloat16,
                device=device,
            )

            local_res = (
                torch.randn(
                    (local_rows, hc_mult, hidden), generator=gen, device=device
                )
                * 0.25
            ).bfloat16()
            prev_packed = (
                torch.randn(
                    (world, local_rows, hidden), generator=gen, device=device
                )
                * 0.25
            ).bfloat16()
            local_post = torch.rand(
                (local_rows, hc_mult),
                generator=gen,
                device=device,
                dtype=torch.float32,
            )
            local_comb = torch.rand(
                (local_rows, hc_mult, hc_mult),
                generator=gen,
                device=device,
                dtype=torch.float32,
            )
            local_ids = torch.full(
                (local_rows,), 11, dtype=torch.int32, device=device
            )
            out_buf = torch.empty_like(local_res)

            def run_base_interleave(
                gathered_full=gathered_full, tokens=tokens, max_cols=max_cols
            ):
                collectives.concat_rank_shards(
                    gathered_full, world, (tokens * max_cols, 256)
                )

            def run_cand_interleave(
                exchanged_blocks=exchanged_blocks, real=real, max_cols=max_cols
            ):
                collectives.concat_rank_shards(
                    exchanged_blocks, world, (real * max_cols, 256)
                )

            def run_base_wkv(
                wkv=wkvs[0], full_rows_buf=full_rows_buf, kv_full_out=kv_full_out
            ):
                wkv.local_product(full_rows_buf, out=kv_full_out)

            def run_cand_wkv(
                wkv=wkvs[0],
                local_rows_buf=local_rows_buf,
                kv_sp_blocks=kv_sp_blocks,
            ):
                wkv.quant_method.apply_shards(
                    wkv, local_rows_buf, wkv.sp_shards, kv_sp_blocks
                )

            def run_base_total(
                gathered_full=gathered_full,
                tokens=tokens,
                max_cols=max_cols,
                wkv=wkvs[0],
                kv_full_out=kv_full_out,
                prev_packed=prev_packed,
                local_res=local_res,
                local_post=local_post,
                local_comb=local_comb,
                kv_sp_blocks=kv_sp_blocks,
                local_ids=local_ids,
                out_buf=out_buf,
            ):
                rows_c = collectives.concat_rank_shards(
                    gathered_full, world, (tokens * max_cols, 256)
                ).view(tokens, k_engram)
                wkv.local_product(rows_c, out=kv_full_out)
                prev_sum = collectives.RankOrderParts(prev_packed).sum()
                tl_engram.engram_post_gate_blocked(
                    prev_sum,
                    local_res,
                    local_post,
                    local_comb,
                    kv_sp_blocks,
                    norm_weights,
                    local_ids,
                    out_buf,
                    eps,
                    hc_mult,
                )

            def run_cand_total(
                exchanged_blocks=exchanged_blocks,
                real=real,
                max_cols=max_cols,
                wkv=wkvs[0],
                kv_sp_blocks=kv_sp_blocks,
                prev_packed=prev_packed,
                local_res=local_res,
                local_post=local_post,
                local_comb=local_comb,
                local_ids=local_ids,
                out_buf=out_buf,
            ):
                rows_l = collectives.concat_rank_shards(
                    exchanged_blocks, world, (real * max_cols, 256)
                ).view(real, k_engram)
                wkv.quant_method.apply_shards(
                    wkv, rows_l, wkv.sp_shards, kv_sp_blocks
                )
                prev_sum = collectives.RankOrderParts(prev_packed).sum()
                tl_engram.engram_post_gate_blocked(
                    prev_sum,
                    local_res,
                    local_post,
                    local_comb,
                    kv_sp_blocks,
                    norm_weights,
                    local_ids,
                    out_buf,
                    eps,
                    hc_mult,
                )

            stage_pairs = [
                (
                    "interleave_concat",
                    [
                        ("baseline_full_rows", run_base_interleave),
                        ("candidate_sp_rows", run_cand_interleave),
                    ],
                ),
                (
                    "wkv_projection",
                    [
                        ("baseline_full_rows", run_base_wkv),
                        ("candidate_sp_rows", run_cand_wkv),
                    ],
                ),
                (
                    "engram_local_total",
                    [
                        ("baseline_full_rows", run_base_total),
                        ("candidate_sp_rows", run_cand_total),
                    ],
                ),
            ]

            for stage_name, pair in stage_pairs:
                variants = list(pair)
                if a.boot.endswith("2"):
                    variants.reverse()
                for cache_mode, wash_buf in (("warm", None), ("128MiB_wash", flush)):
                    for vname, fn in variants:
                        samples = timings(fn, wash_buf)
                        emit(
                            kind="timing",
                            stage=stage_name,
                            world=world,
                            rank=r,
                            tokens=tokens,
                            local_rows=local_rows,
                            real_rows=real,
                            cache_mode=cache_mode,
                            variant=vname,
                            mean_ms=round(statistics.mean(samples), 4),
                            median_ms=round(statistics.median(samples), 4),
                            stdev_ms=round(statistics.stdev(samples), 4),
                            min_ms=round(min(samples), 4),
                            max_ms=round(max(samples), 4),
                            samples_ms=[round(x, 4) for x in samples],
                        )

            del (
                gathered_full,
                exchanged_blocks,
                full_rows_buf,
                local_rows_buf,
                kv_full_out,
                kv_sp_blocks,
                local_res,
                prev_packed,
                local_post,
                local_comb,
                local_ids,
                out_buf,
            )

        del wkvs
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
