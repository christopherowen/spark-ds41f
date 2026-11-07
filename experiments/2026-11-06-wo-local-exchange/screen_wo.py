"""Isolated GPU bit-exactness gate and microbenchmark screen for SP WO local MXFP8 exchange.

Verifies 100% bit-identical outputs (int16 view equality across finite, signed-zero,
and non-finite NaN/+Inf/-Inf inputs) between the baseline full-row SP WO path and
the candidate MXFP8 o_lora local-exchange path on real sm_121a TileLang kernels,
and measures per-rank local compute latency under warm-L2 and 128 MiB L2-wash modes
across TP3 (2048, 4096, 6144, 8192) and TP4 (2048, 4096, 8192, 16384) prefill shapes.
"""
import argparse
import json
import statistics
import time

import torch
from torch import nn

import vllm  # noqa: F401
from vllm.models.deepseek_v4_1 import sp_prefill
from vllm.models.deepseek_v4_1.tilelang import collectives, columns
from vllm.models.deepseek_v4_1.tilelang import wo as tl_wo
from vllm.models.deepseek_v4_1.tilelang.gemm import (
    fp8_prefill_config,
    fp8w_bf16_gemm_cast,
    fp8w_bf16_gemm_partials,
    mxfp8_gemm,
    scale_words,
    splitk_reduce_cast,
)
from vllm.models.deepseek_v4_1.tilelang.wo import (
    _WO_A_COMMON,
    _WO_A_SHARDS,
    _WO_A_TILES,
    TileLangWOProjection,
    _PeerWeight,
)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--boot", required=True, help="Boot identifier (e.g. boot1, boot2)")
p.add_argument("--repeats", type=int, default=15, help="Timed graph replays per case")
a = p.parse_args()
assert a.repeats >= 3


def emit(**record):
    print(
        json.dumps(
            {
                "boot": a.boot,
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **record,
            }
        ),
        flush=True,
    )


def bind_native_proj(
    wo_a_mod: nn.Module,
    wo_b_mod: nn.Module,
    groups: int,
    sp_rank_groups: tuple[int, ...],
) -> TileLangWOProjection:
    proj = TileLangWOProjection(wo_a_mod, wo_b_mod, groups=groups)
    proj.enable_sp(sp_rank_groups)
    n_a, k_a = proj.a.weight.shape
    proj.a.kernels = {
        bucket: (
            fp8w_bf16_gemm_partials(
                n_a // groups,
                k_a,
                groups,
                _WO_A_SHARDS,
                block_M=tile["block_M"],
                **_WO_A_COMMON,
            )
            if tile["split"]
            else fp8w_bf16_gemm_cast(
                n_a // groups,
                k_a,
                groups,
                shards=_WO_A_SHARDS,
                block_M=tile["block_M"],
                **_WO_A_COMMON,
            )
        )
        for bucket, tile in _WO_A_TILES.items()
    }
    proj.a.reduce = splitk_reduce_cast(n_a, _WO_A_SHARDS)
    n_b, k_b = proj.b.weight.shape
    proj.b.tilelang_prefill = mxfp8_gemm(
        n_b, k_b, **fp8_prefill_config(n_b, k_b, 1), groups=1
    )
    proj.b.tilelang_decode = {
        rows: mxfp8_gemm(
            n_b,
            k_b,
            block_M=rows,
            block_N=64,
            block_K=128,
            num_stages=2,
            threads=128,
            shards=1,
            groups=1,
            padded_rows=True,
        )
        for rows in (16, 32, 64, 96)
    }
    proj.b.tilelang_wide = {96: 96}
    proj.b.tilelang_decode_rows = 128
    return proj


def timings(call, wash_buf: torch.Tensor | None):
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
    samples = []
    for _ in range(a.repeats):
        graph.replay()
        torch.cuda.synchronize()
        samples.append(begin.elapsed_time(end))
    return samples


def main() -> int:
    device = torch.device("cuda")
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    tl_wo.current_preallocated_workspace = lambda: ws
    flush = torch.zeros(128 * 1024 * 1024 // 4, device=device, dtype=torch.float32)

    emit(
        kind="identity",
        gpu=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        seed=4106,
        repeats=a.repeats,
        cache_modes=["warm", "128MiB_wash"],
        scope="per-rank WO-A + WO-B prefill compute and local shard orchestration; collective exchange simulated in-memory",
        timer="external CUDA event nodes inside CUDAGraph replay",
        geometry={
            "hidden_size": 5120,
            "o_lora_rank": 1024,
            "o_groups": 8,
            "head_dim": 512,
            "encoder_layers": 20,
            "total_layers": 40,
        },
    )

    hidden = 5120
    rank_width = 1024
    group_input_width = 4096  # 8 heads/group * 512 head_dim
    gen = torch.Generator(device=device).manual_seed(4106)

    shapes_by_world = {
        3: (2048, 4096, 6144, 8192),
        4: (2048, 4096, 8192, 16384),
    }

    for world, token_shapes in shapes_by_world.items():
        sp_rank_groups = (3, 3, 2) if world == 3 else (2, 2, 2, 2)
        projs: list[TileLangWOProjection] = []
        for r, groups in enumerate(sp_rank_groups):
            cols = groups * rank_width
            wo_a = nn.Module()
            wo_a.weight = nn.Parameter(
                torch.randint(
                    0, 120, (cols, group_input_width), generator=gen, device=device, dtype=torch.uint8
                ).view(torch.float8_e4m3fn),
                requires_grad=False,
            )
            wo_a.weight_scale_inv = nn.Parameter(
                torch.randint(
                    125,
                    129,
                    (cols // 32, group_input_width // 32),
                    generator=gen,
                    device=device,
                    dtype=torch.uint8,
                ),
                requires_grad=False,
            )
            wo_b = nn.Module()
            wo_b.weight = nn.Parameter(
                torch.randint(
                    0, 120, (hidden, cols), generator=gen, device=device, dtype=torch.uint8
                ).view(torch.float8_e4m3fn),
                requires_grad=False,
            )
            wo_b.weight_scale_inv = nn.Parameter(
                torch.randint(
                    125,
                    129,
                    (hidden // 32, cols // 32),
                    generator=gen,
                    device=device,
                    dtype=torch.uint8,
                ),
                requires_grad=False,
            )
            projs.append(bind_native_proj(wo_a, wo_b, groups, sp_rank_groups))

        prefill_by_cols = {
            peer_g * rank_width: mxfp8_gemm(
                hidden,
                peer_g * rank_width,
                **fp8_prefill_config(hidden, peer_g * rank_width, 1),
                groups=1,
            )
            for peer_g in dict.fromkeys(sp_rank_groups)
        }
        for r, proj in enumerate(projs):
            proj.sp_shards = tuple(
                proj.b
                if peer == r
                else _PeerWeight(
                    projs[peer].b.weight,
                    projs[peer].b.tilelang_weight_sf,
                    prefill_by_cols[sp_rank_groups[peer] * rank_width],
                )
                for peer in range(world)
            )

        # 1. Bit-exactness verification across aligned, unaligned, and non-finite token counts
        gate_probes = [
            (206 if world == 3 else 205, True),
            *((t, False) for t in token_shapes),
        ]
        for probe_tokens, nonfinite in gate_probes:
            sp0 = sp_prefill.SPRows(probe_tokens, world, 0)
            padded_rows = sp0.padded_rows
            local_rows = sp0.local_rows
            inputs = []
            full_outputs = []
            sent_xq = {}
            sent_sf = {}
            for r, groups in enumerate(sp_rank_groups):
                heads = groups * 8
                o_r = (
                    torch.randn((probe_tokens, heads, 512), generator=gen, device=device) * 0.5
                ).bfloat16()
                o_r[0, 0, 0] = -0.0
                if nonfinite:
                    o_r[1, 0, 0] = float("nan")
                    o_r[2, 0, 1] = float("inf")
                    o_r[3, 0, 2] = -float("inf")
                inputs.append(o_r)

                padded_out = torch.zeros((padded_rows, hidden), dtype=torch.bfloat16, device=device)
                projs[r](o_r, padded_out[:probe_tokens])
                if probe_tokens < padded_rows:
                    columns.zero_(padded_out[probe_tokens:])
                full_outputs.append(padded_out.view(world, local_rows, hidden))

                cols_r = projs[r].a.columns
                sf_bytes_r = 4 * scale_words(cols_r)
                xq_s = torch.empty((world, local_rows, cols_r), dtype=torch.float8_e4m3fn, device=device)
                sf_s = torch.empty((world, local_rows, sf_bytes_r), dtype=torch.uint8, device=device)
                projs[r].a(
                    o_r.view(probe_tokens, -1),
                    xq_s.view(padded_rows, cols_r)[:probe_tokens],
                    sf_s.view(padded_rows, sf_bytes_r)[:probe_tokens],
                    None,
                )
                if probe_tokens < padded_rows:
                    columns.zero_(xq_s.view(padded_rows, cols_r)[probe_tokens:])
                    columns.zero_(sf_s.view(padded_rows, sf_bytes_r)[probe_tokens:])
                sent_xq[r] = xq_s
                sent_sf[r] = sf_s

            orig_exchange = sp_prefill.SPRows.exchange_into

            def fake_exchange_into(self_sp, send, recv):
                r = self_sp.rank
                src = sent_xq if send.dtype == torch.float8_e4m3fn else sent_sf
                for peer in range(world):
                    if peer != r:
                        recv[peer].copy_(src[peer][r])

            sp_prefill.SPRows.exchange_into = fake_exchange_into
            try:
                for r, proj in enumerate(projs):
                    for unreduced in (True, False):
                        sp_r = sp_prefill.SPRows(probe_tokens, world, r, unreduced=unreduced)
                        got = proj.project_sp(inputs[r], sp_r)
                        expected_parts = torch.stack(
                            [full_outputs[peer][r] for peer in range(world)], dim=0
                        )
                        if unreduced:
                            if not torch.equal(got.view(torch.int16), expected_parts.view(torch.int16)):
                                raise AssertionError(
                                    f"TP{world} rank {r} unreduced parts mismatch at tokens={probe_tokens}"
                                )
                        else:
                            expected_sum = collectives.rank_order_sum(list(expected_parts.unbind(0)))
                            if not torch.equal(got.view(torch.int16), expected_sum.view(torch.int16)):
                                raise AssertionError(
                                    f"TP{world} rank {r} rank_order_sum mismatch at tokens={probe_tokens}"
                                )
            finally:
                sp_prefill.SPRows.exchange_into = orig_exchange

            del inputs, full_outputs, sent_xq, sent_sf
            emit(
                kind="gate",
                world=world,
                tokens=probe_tokens,
                nonfinite=nonfinite,
                bits_equal=True,
            )

        # 2. Per-rank compute timing across all token shapes
        distinct_ranks = (0, 2) if world == 3 else (0,)
        for r in distinct_ranks:
            proj = projs[r]
            groups = sp_rank_groups[r]
            cols_r = groups * rank_width
            sf_bytes_r = 4 * scale_words(cols_r)
            max_tokens = max(token_shapes)
            o_full = (
                torch.randn((max_tokens, groups * 8, 512), generator=gen, device=device) * 0.5
            ).bfloat16()

            for tokens in token_shapes:
                sp_r = sp_prefill.SPRows(tokens, world, r)
                local_rows = sp_r.local_rows
                padded_rows = sp_r.padded_rows
                real_rows = sp_r.real_rows
                o_r = o_full[:tokens]

                base_send = torch.empty((world, local_rows, hidden), dtype=torch.bfloat16, device=device)
                base_parts = torch.empty_like(base_send)
                bounds = [local_rows * i // 2 for i in range(3)]

                def run_baseline(
                    o_r=o_r,
                    base_send=base_send,
                    base_parts=base_parts,
                    bounds=bounds,
                    local_rows=local_rows,
                    tokens=tokens,
                ):
                    for lo, hi in zip(bounds, bounds[1:]):
                        for owner in range(world):
                            dest = base_parts if owner == r else base_send
                            start, stop = owner * local_rows + lo, owner * local_rows + hi
                            real = max(0, min(stop, tokens) - start)
                            if real:
                                proj(o_r[start : start + real], dest[owner, lo : lo + real])
                            if lo + real < hi:
                                columns.zero_(dest[owner, lo + real : hi])

                xq_send = torch.empty((world, local_rows, cols_r), dtype=torch.float8_e4m3fn, device=device)
                sf_send = torch.empty((world, local_rows, sf_bytes_r), dtype=torch.uint8, device=device)
                xq_peers = [
                    xq_send[r]
                    if peer == r
                    else torch.empty(
                        (local_rows, sp_rank_groups[peer] * rank_width),
                        dtype=torch.float8_e4m3fn,
                        device=device,
                    )
                    for peer in range(world)
                ]
                sf_peers = [
                    sf_send[r]
                    if peer == r
                    else torch.empty(
                        (local_rows, 4 * scale_words(sp_rank_groups[peer] * rank_width)),
                        dtype=torch.uint8,
                        device=device,
                    )
                    for peer in range(world)
                ]
                cand_parts = torch.empty((world, local_rows, hidden), dtype=torch.bfloat16, device=device)

                def run_candidate(
                    o_r=o_r,
                    xq_send=xq_send,
                    sf_send=sf_send,
                    xq_peers=xq_peers,
                    sf_peers=sf_peers,
                    cand_parts=cand_parts,
                    padded_rows=padded_rows,
                    local_rows=local_rows,
                    real_rows=real_rows,
                    tokens=tokens,
                ):
                    xq_flat = xq_send.view(padded_rows, cols_r)
                    sf_flat = sf_send.view(padded_rows, sf_bytes_r)
                    proj.a(o_r.view(tokens, -1), xq_flat[:tokens], sf_flat[:tokens], None)
                    if tokens < padded_rows:
                        columns.zero_(xq_flat[tokens:])
                        columns.zero_(sf_flat[tokens:])
                    proj._project_sp_shard(r, xq_send[r], sf_send[r], cand_parts[r], real_rows, local_rows)
                    for peer in range(world):
                        if peer != r:
                            proj._project_sp_shard(
                                peer, xq_peers[peer], sf_peers[peer], cand_parts[peer], real_rows, local_rows
                            )

                variants = [("parent_sliced_full_rows", run_baseline), ("candidate_wo_local_exchange", run_candidate)]
                if a.boot.endswith("2"):
                    variants = list(reversed(variants))

                for cache_mode, wash_buf in (("warm", None), ("128MiB_wash", flush)):
                    for vname, fn in variants:
                        samples = timings(fn, wash_buf)
                        emit(
                            kind="timing",
                            world=world,
                            rank=r,
                            groups=groups,
                            tokens=tokens,
                            local_rows=local_rows,
                            real_rows=real_rows,
                            cache_mode=cache_mode,
                            variant=vname,
                            mean_ms=statistics.mean(samples),
                            median_ms=statistics.median(samples),
                            stdev_ms=statistics.stdev(samples),
                            min_ms=min(samples),
                            max_ms=max(samples),
                            samples_ms=samples,
                        )
                del base_send, base_parts, xq_send, sf_send, xq_peers, sf_peers, cand_parts
            del o_full
        del projs
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
