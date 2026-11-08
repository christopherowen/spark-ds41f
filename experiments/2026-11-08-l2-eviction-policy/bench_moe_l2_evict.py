#!/usr/bin/env python3
"""Benchmark TMA L2 evict_first replacement policy on routed MoE expert weights."""

import math
import sys

import torch
from vllm.models.deepseek_v4_1 import l2_prefetch
from vllm.models.deepseek_v4_1.tilelang.gemm import (
    mxfp8_gemm_decode,
    pack_weight_tiles,
    scale_words,
)
from vllm.models.deepseek_v4_1.tilelang.linear import MXFP8Rows, _launch, mxfp8_rows
from vllm.models.deepseek_v4_1.tilelang.moe import (
    MoeExpertsV41Plan,
    grouped_fp8_fp4_gemm,
    workspace_bytes,
)
from vllm.models.glm5next.nvidia import l2_prefetch as _l2pf


def build_plan(inter: int, eviction_policy: str | None = None, max_rows: int = 4096):
    plan = MoeExpertsV41Plan(
        num_experts=384,
        topk=6,
        hidden=5120,
        intermediate=inter,
        swiglu_limit=7.0,
        idx_dtypes=(torch.int32,),
        max_rows=max_rows,
    )
    calls = plan.kernels()
    if eviction_policy is not None:
        patched_calls = {}
        for k, (fn, args, kwargs) in calls.items():
            if fn is grouped_fp8_fp4_gemm and not kwargs.get("swizzle_panel"):
                kwargs = {**kwargs, "eviction_policy": eviction_policy}
            patched_calls[k] = (fn, args, kwargs)
        calls = patched_calls
    for name, kernel in _launch(calls).items():
        setattr(plan, name, kernel)
    for dispatch in plan._dispatches():
        dispatch.bind()
    return plan


def stats(times: list[float]) -> tuple[float, float, float]:
    times = sorted(times)
    n = len(times)
    med = times[n // 2]
    mean = sum(times) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in times) / max(n - 1, 1))
    ci95 = 1.96 * sd / math.sqrt(n)
    return med, mean, ci95


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda")
    l2_prefetch.warmup()

    for shape_name, I in [("TP3", 768), ("TP4", 576)]:
        print(f"\n================ {shape_name} (inter={I}) ================", flush=True)
        plan_base = build_plan(I, eviction_policy="evict_normal", max_rows=4096)
        plan_evict = build_plan(I, eviction_policy="evict_first", max_rows=4096)

        E, H = 384, 5120
        w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=device)
        w13_scale = torch.randint(120, 128, (E, 2 * I, H // 32), dtype=torch.uint8, device=device)
        w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=device)
        w2_scale = torch.randint(120, 128, (E, H, I // 32), dtype=torch.uint8, device=device)
        weights = plan_base.prepare_weights(w13, w13_scale, w2, w2_scale)

        for M in [1, 2, 4, 6, 8, 16, 32, 64, 128, 512, 1024, 2048, 4096]:
            padded = mxfp8_rows(M)
            x_bf16 = torch.randn((M, H), dtype=torch.bfloat16, device=device)
            x_val = torch.randint(0, 120, (padded, H), dtype=torch.uint8, device=device).view(
                torch.float8_e4m3fn
            )
            x_sf = torch.randint(
                120, 128, (padded, 4 * scale_words(H)), dtype=torch.uint8, device=device
            )
            x = MXFP8Rows(x_bf16, x_val, x_sf)
            topk_ids = torch.randint(0, E, (M, 6), dtype=torch.int32, device=device)
            topk_weights = torch.softmax(
                torch.randn((M, 6), dtype=torch.float32, device=device), dim=-1
            )
            shared_out = torch.randn((M, H), dtype=torch.bfloat16, device=device)
            ws_bytes = workspace_bytes(plan_base.workspace_specs(M, torch.int32))
            ws1 = torch.empty(ws_bytes, dtype=torch.uint8, device=device)
            ws2 = torch.empty(ws_bytes, dtype=torch.uint8, device=device)
            out1 = torch.empty((M, H), dtype=torch.bfloat16, device=device)
            out2 = torch.empty((M, H), dtype=torch.bfloat16, device=device)
            plan_base.run(
                x, topk_ids, topk_weights, weights, out1, workspace=ws1, shared=shared_out
            )
            plan_evict.run(
                x, topk_ids, topk_weights, weights, out2, workspace=ws2, shared=shared_out
            )
            torch.cuda.synchronize()
            assert not torch.isnan(out1).any(), f"NaN at {shape_name} M={M}"
            assert torch.equal(out1, out2), f"Mismatch at {shape_name} M={M}!"
        print(
            f"{shape_name}: 100% bit-identical (torch.equal) across all M=1..4096!",
            flush=True,
        )

        sh_w13 = torch.randint(0, 120, (2 * I, H), dtype=torch.uint8, device=device).view(
            torch.float8_e4m3fn
        )
        sh_w13_packed = pack_weight_tiles(sh_w13)
        sh_s13 = torch.randint(
            120, 128, ((2 * I) // 32, scale_words(H)), dtype=torch.int32, device=device
        ).view(torch.uint32)
        sh_w2 = torch.randint(0, 120, (H, I), dtype=torch.uint8, device=device).view(
            torch.float8_e4m3fn
        )
        sh_w2_arg = pack_weight_tiles(sh_w2) if (I % 128 == 0) else sh_w2
        sh_s2 = torch.randint(
            120, 128, (H // 32, scale_words(I)), dtype=torch.int32, device=device
        ).view(torch.uint32)

        sh_gu_decode = mxfp8_gemm_decode(
            2 * I, H, block_M=16, block_N=128, block_K=128, padded_rows=True, tile_contiguous=True
        )
        sh_dn_decode = mxfp8_gemm_decode(
            H,
            I,
            block_M=16,
            block_N=128,
            block_K=128 if I % 128 == 0 else 64,
            padded_rows=True,
            tile_contiguous=(I % 128 == 0),
        )

        print(f"\n--- {shape_name} Standalone MoE (Rotating 16 Layer Routings) ---", flush=True)
        print(
            f"{'M':>5} | {'Base med (mean+-ci95 us)':>24} | {'EvictFirst med (mean+-ci95 us)':>30} | {'Delta':>9}",
            flush=True,
        )
        for M in [1, 2, 4, 6, 8, 16, 32, 64, 128, 512, 1024, 2048, 4096]:
            padded = mxfp8_rows(M)
            x_bf16 = torch.randn((M, H), dtype=torch.bfloat16, device=device)
            x_val = torch.randint(0, 120, (padded, H), dtype=torch.uint8, device=device).view(
                torch.float8_e4m3fn
            )
            x_sf = torch.randint(
                120, 128, (padded, 4 * scale_words(H)), dtype=torch.uint8, device=device
            )
            x = MXFP8Rows(x_bf16, x_val, x_sf)
            n_layers = 16
            topk_ids_list = [
                (
                    (torch.arange(M * 6, device=device, dtype=torch.int32).view(M, 6) + l * 23)
                    % E
                ).contiguous()
                for l in range(n_layers)
            ]
            topk_weights = torch.softmax(
                torch.randn((M, 6), dtype=torch.float32, device=device), dim=-1
            )
            shared_out = torch.randn((M, H), dtype=torch.bfloat16, device=device)
            ws_bytes = workspace_bytes(plan_base.workspace_specs(M, torch.int32))
            ws = torch.empty(ws_bytes, dtype=torch.uint8, device=device)
            out = torch.empty((M, H), dtype=torch.bfloat16, device=device)

            def run_plan(plan, iters=64):
                for _ in range(16):
                    plan.run(
                        x,
                        topk_ids_list[_ % n_layers],
                        topk_weights,
                        weights,
                        out,
                        workspace=ws,
                        shared=shared_out,
                    )
                torch.cuda.synchronize()
                s_ev = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
                e_ev = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
                for i in range(iters):
                    s_ev[i].record()
                    plan.run(
                        x,
                        topk_ids_list[i % n_layers],
                        topk_weights,
                        weights,
                        out,
                        workspace=ws,
                        shared=shared_out,
                    )
                    e_ev[i].record()
                torch.cuda.synchronize()
                return stats([s.elapsed_time(e) * 1000.0 for s, e in zip(s_ev, e_ev)])

            iters = 64 if M <= 128 else 24
            b_med, b_mean, b_ci = run_plan(plan_base, iters)
            e_med, e_mean, e_ci = run_plan(plan_evict, iters)
            pct = (e_med - b_med) / b_med * 100.0
            print(
                f"{M:5d} | {b_med:8.2f} ({b_mean:8.2f}+-{b_ci:4.2f}) | {e_med:10.2f} ({e_mean:8.2f}+-{e_ci:4.2f})  | {pct:+8.2f}%",
                flush=True,
            )

        print(
            f"\n--- {shape_name} Full FFN Layer (L2 Prefetch Shared Experts + Routed MoE + Shared GEMMs + Residual) ---",
            flush=True,
        )
        print(
            f"{'M':>5} | {'Base med (mean+-ci95 us)':>24} | {'EvictFirst med (mean+-ci95 us)':>30} | {'Delta':>9}",
            flush=True,
        )
        for M in [1, 2, 4, 6, 8, 16, 32]:
            padded = mxfp8_rows(M)
            x_bf16 = torch.randn((M, H), dtype=torch.bfloat16, device=device)
            x_val = torch.randint(0, 120, (padded, H), dtype=torch.uint8, device=device).view(
                torch.float8_e4m3fn
            )
            x_sf = torch.randint(
                120, 128, (padded, 4 * scale_words(H)), dtype=torch.uint8, device=device
            )
            x = MXFP8Rows(x_bf16, x_val, x_sf)
            sh_inter_q = torch.randint(
                0, 120, (padded, I), dtype=torch.uint8, device=device
            ).view(torch.float8_e4m3fn)
            sh_inter_sf = torch.randint(
                120, 128, (padded, scale_words(I)), dtype=torch.int32, device=device
            ).view(torch.uint32)
            sh_gu_out = torch.empty((M, 2 * I), dtype=torch.bfloat16, device=device)
            shared_out = torch.empty((M, H), dtype=torch.bfloat16, device=device)
            residual = torch.randn((M, 4, H), dtype=torch.bfloat16, device=device)

            n_layers = 16
            topk_ids_list = [
                (
                    (torch.arange(M * 6, device=device, dtype=torch.int32).view(M, 6) + l * 23)
                    % E
                ).contiguous()
                for l in range(n_layers)
            ]
            topk_weights = torch.softmax(
                torch.randn((M, 6), dtype=torch.float32, device=device), dim=-1
            )
            ws_bytes = workspace_bytes(plan_base.workspace_specs(M, torch.int32))
            ws = torch.empty(ws_bytes, dtype=torch.uint8, device=device)
            out = torch.empty((M, H), dtype=torch.bfloat16, device=device)

            segs = [
                _l2pf.tensor_segment("sh_w13", sh_w13_packed, min_bytes=0),
                _l2pf.tensor_segment("sh_w2", sh_w2_arg, min_bytes=0),
                _l2pf.tensor_segment("residual", residual, min_bytes=0),
            ]
            pf_plan = l2_prefetch.make_plan(segs, l2_prefetch.BUDGET_FFN, device)

            def run_ffn_layer(plan, iters=64):
                for i in range(16):
                    l2_prefetch.issue(pf_plan, M)
                    l2_prefetch.join()
                    sh_gu_decode(
                        x.values, sh_w13_packed, x.scales.view(torch.uint32), sh_s13, sh_gu_out
                    )
                    plan.run(
                        x,
                        topk_ids_list[i % n_layers],
                        topk_weights,
                        weights,
                        out,
                        workspace=ws,
                        shared=shared_out,
                    )
                    sh_dn_decode(sh_inter_q, sh_w2_arg, sh_inter_sf, sh_s2, shared_out)
                torch.cuda.synchronize()
                s_ev = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
                e_ev = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
                for i in range(iters):
                    l2_prefetch.issue(pf_plan, M)
                    l2_prefetch.join()
                    s_ev[i].record()
                    sh_gu_decode(
                        x.values, sh_w13_packed, x.scales.view(torch.uint32), sh_s13, sh_gu_out
                    )
                    plan.run(
                        x,
                        topk_ids_list[i % n_layers],
                        topk_weights,
                        weights,
                        out,
                        workspace=ws,
                        shared=shared_out,
                    )
                    sh_dn_decode(sh_inter_q, sh_w2_arg, sh_inter_sf, sh_s2, shared_out)
                    e_ev[i].record()
                torch.cuda.synchronize()
                return stats([s.elapsed_time(e) * 1000.0 for s, e in zip(s_ev, e_ev)])

            b_med, b_mean, b_ci = run_ffn_layer(plan_base, 64)
            e_med, e_mean, e_ci = run_ffn_layer(plan_evict, 64)
            pct = (e_med - b_med) / b_med * 100.0
            print(
                f"{M:5d} | {b_med:8.2f} ({b_mean:8.2f}+-{b_ci:4.2f}) | {e_med:10.2f} ({e_mean:8.2f}+-{e_ci:4.2f})  | {pct:+8.2f}%",
                flush=True,
            )


if __name__ == "__main__":
    sys.exit(main())
