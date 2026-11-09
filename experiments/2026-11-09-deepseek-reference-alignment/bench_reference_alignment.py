"""Benchmark and bit-level verification for DeepSeek-V4.1-Flash reference alignment (#1018).

Compares the promoted 1107 baseline against the reference-aligned candidate across:
1. Routed MoE SwiGLU + MXFP8 cast (`moe_swiglu_cast`) vs DeepSeek `inference/model.py:Expert.forward`
   (`x = (F.silu(gate) * up * weight).bfloat16()` followed by `per_token_cast`) for TP4 (`intermediate=576`)
   and TP3 (`intermediate=768`) across decode (`block_M=16, rows=16`), mid (`block_M=32, rows=96`), and
   prefill (`block_M=64, rows=2048`) tile heights.
2. End-to-end routed MoE layer (`MoeExpertsV41Plan.run`: dispatch -> gate_up grouped GEMM -> swiglu_cast ->
   down grouped GEMM -> FP32 slot-order combine) vs DeepSeek `Expert.forward` + `MoE.forward` reference
   (`_arithmetic`) across input scales (`small_scale=0.1`, `normal=0.5`, `heavy_clamp=25.0`), plus CUDA
   event latency at `rows=12` (decode) and `rows=300` (prefill).
3. 10-layer residual MoE stack accumulation drift (`x = x + 0.15 * moe(x)`).
"""


import json
import math
import sys
import torch
import tilelang
from tilelang import language as T

sys.path.append("/opt/spark-ds41f/candidate/vllm")

from vllm.models.deepseek_v4_1.tilelang.gemm import SF_BLOCK, scale_words
from vllm.models.deepseek_v4_1.tilelang.linear import cast_mxfp8
from vllm.models.deepseek_v4_1.tilelang.moe import (
    MoeExpertsV41Plan,
    moe_swiglu_cast,
    pdl,
    workspace_bytes,
)
from vllm.models.deepseek_v4_1.tilelang.quant_common import (
    get_best_vectorize_size,
    get_cast_output_config,
    get_sf_and_inv,
)
from tests.kernels.moe.test_deepseek_v41_tilelang_moe import (
    LIMIT,
    _arithmetic,
    _error,
    _experts,
    _gen,
    _plan,
    _routes,
    _swiglu_case,
    _swiglu_outputs,
    _tile_kernels_swiglu,
)


def _bench_cuda_us(fn, warmup: int = 40, rep: int = 200) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    return {
        "p50_us": round(times[rep // 2], 3),
        "mean_us": round(sum(times) / rep, 3),
        "p90_us": round(times[int(rep * 0.9)], 3),
        "min_us": round(times[0], 3),
    }


def _bf16_ulp_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float | int]:
    au = a.bfloat16().view(torch.int16).to(torch.int32)
    bu = b.bfloat16().view(torch.int16).to(torch.int32)
    am = torch.where(au < 0, 0x8000 - au, au)
    bm = torch.where(bu < 0, 0x8000 - bu, bu)
    ulp = (am - bm).abs()
    diff = (a.float() - b.float()).abs()
    return {
        "total": int(a.numel()),
        "ulp_gt0": int((ulp > 0).sum().item()),
        "ulp_gt0_pct": round(100.0 * float((ulp > 0).sum().item()) / a.numel(), 4),
        "ulp_gt1": int((ulp > 1).sum().item()),
        "max_ulp": int(ulp.max().item()),
        "mae": float(diff.mean().item()),
        "max_abs": float(diff.max().item()),
        "rel_l1": float(_error(a, b)),
    }


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def moe_swiglu_cast_1107_baseline(intermediate, block_M, limit, rows_per_cta=None):
    """Exact 1107 baseline moe_swiglu_cast (prior to DeepSeek Expert.forward alignment)."""
    in_dtype = T.bfloat16
    out_config = get_cast_output_config("e4m3", (1, SF_BLOCK), False, True, True)
    warp = 32
    vec = max(
        math.gcd(intermediate, get_best_vectorize_size(in_dtype) * warp) // warp, 1
    )
    chunk = warp * vec
    warps = intermediate // chunk
    if rows_per_cta is None:
        rows_per_cta = 4 if warps >= 4 else 2
    threads = warp * warps
    reduce_rounds = (SF_BLOCK // vec).bit_length() - 1
    sf_lanes = SF_BLOCK // vec
    sf_width = 4 * scale_words(intermediate)
    cert_e = 32
    tiles = T.dynamic("tiles")
    rows = tiles * block_M

    @T.prim_func
    def moe_swiglu_cast_1107_kernel(
        H: T.Tensor((rows, 2 * intermediate), in_dtype),
        Out: T.Tensor((rows, intermediate), T.float8_e4m3fn),
        OutSF: T.Tensor((rows, sf_width), T.uint8),
        Order: T.Tensor((rows,), T.int32),
        Weights: T.Tensor((rows,), T.float32),
        TileExpert: T.Tensor((tiles,), T.int32),
    ):
        with T.Kernel(block_M // rows_per_cta, tiles, threads=threads) as (slab, t):
            tid = T.get_thread_binding()
            col = (tid // warp) * chunk + (tid % warp) * vec
            orders = T.alloc_local((rows_per_cta,), T.int32)
            weights = T.alloc_local((rows_per_cta,), T.float32)
            xl = T.alloc_local((rows_per_cta, vec), in_dtype)
            xr = T.alloc_local((rows_per_cta, vec), in_dtype)
            xl_f = T.alloc_local((vec,), T.float32)
            xr_f = T.alloc_local((vec,), T.float32)
            x = T.alloc_local((vec,), T.float32)
            stored = T.alloc_local((vec,), T.float8_e4m3fn)
            amax = T.alloc_local((2,), T.float32)
            cert_min_l = T.alloc_local((2,), T.float32)
            weight = T.alloc_var(T.float32)
            base = t * block_M + slab * rows_per_cta
            expert = TileExpert[t]
            if expert >= 0:
                for r in T.unroll(rows_per_cta):
                    orders[r] = Order[base + r]
                for r in T.unroll(rows_per_cta):
                    if orders[r] >= 0:
                        weights[r] = Weights[base + r]
            pdl.chain()
            if expert >= 0:
                for r in T.unroll(rows_per_cta):
                    if orders[r] >= 0:
                        for j in T.vectorized(vec):
                            xl[r, j] = H[base + r, col + j]
                        for j in T.vectorized(vec):
                            xr[r, j] = H[base + r, intermediate + col + j]
                for r in T.unroll(rows_per_cta):
                    row = base + r
                    if orders[r] >= 0:
                        weight = weights[r]
                        for j in T.vectorized(vec):
                            xl_f[j] = T.float32(xl[r, j])
                            xr_f[j] = T.float32(xr[r, j])
                        for j in T.unroll(vec):
                            val_l = T.alloc_var(T.float32, init=xl_f[j])
                            val_r = T.alloc_var(T.float32, init=xr_f[j])
                            if limit is not None:
                                val_l = T.Select(val_l > limit, limit, val_l)
                                val_r = T.Select(val_r > limit, limit, val_r)
                                val_r = T.Select(val_r < -limit, -limit, val_r)
                            val = T.alloc_var(
                                T.float32,
                                init=T.call_extern(
                                    T.float32,
                                    "__fdividef",
                                    val_l,
                                    1 + T.call_extern(T.float32, "__expf", -val_l),
                                )
                                * val_r,
                            )
                            val = val * weight
                            x[j] = val
                        cert_min_metric = T.alloc_var(T.uint32, init=0xFFFFFFFF)
                        cert_max_metric = T.alloc_var(T.uint32, init=0)
                        for j in T.vectorized(2):
                            cert_min_l[j] = T.float32(0)
                        for j in T.unroll(vec):
                            cert_min_l[j % 2] = T.min(cert_min_l[j % 2], xl_f[j])
                        cert_min_l[0] = T.min(cert_min_l[0], cert_min_l[1])
                        for j in T.unroll(vec):
                            cert_t = T.reinterpret(x[j], T.uint32) & T.uint32(0x7FFFF)
                            cert_min_metric = T.min(cert_min_metric, cert_t)
                            cert_max_metric = T.max(cert_max_metric, cert_t)
                        if (
                            T.all_sync(
                                (cert_min_l[0] > -16.0)
                                and (cert_min_metric > cert_e)
                                and (cert_max_metric < 0x80000 - cert_e)
                            )
                            == 0
                        ):
                            for j in T.unroll(vec):
                                val_l = T.alloc_var(T.float32, init=xl_f[j])
                                val_r = T.alloc_var(T.float32, init=xr_f[j])
                                if limit is not None:
                                    val_l = T.Select(val_l > limit, limit, val_l)
                                    val_r = T.Select(val_r > limit, limit, val_r)
                                    val_r = T.Select(val_r < -limit, -limit, val_r)
                                val = T.alloc_var(
                                    T.float32,
                                    init=(val_l / (1 + T.exp(-val_l))) * val_r,
                                )
                                val = val * weight
                                x[j] = val
                        for j in T.vectorized(2):
                            amax[j] = 0
                        for j in T.unroll(vec):
                            amax[j % 2] = T.max(amax[j % 2], T.abs(x[j]))
                        amax[0] = T.max(amax[0], amax[1])
                        for j in T.unroll(reduce_rounds):
                            amax[0] = T.max(amax[0], T.shfl_xor(amax[0], 1 << j))
                        sf, sf_inv = get_sf_and_inv(amax[0], out_config)
                        if (tid % warp) % sf_lanes == 0:
                            OutSF[row, col // SF_BLOCK] = sf
                        for j in T.vectorized(vec):
                            x[j] *= sf_inv
                        for j in T.vectorized(vec):
                            stored[j] = T.cast(x[j], T.float8_e4m3fn)
                        for j in T.vectorized(vec):
                            Out[row, col + j] = stored[j]

    return moe_swiglu_cast_1107_kernel


def bench_swiglu_section() -> dict:
    results = {}
    num_experts, topk = 384, 6
    for shape_name, inter in [("tp4_inter576", 576), ("tp3_inter768", 768)]:
        results[shape_name] = {}
        cols = inter // 32
        for tier_name, block_m, rows in [
            ("decode_blockM16_rows16", 16, 16),
            ("mid_blockM32_rows96", 32, 96),
            ("prefill_blockM64_rows2048", 64, 2048),
        ]:
            layout, h = _swiglu_case(rows, inter, num_experts, topk, block_M=block_m)
            live = layout["order"] >= 0
            live_rows = int(live.sum().item())
            order = layout["order"]
            w = layout["layout_weights"]
            te = layout["tile_expert"]
            psum = layout["psum"]

            ref_fn = _tile_kernels_swiglu(inter, num_experts, block_m)
            ref_q, ref_sf = _swiglu_outputs(h.shape[0], inter)
            ref_fn(h, ref_q, ref_sf, psum, w, 1.0, None)

            k_base = moe_swiglu_cast_1107_baseline(inter, block_m, LIMIT)
            bq, bsf = _swiglu_outputs(h.shape[0], inter)
            k_base(h, bq, bsf, order, w, te)

            k_cand = moe_swiglu_cast(inter, block_m, LIMIT)
            cq, csf = _swiglu_outputs(h.shape[0], inter)
            k_cand(h, cq, csf, order, w, te)
            torch.cuda.synchronize()

            base_q_diff = int((bq[live].view(torch.uint8) != ref_q[live].view(torch.uint8)).sum().item())
            base_s_diff = int((bsf[live, :cols] != ref_sf[live, :cols]).sum().item())
            cand_q_diff = int((cq[live].view(torch.uint8) != ref_q[live].view(torch.uint8)).sum().item())
            cand_s_diff = int((csf[live, :cols] != ref_sf[live, :cols]).sum().item())

            t_base = _bench_cuda_us(lambda: k_base(h, bq, bsf, order, w, te))
            t_cand = _bench_cuda_us(lambda: k_cand(h, cq, csf, order, w, te))

            results[shape_name][tier_name] = {
                "live_rows": live_rows,
                "live_elements": live_rows * inter,
                "live_scales": live_rows * cols,
                "baseline_1107": {
                    "fp8_byte_diffs_vs_deepseek_ref": base_q_diff,
                    "fp8_byte_diff_pct": round(100.0 * base_q_diff / (live_rows * inter), 4),
                    "scale_byte_diffs_vs_deepseek_ref": base_s_diff,
                    **t_base,
                },
                "candidate": {
                    "fp8_byte_diffs_vs_deepseek_ref": cand_q_diff,
                    "fp8_byte_diff_pct": round(100.0 * cand_q_diff / (live_rows * inter), 4),
                    "scale_byte_diffs_vs_deepseek_ref": cand_s_diff,
                    **t_cand,
                },
                "speedup": round(t_base["p50_us"] / t_cand["p50_us"], 4),
            }
    return results


def bench_e2e_moe_section() -> dict:
    hidden = 384
    inter = 576
    num_experts = 32
    topk = 6

    w13_packed, w13_sf, w13 = _experts(num_experts, 2 * inter, hidden, 1)
    w2_packed, w2_sf, w2 = _experts(num_experts, hidden, inter, 2)

    plan_cand = _plan(num_experts, topk, hidden, inter, swiglu_limit=LIMIT)
    weights_cand = plan_cand.prepare_weights(w13_packed, w13_sf, w2_packed, w2_sf)

    plan_base = _plan(num_experts, topk, hidden, inter, swiglu_limit=LIMIT)
    for attr, bm in [("swiglu", 16), ("mid_swiglu", 32), ("prefill_swiglu", 64)]:
        if hasattr(plan_base, attr):
            setattr(plan_base, attr, moe_swiglu_cast_1107_baseline(inter, bm, LIMIT))
    weights_base = plan_base.prepare_weights(w13_packed, w13_sf, w2_packed, w2_sf)

    regimes = {}
    for rows in (12, 300):
        ids, route_weights = _routes(rows, num_experts, topk, 4, torch.int32)
        ws_cand = torch.empty(
            workspace_bytes(plan_cand.workspace_specs(rows, torch.int32)),
            dtype=torch.uint8,
            device="cuda",
        )
        ws_base = torch.empty_like(ws_cand)
        regimes[f"rows{rows}"] = {}
        for name, scale in [("small_scale", 0.1), ("normal", 0.5), ("heavy_clamp", 25.0)]:
            x = (torch.randn((rows, hidden), generator=_gen(3), device="cuda") * scale).bfloat16()
            ref_exact = _arithmetic(x, ids, route_weights, w13, w2, inter)
            tokens = cast_mxfp8(x)
            out_base = torch.empty_like(x)
            out_cand = torch.empty_like(x)
            plan_base.run(tokens, ids, route_weights, weights_base, out_base, ws_base)
            plan_cand.run(tokens, ids, route_weights, weights_cand, out_cand, ws_cand)
            torch.cuda.synchronize()
            regimes[f"rows{rows}"][name] = {
                "baseline_1107_vs_deepseek_ref": _bf16_ulp_stats(out_base, ref_exact),
                "candidate_vs_deepseek_ref": _bf16_ulp_stats(out_cand, ref_exact),
            }

    latency = {}
    for rows in (12, 300):
        ids, route_weights = _routes(rows, num_experts, topk, 4, torch.int32)
        x = (torch.randn((rows, hidden), generator=_gen(3), device="cuda") * 0.5).bfloat16()
        tokens = cast_mxfp8(x)
        out_base = torch.empty_like(x)
        out_cand = torch.empty_like(x)
        ws_cand = torch.empty(
            workspace_bytes(plan_cand.workspace_specs(rows, torch.int32)),
            dtype=torch.uint8,
            device="cuda",
        )
        ws_base = torch.empty_like(ws_cand)
        t_base = _bench_cuda_us(
            lambda: plan_base.run(tokens, ids, route_weights, weights_base, out_base, ws_base)
        )
        t_cand = _bench_cuda_us(
            lambda: plan_cand.run(tokens, ids, route_weights, weights_cand, out_cand, ws_cand)
        )
        latency[f"rows{rows}"] = {
            "baseline_1107": t_base,
            "candidate": t_cand,
            "speedup": round(t_base["p50_us"] / t_cand["p50_us"], 4),
        }

    # 10-layer residual MoE stack accumulation drift at rows=12
    rows = 12
    x_init = (torch.randn((rows, hidden), generator=_gen(3), device="cuda") * 0.5).bfloat16()
    x_ref = x_init.clone()
    x_b = x_init.clone()
    x_c = x_init.clone()
    ws_cand = torch.empty(
        workspace_bytes(plan_cand.workspace_specs(rows, torch.int32)),
        dtype=torch.uint8,
        device="cuda",
    )
    ws_base = torch.empty_like(ws_cand)
    layer_trace = []
    for layer in range(10):
        ids_l, rw_l = _routes(rows, num_experts, topk, 100 + layer, torch.int32)
        y_ref = _arithmetic(x_ref, ids_l, rw_l, w13, w2, inter)
        yb = torch.empty_like(x_b)
        yc = torch.empty_like(x_c)
        plan_base.run(cast_mxfp8(x_b), ids_l, rw_l, weights_base, yb, ws_base)
        plan_cand.run(cast_mxfp8(x_c), ids_l, rw_l, weights_cand, yc, ws_cand)
        x_ref = (x_ref.float() + 0.15 * y_ref.float()).bfloat16()
        x_b = (x_b.float() + 0.15 * yb.float()).bfloat16()
        x_c = (x_c.float() + 0.15 * yc.float()).bfloat16()
        if layer in (0, 4, 9):
            layer_trace.append(
                {
                    "layer": layer + 1,
                    "baseline_1107_vs_deepseek_ref": _bf16_ulp_stats(x_b, x_ref),
                    "candidate_vs_deepseek_ref": _bf16_ulp_stats(x_c, x_ref),
                }
            )

    return {
        "accuracy_by_regime": regimes,
        "latency": latency,
        "stack_10_layer": layer_trace,
    }


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else None
    report = {
        "device": torch.cuda.get_device_name(0),
        "swiglu_cast": bench_swiglu_section(),
        "e2e_moe": bench_e2e_moe_section(),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
