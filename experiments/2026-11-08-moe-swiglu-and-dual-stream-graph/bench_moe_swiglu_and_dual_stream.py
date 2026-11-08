#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark routed MoE moe_swiglu_cast and 2-stream shared_experts overlap.

Measures across TP=4 and TP=3 decode batch sizes M in (1, 2, 3, 6, 12, 24, 48):
1. Standalone moe_swiglu_cast (baseline 1107 vs candidate with cascaded empty-slab /
   1-row skip and 4-row CTA slabs through intermediate <= 1152), both eager and
   inside captured CUDA graphs, across intermediate in (576, 768, 1152, 1536),
   verifying 100% bit-exactness (torch.equal on both E4M3 activations and UE8M0 scales).
2. Full MoE FFN sublayer (_l2pf_ffn L2 prefetch + mhc_pre + TileLangGateRouter +
   SharedExperts + MoeExpertsV41Plan), both eager and inside captured CUDA graphs
   (both warm-L2 multi-rep graph replay and cold-L2 + _l2pf_ffn single-layer replay),
   comparing interleaved paired windows of:
   - serial_1stream: 1-stream serial execution (shared_experts before router + routed_moe)
   - baseline_1107: 1107 2-stream execution (where workspace_size == 0 launched
     shared_experts before _apply_quant_method, bypassing _post_gate_hook, and used
     baseline moe_swiglu_cast)
   - candidate_swiglu_only: 1107 stream schedule + candidate moe_swiglu_cast
   - candidate_full: candidate _post_gate_hook 2-stream overlap (with _input_ready_event
     recorded before partials_kernel for M <= 32) + candidate moe_swiglu_cast
   Verifies 100% bit-exactness (torch.equal, 0 uint16 diffs, max_abs_diff == 0.0) across
   all modes and graph replays.
"""

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import tilelang
from tilelang import language as T
import torch

from tests.kernels.deepseek_v41_pdl_chains import _Entry, _block32
from tests.kernels.deepseek_v41_pdl_sources import violations
from vllm.models.deepseek_v4_1 import l2_prefetch
from vllm.models.deepseek_v4_1.tilelang import linear, mhc, pdl, prepare
from vllm.models.deepseek_v4_1.tilelang.activation import swiglu_clamp_cast_kernel
from vllm.models.deepseek_v4_1.tilelang.linear import _calls
from vllm.models.deepseek_v4_1.tilelang.moe import (
    SF_BLOCK,
    MoeDispatchV41Plan,
    MoeExpertsV41Plan,
    _present,
    carve,
    moe_swiglu_cast as moe_swiglu_cast_candidate,
    scale_words,
    workspace_bytes,
)
from vllm.models.deepseek_v4_1.tilelang.quant_common import (
    get_best_vectorize_size,
    get_cast_output_config,
    get_sf_and_inv,
)
from vllm.models.deepseek_v4_1.tilelang.router import route_partials

DEVICE = torch.device("cuda:0")
E = 384
TOPK = 6
H = 5120
LIMIT = 10.0
BATCH_SIZES = (1, 2, 3, 6, 12, 24, 48)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def moe_swiglu_cast_baseline_1107(intermediate, block_M, limit, rows_per_cta=None):
    """Exact 1107 baseline moe_swiglu_cast kernel for side-by-side comparison."""
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
    def moe_swiglu_cast_kernel(
        H_in: T.Tensor((rows, 2 * intermediate), in_dtype),
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
                            xl[r, j] = H_in[base + r, col + j]
                        for j in T.vectorized(vec):
                            xr[r, j] = H_in[base + r, intermediate + col + j]
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

    return moe_swiglu_cast_kernel


def bench_standalone_swiglu(windows: int = 35, reps_per_graph: int = 20):
    results = {}
    dispatch = MoeDispatchV41Plan(E, TOPK, 16, 512, torch.int32, torch.bfloat16)
    prepare.declare(lambda: _calls(dispatch.kernels()), bind=dispatch.bind)

    for inter in (576, 768, 1152, 1536):
        k_base = moe_swiglu_cast_baseline_1107(inter, 16, LIMIT)
        k_cand = moe_swiglu_cast_candidate(inter, 16, LIMIT)
        src_cand = k_cand.get_kernel_source()
        pdl_violations = violations(src_cand, False)
        assert not pdl_violations, f"PDL violations for inter={inter}: {pdl_violations}"

        sf_w = (inter // 32 + 3) // 4 * 4
        inter_rows = {}
        for M in BATCH_SIZES:
            gen = torch.Generator(device=DEVICE).manual_seed(100 + M)
            ids = torch.stack(
                [torch.randperm(E, generator=gen, device=DEVICE)[:TOPK] for _ in range(M)]
            ).to(torch.int32)
            weights = (torch.rand((M, TOPK), generator=gen, device=DEVICE) + 0.1).float()
            specs = dispatch.workspace_specs(M)
            ws = torch.zeros(workspace_bytes(specs), dtype=torch.uint8, device=DEVICE)
            layout = dispatch.dispatch(ids, weights, carve(ws, specs))
            L = layout["order"].shape[0]
            h = (torch.randn((L, 2 * inter), generator=gen, device=DEVICE) * 4.0).bfloat16().contiguous()
            h[::7, : inter // 4] = -20.0

            hq_b = torch.zeros((L, inter), dtype=torch.float8_e4m3fn, device=DEVICE)
            sf_b = torch.zeros((L, sf_w), dtype=torch.uint8, device=DEVICE)
            hq_c = torch.zeros((L, inter), dtype=torch.float8_e4m3fn, device=DEVICE)
            sf_c = torch.zeros((L, sf_w), dtype=torch.uint8, device=DEVICE)

            k_base(h, hq_b, sf_b, layout["order"], layout["layout_weights"], layout["tile_expert"])
            k_cand(h, hq_c, sf_c, layout["order"], layout["layout_weights"], layout["tile_expert"])
            torch.cuda.synchronize()

            active_tile_mask = layout["tile_expert"].repeat_interleave(16) >= 0
            live = (layout["order"] >= 0) & active_tile_mask
            cols = inter // 32
            bit_equal_q = bool(torch.equal(hq_b[live].view(torch.uint8), hq_c[live].view(torch.uint8)))
            bit_equal_sf = bool(torch.equal(sf_b[live, :cols], sf_c[live, :cols]))
            assert bit_equal_q and bit_equal_sf, f"Mismatch at inter={inter}, M={M}"

            # Capture graphs and interleave windows
            g_base = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_base):
                for _ in range(reps_per_graph):
                    k_base(h, hq_b, sf_b, layout["order"], layout["layout_weights"], layout["tile_expert"])
            g_cand = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_cand):
                for _ in range(reps_per_graph):
                    k_cand(h, hq_c, sf_c, layout["order"], layout["layout_weights"], layout["tile_expert"])

            for _ in range(10):
                g_base.replay()
                g_cand.replay()
            torch.cuda.synchronize()

            times_base_g, times_cand_g = [], []
            times_base_e, times_cand_e = [], []
            for _ in range(windows):
                for lst, g in ((times_base_g, g_base), (times_cand_g, g_cand)):
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    g.replay()
                    e.record()
                    torch.cuda.synchronize()
                    lst.append(s.elapsed_time(e) * 1000.0 / reps_per_graph)
                for lst, fn, oq, osf in (
                    (times_base_e, k_base, hq_b, sf_b),
                    (times_cand_e, k_cand, hq_c, sf_c),
                ):
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    fn(h, oq, osf, layout["order"], layout["layout_weights"], layout["tile_expert"])
                    e.record()
                    torch.cuda.synchronize()
                    lst.append(s.elapsed_time(e) * 1000.0)

            times_base_g.sort()
            times_cand_g.sort()
            times_base_e.sort()
            times_cand_e.sort()
            tb_g = times_base_g[windows // 2]
            tc_g = times_cand_g[windows // 2]
            tb_e = times_base_e[windows // 2]
            tc_e = times_cand_e[windows // 2]

            inter_rows[str(M)] = {
                "layout_rows": L,
                "live_rows": int(live.sum().item()),
                "active_tiles": int((layout["tile_expert"] >= 0).sum().item()),
                "bit_equal_e4m3": bit_equal_q,
                "bit_equal_ue8m0": bit_equal_sf,
                "baseline_1107_eager_us": round(tb_e, 3),
                "candidate_eager_us": round(tc_e, 3),
                "baseline_1107_graph_us": round(tb_g, 3),
                "candidate_graph_us": round(tc_g, 3),
                "delta_graph_us": round(tc_g - tb_g, 3),
                "speedup_graph_pct": round(100.0 * (tb_g - tc_g) / tb_g, 2),
            }
        results[str(inter)] = inter_rows
    return results


def bench_full_ffn_sublayer(routed_I: int, shared_I: int, windows: int = 25, warm_reps: int = 10):
    l2_prefetch.warmup()
    l2_flush_buf = torch.ones((8 * 1024 * 1024,), dtype=torch.float32, device=DEVICE)
    l2_flush_out = torch.empty((1,), dtype=torch.float32, device=DEVICE)

    def prepare_l2_before_ffn(l2_ffn_plan, M):
        torch.sum(l2_flush_buf, dim=0, keepdim=True, out=l2_flush_out)
        l2_prefetch.issue(l2_ffn_plan, M)
        l2_prefetch.join()

    gen = torch.Generator(device=DEVICE).manual_seed(123)
    gate_layer = torch.nn.Module()
    gate_layer.weight = torch.nn.Parameter(
        torch.randn((E, H), generator=gen, device=DEVICE, dtype=torch.bfloat16) * 0.02,
        requires_grad=False,
    )
    linear.TileLangLinearMethod().declare_layer_kernels(gate_layer)
    bias = torch.randn((E,), generator=gen, device=DEVICE, dtype=torch.float32) * 0.1
    no_img = torch.zeros((64,), dtype=torch.bool, device=DEVICE)
    r_topk = route_partials(TOPK, E, gate_layer.tilelang_shards, False, False)

    sh_gate_up = _block32(2 * shared_I, H, gen, mxfp8_input=True)
    sh_swiglu = swiglu_clamp_cast_kernel(shared_I, LIMIT)
    sh_down = _block32(H, shared_I, gen, mxfp8_input=True)

    w13 = torch.randint(0, 256, (E, 2 * routed_I, H // 2), generator=gen, dtype=torch.uint8, device=DEVICE)
    w13_sf = torch.randint(118, 122, (E, 2 * routed_I, H // 32), generator=gen, dtype=torch.uint8, device=DEVICE)
    w2 = torch.randint(0, 256, (E, H, routed_I // 2), generator=gen, dtype=torch.uint8, device=DEVICE)
    w2_sf = torch.randint(118, 122, (E, H, routed_I // 32), generator=gen, dtype=torch.uint8, device=DEVICE)

    plan_base = MoeExpertsV41Plan(
        E, TOPK, H, routed_I, swiglu_limit=LIMIT, out_dtype=torch.float32, max_rows=64
    )
    plan_base.declare_kernels()
    plan_base.swiglu = moe_swiglu_cast_baseline_1107(routed_I, 16, LIMIT)
    experts = plan_base.prepare_weights(w13, w13_sf, w2, w2_sf)

    plan_cand = MoeExpertsV41Plan(
        E, TOPK, H, routed_I, swiglu_limit=LIMIT, out_dtype=torch.float32, max_rows=64
    )
    plan_cand.declare_kernels()
    del w13, w13_sf, w2, w2_sf

    aux = torch.cuda.Stream()
    ev_in = torch.cuda.Event()
    ev_out = torch.cuda.Event()
    tp_results = {}

    for M in BATCH_SIZES:
        entry = _Entry(M, gen)
        x_in = torch.randn((M, H), generator=gen, device=DEVICE, dtype=torch.bfloat16) * 0.1
        streams_in = torch.randn((M, mhc.MULT * H), generator=gen, device=DEVICE, dtype=torch.bfloat16) * 0.1
        post_in = torch.rand((M, mhc.MULT), generator=gen, device=DEVICE) * 2
        comb_in = torch.softmax(torch.randn((M, mhc.MULT, mhc.MULT), generator=gen, device=DEVICE), dim=-1)
        pre_in = torch.softmax(torch.randn((M, mhc.MULT), generator=gen, device=DEVICE), dim=-1) + 1e-6

        ffn_segs = [
            ("mhc.fn", entry.fn.data_ptr(), entry.fn.numel() * entry.fn.element_size()),
            ("mhc.scale", entry.scale.data_ptr(), entry.scale.numel() * entry.scale.element_size()),
            ("mhc.base", entry.base.data_ptr(), entry.base.numel() * entry.base.element_size()),
            ("ffn_norm", entry.weight.data_ptr(), entry.weight.numel() * entry.weight.element_size()),
            ("gate", gate_layer.weight.data_ptr(), gate_layer.weight.numel() * gate_layer.weight.element_size()),
            ("gate_up", sh_gate_up.weight.data_ptr(), sh_gate_up.weight.numel() * sh_gate_up.weight.element_size()),
            ("down", sh_down.weight.data_ptr(), sh_down.weight.numel() * sh_down.weight.element_size()),
        ]
        l2_ffn_plan = l2_prefetch.make_plan(ffn_segs, l2_prefetch.BUDGET_FFN, DEVICE)

        partials = torch.empty((gate_layer.tilelang_shards, M, E), dtype=torch.float32, device=DEVICE)
        topk_ids = torch.empty((M, TOPK), dtype=torch.int64, device=DEVICE)
        topk_weights = torch.empty((M, TOPK), dtype=torch.float32, device=DEVICE)
        sh_h = torch.empty((M, 2 * shared_I), dtype=torch.bfloat16, device=DEVICE)
        sh_act = linear.empty_mxfp8(torch.empty((M, shared_I), dtype=torch.bfloat16, device=DEVICE))
        sh_out = torch.empty((M, H), dtype=torch.bfloat16, device=DEVICE)
        out = torch.empty((M, H), dtype=torch.bfloat16, device=DEVICE)
        ws = torch.zeros(workspace_bytes(plan_cand.workspace_specs(M, torch.int64)), dtype=torch.uint8, device=DEVICE)

        def run_entry():
            entry(x_in, streams_in, post_in, comb_in, pre_in)

        def run_shared():
            linear._block32_gemm(sh_gate_up, entry.mx.values, entry.mx.scales, sh_h, M)
            sh_swiglu(sh_h, sh_act.values[:M], sh_act.scales[:M])
            linear._block32_gemm(sh_down, sh_act.values, sh_act.scales, sh_out, M)

        def run_router_gate():
            linear.partials_kernel(gate_layer, M)(entry.y, gate_layer.weight, partials)

        def run_router_topk():
            r_topk(partials, bias, bias, no_img[:M], no_img[:M], topk_ids, topk_weights, 1.5)

        def run_routed(plan_obj, shared_cb=None):
            dispatch = plan_obj._dispatch_for(M, topk_ids.dtype)
            gate_up_k = plan_obj.late_gate_up if M >= 16 else plan_obj.gate_up
            swiglu_k = plan_obj.swiglu
            down_k = plan_obj.late_down if M >= 16 else plan_obj.down
            specs = plan_obj.workspace_specs(M, topk_ids.dtype)
            buffers = carve(ws, specs)
            count = len(dispatch.workspace_specs(M))
            h, hq, h_sf, y = buffers[count:]
            gate_up_w, down_w = experts
            layout = dispatch.dispatch(topk_ids, topk_weights, buffers[:count])
            order, tile_expert = layout["order"], layout["tile_expert"]
            x_words, h_words = entry.mx.scales.view(torch.uint32), h_sf.view(torch.uint32)
            gate_up_k(entry.mx.values, x_words, *_present(gate_up_w), order, tile_expert, h)
            swiglu_k(h, hq, h_sf, order, layout["layout_weights"], tile_expert)
            down_k(hq, h_words, *_present(down_w), order, tile_expert, y)
            sh = shared_cb() if callable(shared_cb) else shared_cb
            dispatch.combine(y, layout, out, shared=sh)

        def fn_serial():
            run_entry()
            run_shared()
            run_router_gate()
            run_router_topk()
            run_routed(plan_base, shared_cb=sh_out)

        def fn_1107():
            main = torch.cuda.current_stream()
            run_entry()
            ev_in.record(main)
            with torch.cuda.stream(aux):
                ev_in.wait(aux)
                run_shared()
                ev_out.record(aux)
            run_router_gate()
            run_router_topk()

            def wait_sh():
                ev_out.wait(main)
                return sh_out

            run_routed(plan_base, shared_cb=wait_sh)

        def fn_cand_swiglu_only():
            main = torch.cuda.current_stream()
            run_entry()
            ev_in.record(main)
            with torch.cuda.stream(aux):
                ev_in.wait(aux)
                run_shared()
                ev_out.record(aux)
            run_router_gate()
            run_router_topk()

            def wait_sh():
                ev_out.wait(main)
                return sh_out

            run_routed(plan_cand, shared_cb=wait_sh)

        def fn_cand_full():
            main = torch.cuda.current_stream()
            run_entry()
            ev_in.record(main)
            if M > 32:
                with torch.cuda.stream(aux):
                    ev_in.wait(aux)
                    run_shared()
                    ev_out.record(aux)
                run_router_gate()
            else:
                run_router_gate()
                with torch.cuda.stream(aux):
                    ev_in.wait(aux)
                    run_shared()
                    ev_out.record(aux)
            run_router_topk()

            def wait_sh():
                ev_out.wait(main)
                return sh_out

            run_routed(plan_cand, shared_cb=wait_sh)

        modes = [
            ("serial_1stream", fn_serial),
            ("baseline_1107", fn_1107),
            ("candidate_swiglu_only", fn_cand_swiglu_only),
            ("candidate_full", fn_cand_full),
        ]

        # Eager bit-exactness check
        eager_outs = {}
        for name, fn in modes:
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
            eager_outs[name] = out.clone()

        ref_bits = eager_outs["serial_1stream"].view(torch.uint16)
        for name in ("baseline_1107", "candidate_swiglu_only", "candidate_full"):
            diff_u16 = int((eager_outs[name].view(torch.uint16) != ref_bits).sum().item())
            max_abs = float((eager_outs[name].float() - eager_outs["serial_1stream"].float()).abs().max().item())
            assert diff_u16 == 0 and max_abs == 0.0, f"Eager mismatch on {name} at M={M}: diff={diff_u16}"

        # Capture 1-rep graphs (for cold-L2 + _l2pf_ffn) and 10-rep graphs (for warm-L2)
        graphs_1rep = {}
        graphs_warm = {}
        for name, fn in modes:
            g1 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g1):
                fn()
            g1.replay()
            torch.cuda.synchronize()
            assert torch.equal(out.view(torch.uint16), ref_bits), f"1-rep graph mismatch on {name} at M={M}"
            graphs_1rep[name] = g1

            gw = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gw):
                for _ in range(warm_reps):
                    fn()
            gw.replay()
            torch.cuda.synchronize()
            assert torch.equal(out.view(torch.uint16), ref_bits), f"Warm graph mismatch on {name} at M={M}"
            graphs_warm[name] = gw

        times_warm = {name: [] for name, _ in modes}
        times_cold = {name: [] for name, _ in modes}

        # Interleaved windows across all 4 modes
        for _ in range(windows):
            for name, _ in modes:
                gw = graphs_warm[name]
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                gw.replay()
                e.record()
                torch.cuda.synchronize()
                times_warm[name].append(s.elapsed_time(e) * 1000.0 / warm_reps)

            for name, _ in modes:
                g1 = graphs_1rep[name]
                prepare_l2_before_ffn(l2_ffn_plan, M)
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                g1.replay()
                e.record()
                torch.cuda.synchronize()
                times_cold[name].append(s.elapsed_time(e) * 1000.0)

        m_entry = {
            "M": M,
            "eager_bit_equal_all_modes": True,
            "graph_bit_equal_all_modes": True,
            "max_abs_diff": 0.0,
            "uint16_diffs": 0,
        }
        for name, _ in modes:
            tw = sorted(times_warm[name])[windows // 2]
            tc = sorted(times_cold[name])[windows // 2]
            m_entry[f"{name}_warm_graph_us"] = round(tw, 2)
            m_entry[f"{name}_cold_l2pf_graph_us"] = round(tc, 2)

        m_entry["warm_delta_vs_1107_us"] = round(
            m_entry["candidate_full_warm_graph_us"] - m_entry["baseline_1107_warm_graph_us"], 2
        )
        m_entry["warm_delta_vs_serial_us"] = round(
            m_entry["candidate_full_warm_graph_us"] - m_entry["serial_1stream_warm_graph_us"], 2
        )
        m_entry["cold_l2pf_delta_vs_1107_us"] = round(
            m_entry["candidate_full_cold_l2pf_graph_us"] - m_entry["baseline_1107_cold_l2pf_graph_us"], 2
        )
        m_entry["cold_l2pf_delta_vs_serial_us"] = round(
            m_entry["candidate_full_cold_l2pf_graph_us"] - m_entry["serial_1stream_cold_l2pf_graph_us"], 2
        )
        m_entry["step_58layers_warm_delta_vs_1107_ms"] = round(
            58.0 * m_entry["warm_delta_vs_1107_us"] / 1000.0, 3
        )
        m_entry["step_58layers_warm_delta_vs_serial_ms"] = round(
            58.0 * m_entry["warm_delta_vs_serial_us"] / 1000.0, 3
        )
        m_entry["step_58layers_cold_l2pf_delta_vs_1107_ms"] = round(
            58.0 * m_entry["cold_l2pf_delta_vs_1107_us"] / 1000.0, 3
        )
        m_entry["step_58layers_cold_l2pf_delta_vs_serial_ms"] = round(
            58.0 * m_entry["cold_l2pf_delta_vs_serial_us"] / 1000.0, 3
        )
        tp_results[str(M)] = m_entry

    return tp_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    boot_id = subprocess.check_output(
        ["cat", "/proc/sys/kernel/random/boot_id"], text=True
    ).strip()
    uname = subprocess.check_output(["uname", "-r"], text=True).strip()
    gpu_info = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,temperature.gpu,clocks.sm",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()

    print(f"Running Boot {args.boot} benchmark (boot_id={boot_id}, gpu={gpu_info})...", flush=True)
    t0 = time.time()
    swiglu_results = bench_standalone_swiglu()
    tp4_ffn = bench_full_ffn_sublayer(routed_I=576, shared_I=1152)
    tp3_ffn = bench_full_ffn_sublayer(routed_I=768, shared_I=1536)
    elapsed_s = round(time.time() - t0, 2)

    payload = {
        "boot": args.boot,
        "boot_id": boot_id,
        "kernel_release": uname,
        "gpu": gpu_info,
        "elapsed_seconds": elapsed_s,
        "standalone_moe_swiglu_cast": swiglu_results,
        "full_ffn_sublayer": {
            "tp4_routed576_shared1152": tp4_ffn,
            "tp3_routed768_shared1536": tp3_ffn,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output} in {elapsed_s}s", flush=True)


if __name__ == "__main__":
    main()
