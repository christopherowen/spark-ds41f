# usage: python3 /h/h987b.py OUT.json
#
# h987b: h987.py's cells again (same seeds), for a second ABBA pair of boots, with two
# additions: CUPTI kernel time with a synchronize after each call (kernel_isolated_us;
# h987's back-to-back kernel time also counts a PDL kernel's early-launch wait), and
# WO-A alone, so the end-to-end cells can be decomposed.
#
# #987's WO-B bench, written for candidate-51's own marks because the PR's
# /tmp/bench_wo_b.py was never committed (only its receipts wo_b_boot{1,2}.json).
# It keeps the PR's workload and claim: WO-B (5120 x K, block-32 FP8 weight, MXFP8
# activations) at K = 3072 / 2048 / 1024 (TP3 ranks 0-1 / TP4 and TP3 rank 2 / TP8,
# WO groups 3 / 2 / 1) and the end-to-end TileLangWOProjection (WO-A + its cast + WO-B),
# at M = 64, 96 (unchanged tiles) and 97, 104, 107, 112, 120, 128 (the 128-row tile the PR
# adds), with WO-B bit-exact (torch.equal) against the prefill GEMM (mxfp8_gemm,
# fp8_prefill_config) as the PR's test checks it (padding rows poisoned).
# It runs the image's own kernels through the serving dispatch (linear._block32_gemm on
# TileLangWOProjection's WO-B, which picks the tile by decode_tile_rows), so the same
# script measures the base arm (WIDE_DECODE {96: 96}: 64-row tiles at 97..128) and the
# candidate arm (WIDE_DECODE {96: 96, 128: 128}).
# Timings: CUDA-graph replay L2-hot (20 back-to-back calls per graph) and L2-cold (one call
# per graph after a 256 MiB write), CUPTI kernel time, and the eager loop.
import json
import sys
import traceback

sys.path.insert(0, "/h")
import torch  # noqa: E402

from omlib import (  # noqa: E402
    eager_us, env_info, gpu_warmup, graph_cold_us, graph_hot_us, h, kernel_isolated_us, kernel_us,
)

from tests.kernels.deepseek_v41_pdl_chains import _Projection  # noqa: E402
from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang import gemm, linear  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang.wo import TileLangWOProjection  # noqa: E402

out_path = sys.argv[1]
HIDDEN = 5120
WO_A_K = 4096
MS = [64, 96, 97, 104, 107, 112, 120, 128]
res = {"env": env_info(), "wide_decode": {}, "cells": {}, "errors": []}
res["warmup_ms"] = gpu_warmup(250.0)


def save():
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)


def timings(fn):
    fn()
    torch.cuda.synchronize()
    return {
        "graph_hot": graph_hot_us(fn),
        "graph_cold": graph_cold_us(fn),
        "kernel": kernel_us(fn),
        "kernel_isolated_us": kernel_isolated_us(fn),
        "eager_us": eager_us(fn),
    }


with set_current_vllm_config(VllmConfig()):
    for groups, K in ((3, 3072), (2, 2048), (1, 1024)):
        key = f"groups_{groups}_{HIDDEN}x{K}"
        try:
            res["wide_decode"][key] = {str(a): b for a, b in gemm.WIDE_DECODE[(HIDDEN, K)].items()}
            gen = torch.Generator(device="cuda").manual_seed(987 + K)
            wo = TileLangWOProjection(
                _Projection(groups * 1024, WO_A_K, gen), _Projection(HIDDEN, K, gen), groups=groups
            )
            wo.declare_kernels()
            b = wo.b
            res["cells"][key] = {"tilelang_decode_rows": b.tilelang_decode_rows,
                                 "tilelang_wide": {str(a): v for a, v in b.tilelang_wide.items()}}
            prefill = gemm.mxfp8_gemm(HIDDEN, K, **gemm.fp8_prefill_config(HIDDEN, K))
            # WO-A writes the MXFP8 cast of 128 rows, as at the widest decode step.
            qb_full = (torch.randn((128, groups * WO_A_K), generator=gen, device="cuda") * 0.5).bfloat16()
            (xs, xd), (ss, sd) = linear._activation_specs(128, K, b.tilelang_decode_rows)
            xq_full = torch.empty(xs, dtype=xd, device="cuda")
            sf_full = torch.empty(ss, dtype=sd, device="cuda")
            parts_full = [torch.empty(s, dtype=d, device="cuda") for s, d in wo.a.partial_specs(128)]
            wo.a(qb_full, xq_full[:128], sf_full[:128], parts_full[0] if parts_full else None)
            torch.cuda.synchronize()
            ref_full = torch.empty((128, HIDDEN), dtype=torch.bfloat16, device="cuda")
            prefill(xq_full, b.weight, sf_full.view(torch.uint32), b.tilelang_weight_sf, ref_full)
            torch.cuda.synchronize()
            for M in MS:
                try:
                    cell = {"decode_tile_rows": gemm.decode_tile_rows(M, b.tilelang_wide)}
                    (xs, xd), (ss, sd) = linear._activation_specs(M, K, b.tilelang_decode_rows)
                    tile_rows = xs[0]
                    cell["activation_rows"] = tile_rows
                    # WO-B alone: the live rows from WO-A, padding rows poisoned (E4M3 NaN,
                    # UE8M0 0xFF), as test_wo_b_wide_decode_128_rows_matches_prefill does.
                    xq = xq_full[:tile_rows].clone()
                    sf = sf_full[:tile_rows].clone()
                    if M < tile_rows:
                        xq[M:].view(torch.uint8).fill_(0x7F)
                        sf[M:].fill_(0xFF)
                    out = torch.empty((M, HIDDEN), dtype=torch.bfloat16, device="cuda")
                    fb = lambda xq=xq, sf=sf, out=out, M=M: linear._block32_gemm(b, xq, sf, out, M)  # noqa: E731
                    fb()
                    torch.cuda.synchronize()
                    cell["wob_bit_exact_vs_prefill"] = bool(torch.equal(out, ref_full[:M]))
                    cell["wob_max_abs_diff_vs_prefill"] = float((out.float() - ref_full[:M].float()).abs().max())
                    cell["wob_out_hash"] = h(out)
                    cell["wob"] = timings(fb)
                    # the prefill GEMM on the live rows, for context
                    pout = torch.empty((M, HIDDEN), dtype=torch.bfloat16, device="cuda")
                    xqm, sfm = xq_full[:M].clone(), sf_full[:M].clone()
                    fp = lambda xqm=xqm, sfm=sfm, pout=pout: prefill(  # noqa: E731
                        xqm, b.weight, sfm.view(torch.uint32), b.tilelang_weight_sf, pout)
                    fp()
                    torch.cuda.synchronize()
                    cell["prefill_bit_exact_vs_prefill128"] = bool(torch.equal(pout, ref_full[:M]))
                    cell["prefill"] = {"graph_hot": graph_hot_us(fp), "eager_us": eager_us(fp)}
                    # end to end: WO-A (+ its cast) then WO-B, as LayerChain and the model run it
                    qb = qb_full[:M].contiguous()
                    (xs, xd), (ss, sd) = linear._activation_specs(M, K, b.tilelang_decode_rows)
                    xq2 = torch.empty(xs, dtype=xd, device="cuda")
                    sf2 = torch.empty(ss, dtype=sd, device="cuda")
                    parts = [torch.empty(s, dtype=d, device="cuda") for s, d in wo.a.partial_specs(M)]
                    out2 = torch.empty((M, HIDDEN), dtype=torch.bfloat16, device="cuda")

                    def fe(qb=qb, xq2=xq2, sf2=sf2, parts=parts, out2=out2, M=M):
                        wo.a(qb, xq2[:M], sf2[:M], parts[0] if parts else None)
                        linear._block32_gemm(b, xq2, sf2, out2, M)

                    fe()
                    torch.cuda.synchronize()
                    cell["e2e_bit_exact_vs_prefill"] = bool(torch.equal(out2, ref_full[:M]))
                    cell["e2e_out_hash"] = h(out2)
                    cell["e2e"] = timings(fe)
                    fa = lambda qb=qb, xq2=xq2, sf2=sf2, parts=parts, M=M: wo.a(  # noqa: E731
                        qb, xq2[:M], sf2[:M], parts[0] if parts else None)
                    cell["woa"] = timings(fa)
                    res["cells"][key][str(M)] = cell
                    print(f"[{key}] M={M} tile={cell['decode_tile_rows']} exact={cell['wob_bit_exact_vs_prefill']}/"
                          f"{cell['e2e_bit_exact_vs_prefill']} wob hot {cell['wob']['graph_hot']['median_us']} "
                          f"cold {cell['wob']['graph_cold']['median_us']} kern {cell['wob']['kernel']['per_call_us']} "
                          f"eager {cell['wob']['eager_us']} | e2e hot {cell['e2e']['graph_hot']['median_us']} "
                          f"cold {cell['e2e']['graph_cold']['median_us']} eager {cell['e2e']['eager_us']} "
                          f"| prefill hot {cell['prefill']['graph_hot']['median_us']} | woa hot {cell['woa']['graph_hot']['median_us']} "
                          f"| kiso wob {cell['wob']['kernel_isolated_us']} e2e {cell['e2e']['kernel_isolated_us']} "
                          f"woa {cell['woa']['kernel_isolated_us']}", flush=True)
                    save()
                except Exception:
                    tb = traceback.format_exc()
                    print(f"[error] {key} M={M}: {tb}", flush=True)
                    res["errors"].append({"where": f"{key} M={M}", "traceback": tb})
            del wo, b
            torch.cuda.empty_cache()
        except Exception:
            tb = traceback.format_exc()
            print(f"[error] {key}: {tb}", flush=True)
            res["errors"].append({"where": key, "traceback": tb})
save()
print(f"wrote {out_path}; errors: {len(res['errors'])}", flush=True)
