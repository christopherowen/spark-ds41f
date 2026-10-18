"""The routed experts' grouped GEMM with 4-bit-coded weight scales against the uncoded one.

Weights: E2M1 codes uniform over the 16 values; UE8M0 scales with a base per row in
[110, 125] and deltas in [0, 8), the measured shape of DeepSeek-V4.1-Flash's expert
scales (a sample of 7 of 40 layers and the drafter: 17 values, 95% two of them, every row
within 8 of its maximum). Activations: TileKernels' MXFP8 cast of random rows. Layouts:
vLLM's MoE dispatch of random routes (384 experts, top 6).

For TP4 (576 columns a rank) and TP3 (768): the gate/up GEMM (N = 2I, K = 5120, gathered
rows) and the down GEMM (N = 5120, K = I, layout rows; TP4's has a 64-wide K tail), at the
decode captures and prefill chunks:
1. the coded GEMM's output bits equal the uncoded one's;
2. time per call, warm and cold, three alternating rounds.
Exit 0 only if every bit check passes.
"""
import statistics
import sys
import traceback

import torch

sys.path.insert(0, "/b")
from kbench import BUCKETS, DECODE, DEVICE, PREFILL, SLOWER, per_call  # noqa: E402

import tile_kernels  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang import prepare  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang.gemm import (  # noqa: E402
    pack_scale_codes,
    pack_scale_words,
)
from vllm.models.deepseek_v4_1.tilelang.linear import _calls  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang.moe import (  # noqa: E402
    MoeDispatchV41Plan,
    carve,
    grouped_fp8_fp4_gemm,
    tile_codes,
    tile_expert_weights,
    workspace_bytes,
)

EXPERTS, TOPK, HIDDEN, ALIGN, BLOCK_N = 384, 6, 5120, 32, 128
SHAPES = (("tp4", 576), ("tp3", 768))
gen = torch.Generator(device=DEVICE).manual_seed(5)
failures = 0


def weights(n, k):
    packed = torch.randint(0, 256, (EXPERTS, n, k // 2), generator=gen, device=DEVICE, dtype=torch.uint8)
    base = torch.randint(110, 126, (EXPERTS, n, 1), generator=gen, device=DEVICE, dtype=torch.int32)
    exps = (base + torch.randint(0, 8, (EXPERTS, n, k // 32), generator=gen, device=DEVICE,
                                 dtype=torch.int32)).to(torch.uint8)
    tiles, tails = tile_expert_weights(packed.view(torch.int8), BLOCK_N)
    codes, row_base = pack_scale_codes(exps)
    return tiles, tails, pack_scale_words(exps), tile_codes(codes, BLOCK_N), row_base


def layout(rows, dispatch):
    ids = torch.stack([torch.randperm(EXPERTS, generator=gen, device=DEVICE)[:TOPK]
                       for _ in range(rows)]).int().contiguous()
    w = (torch.rand((rows, TOPK), generator=gen, device=DEVICE) + 0.1).contiguous()
    specs = dispatch.workspace_specs(rows)
    return dispatch.dispatch(ids, w, carve(torch.empty(workspace_bytes(specs), dtype=torch.uint8,
                                                       device=DEVICE), specs))


def mxfp8(rows, k):
    x = (torch.randn((rows, k), generator=gen, device=DEVICE) * 0.5).bfloat16()
    xq, sf = tile_kernels.quant.per_token_cast(x, "e4m3", 32, round_sf=True, use_packed_ue8m0=True)
    return xq, sf.contiguous().view(torch.uint32)


dispatch = MoeDispatchV41Plan(EXPERTS, TOPK, ALIGN, HIDDEN, torch.int32, torch.bfloat16)
prepare.declare(lambda: _calls(dispatch.kernels()), bind=dispatch.bind)
for name, inter in SHAPES:
    for gemm, n, k, gather in (("gate_up", 2 * inter, HIDDEN, True), ("down", HIDDEN, inter, False)):
        tiles, tails, words, codes, base = weights(n, k)
        plain = grouped_fp8_fp4_gemm(EXPERTS, n, k, TOPK, gather, ALIGN, BLOCK_N)
        coded = grouped_fp8_fp4_gemm(EXPERTS, n, k, TOPK, gather, ALIGN, BLOCK_N, coded_scales=True)
        extra = (tails,) if tails is not None else ()
        print(f"== {name} {gemm}: N {n}, K {k}, scale bytes {words.numel() * 4} -> "
              f"{codes.numel() * 2 + base.numel()}", flush=True)
        print("rows   plain us (warm/cold)    coded us (warm/cold)   coded/plain (warm/cold)  bits", flush=True)
        ratios = {}
        for rows in DECODE + PREFILL:
            try:
                lay = layout(rows, dispatch)
                order, tile_expert = lay["order"], lay["tile_expert"]
                a_rows = rows if gather else order.shape[0]
                xq, x_words = mxfp8(a_rows, k)
                out1 = torch.zeros((order.shape[0], n), dtype=torch.bfloat16, device=DEVICE)
                out2 = torch.zeros_like(out1)
                run_plain = lambda: plain(xq, x_words, tiles, *extra, words, order, tile_expert, out1)
                run_coded = lambda: coded(xq, x_words, tiles, *extra, codes, base, order, tile_expert, out2)
                run_plain()
                run_coded()
                torch.cuda.synchronize()
                live = order >= 0
                bits = torch.equal(out1[live].view(torch.int16), out2[live].view(torch.int16))
                failures += not bits
                pt, ct, warm, cold = [], [], [], []
                for r in range(3):
                    if r % 2:
                        b, a = per_call(run_coded, rows), per_call(run_plain, rows)
                    else:
                        a, b = per_call(run_plain, rows), per_call(run_coded, rows)
                    pt.append(a)
                    ct.append(b)
                    warm.append(b[0] / a[0])
                    cold.append(b[1] / a[1])
                ratios[rows] = (statistics.median(warm), statistics.median(cold))
                pw, pc = statistics.median(t[0] for t in pt), statistics.median(t[1] for t in pt)
                cw, cc = statistics.median(t[0] for t in ct), statistics.median(t[1] for t in ct)
                print(f"{rows:5d}  {pw:9.2f} / {pc:9.2f}  {cw:9.2f} / {cc:9.2f}  "
                      f"{ratios[rows][0]:6.3f} / {ratios[rows][1]:6.3f}  {'equal' if bits else 'DIFFER'}",
                      flush=True)
            except Exception:
                failures += 1
                print(f"{rows:5d} ERROR\n{traceback.format_exc(limit=6)}", flush=True)
        for bucket, lo, hi in BUCKETS:
            sel = [v for kk, v in ratios.items() if lo <= kk <= hi]
            if sel:
                print(f"  {bucket}: coded/plain best {min(min(w, c) for w, c in sel):.3f}, "
                      f"worst {max(max(w, c) for w, c in sel):.3f} "
                      f"({'within' if max(max(w, c) for w, c in sel) <= SLOWER else 'OVER'} the floor)",
                      flush=True)

print("4 passed" if not failures else f"{failures} failed", flush=True)
sys.exit(1 if failures else 0)
