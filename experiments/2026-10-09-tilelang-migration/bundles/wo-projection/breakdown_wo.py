"""Where the TileLang WO projection's time goes, and prefill tiles for its GEMMs (TP4:
16 heads x 512 in 2 groups, rank 1024, hidden 5120).

1. Components at 1, 96, 512 and 2048 rows under CUDA graphs, warm and cold: inverse
   RoPE in place, the WO-A activation cast, the grouped WO-A GEMM, the WO-B cast and
   the WO-B GEMM, each alone. Warm parts each keep their own data in L2, which the
   whole call (over 24 MB from 512 rows) cannot, so warm parts sum below it.
2. Prefill tiles for WO-A (grouped) and WO-B at 512, 1024 and 2048 rows, warm and cold,
   with the bits of every tile checked against the current one. A tile that does
   not launch (shared memory) is skipped.
"""
import itertools
import sys

import tile_kernels
import torch

sys.path.insert(0, "/b")
from kbench import DEVICE, per_call  # noqa: E402

from vllm.models.deepseek_v4_1.tilelang import gemm as g  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang.rope import rope_  # noqa: E402

HEADS, HEAD_DIM, GROUPS, RANK, HIDDEN = 16, 512, 2, 1024, 5120
WIDTH = HEADS // GROUPS * HEAD_DIM
GEN = torch.Generator(device=DEVICE).manual_seed(121)
ROWS = (1, 96, 512, 2048)


def fp8(shape):
    return (torch.randn(shape, device=DEVICE, generator=GEN) * 0.5).to(torch.float8_e4m3fn)


def sfw(rows, k):
    return g.pack_scale_words(torch.randint(118, 123, (rows, k // 32), device=DEVICE, generator=GEN,
                                            dtype=torch.uint8), rows=rows)


wa, wa_sf = fp8((GROUPS * RANK, WIDTH)), sfw(GROUPS * RANK, WIDTH)
wb, wb_sf = fp8((HIDDEN, GROUPS * RANK)), sfw(HIDDEN, GROUPS * RANK)
inv = 1.0 / (10000 ** (torch.arange(0, 64, 2, device=DEVICE).double() / 64))
table = torch.cat(((torch.arange(1 << 16, device=DEVICE).double()[:, None] * inv).cos(),
                   (torch.arange(1 << 16, device=DEVICE).double()[:, None] * inv).sin()), dim=-1).float()
R = max(ROWS)
o = torch.randn((R, HEADS, HEAD_DIM), device=DEVICE, generator=GEN).bfloat16()
positions = torch.randint(0, 1 << 16, (R,), device=DEVICE, generator=GEN)
a = torch.empty((R, GROUPS * RANK), dtype=torch.bfloat16, device=DEVICE)
out = torch.empty((R, HIDDEN), dtype=torch.bfloat16, device=DEVICE)
xq_a = torch.empty((R, GROUPS * WIDTH), dtype=torch.float8_e4m3fn, device=DEVICE)
sf_a = torch.empty((R, 4 * g.scale_words(GROUPS * WIDTH)), dtype=torch.uint8, device=DEVICE)
xq_b = torch.empty((R, GROUPS * RANK), dtype=torch.float8_e4m3fn, device=DEVICE)
sf_b = torch.empty((R, 4 * g.scale_words(GROUPS * RANK)), dtype=torch.uint8, device=DEVICE)


def kernels_for(rows):
    if rows <= g.DECODE_ROWS:
        h = g.decode_tile_rows(rows)
        ka = g.mxfp8_gemm_decode(RANK, WIDTH, **g.fp8_decode_config(RANK, WIDTH, h, GROUPS), padded_rows=True,
                                 groups=GROUPS)
        kb = g.mxfp8_gemm_decode(HIDDEN, GROUPS * RANK, **g.fp8_decode_config(HIDDEN, GROUPS * RANK, h),
                                 padded_rows=True)
        return ka, kb, max(64, rows)
    if rows <= 128:  # two 64-row decode tiles
        ka = g.mxfp8_gemm_decode(RANK, WIDTH, **g.fp8_decode_config(RANK, WIDTH, 64, GROUPS), padded_rows=True,
                                 groups=GROUPS)
        kb = g.mxfp8_gemm_decode(HIDDEN, GROUPS * RANK, **g.fp8_decode_config(HIDDEN, GROUPS * RANK, 64),
                                 padded_rows=True)
        return ka, kb, -(-rows // 64) * 64
    ka = g.mxfp8_gemm(RANK, WIDTH, **g.fp8_prefill_config(RANK, WIDTH, GROUPS), groups=GROUPS)
    kb = g.mxfp8_gemm(HIDDEN, GROUPS * RANK, **g.fp8_prefill_config(HIDDEN, GROUPS * RANK))
    return ka, kb, rows


print("components, us warm/cold: rope | cast A | WO-A | cast B | WO-B | sum", flush=True)
for rows in ROWS:
    ka, kb, padded = kernels_for(rows)
    parts = {
        "rope": lambda: rope_(o[:rows], positions[:rows], table, 1, True),
        "cast A": lambda: tile_kernels.quant.per_token_cast(o[:rows].view(rows, -1), "e4m3", 32, round_sf=True,
                                                            use_packed_ue8m0=True, out=(xq_a[:rows], sf_a[:rows])),
        "WO-A": lambda: ka(xq_a[:padded], wa, sf_a[:padded].view(torch.uint32), wa_sf, a[:rows]),
        "cast B": lambda: tile_kernels.quant.per_token_cast(a[:rows], "e4m3", 32, round_sf=True,
                                                            use_packed_ue8m0=True, out=(xq_b[:rows], sf_b[:rows])),
        "WO-B": lambda: kb(xq_b[:padded], wb, sf_b[:padded].view(torch.uint32), wb_sf, out[:rows]),
    }
    times = {name: per_call(call, rows) for name, call in parts.items()}
    print(f"  rows {rows:5d}: " + " | ".join(f"{name} {w:.2f}/{c:.2f}" for name, (w, c) in times.items())
          + f" | sum {sum(w for w, _ in times.values()):.2f}/{sum(c for _, c in times.values()):.2f}", flush=True)

failures = []
for label, (n, k, groups, weight, weight_sf, xq, sf, dst) in {
        "WO-A": (RANK, WIDTH, GROUPS, wa, wa_sf, xq_a, sf_a, a),
        "WO-B": (HIDDEN, GROUPS * RANK, 1, wb, wb_sf, xq_b, sf_b, out)}.items():
    current = g.fp8_prefill_config(n, k, groups)
    configs = [dict(block_M=bm, block_N=bn, block_K=128, num_stages=st)
               for bm, bn, st in itertools.product((64, 128), (64, 128), (2, 3))]
    for rows in (512, 1024, 2048):
        reference = None
        results = {}
        base = g.mxfp8_gemm(n, k, **current, groups=groups)
        base(xq[:rows], weight, sf[:rows].view(torch.uint32), weight_sf, dst[:rows])
        torch.cuda.synchronize()
        reference = dst[:rows].clone()
        for cfg in configs:
            try:
                kernel = g.mxfp8_gemm(n, k, **cfg, swizzle_panel=current.get("swizzle_panel", 0), groups=groups)
                kernel(xq[:rows], weight, sf[:rows].view(torch.uint32), weight_sf, dst[:rows])
                torch.cuda.synchronize()
            except Exception as error:
                print(f"  {label} {cfg}: {type(error).__name__}: {str(error)[:80]}")
                continue
            if not torch.equal(dst[:rows].view(torch.int16), reference.view(torch.int16)):
                failures.append(f"{label} {rows} {cfg}: bits differ")
            results["m{block_M}-n{block_N}-st{num_stages}".format(**cfg)] = per_call(
                lambda kernel=kernel: kernel(xq[:rows], weight, sf[:rows].view(torch.uint32), weight_sf, dst[:rows]),
                rows)
        cur = "m{block_M}-n{block_N}-st{num_stages}".format(**current)
        ranked = sorted(results, key=lambda key: sum(results[key]))
        print(f"{label} prefill {rows} rows (current {cur}): "
              + ", ".join(f"{key} {results[key][0]:.1f}/{results[key][1]:.1f}" for key in ranked), flush=True)
for failure in failures:
    print("FAIL", failure)
sys.exit(1 if failures else 0)
