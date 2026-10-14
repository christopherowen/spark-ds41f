"""relu4 probe: what the index quantizer (queries and keys share it) makes of a non-finite BF16
value, beside TileKernels' torch reference cast of the same rows. Report only, always exit 0:
the quantizer's NaN semantics are the owner's decision.

Reading the source predicts: r6k's group amax is ``max(amax, |x|)`` (``fmaxf``, drops NaN) and
its E2M1 code is a chain of ``|x| > threshold`` tests (all false for NaN), so a NaN becomes a
zero code under a finite scale; an infinity gives the UE8M0 scale 0xFF (NaN) for its group.
TileKernels' torch max keeps the NaN, and its UE8M0 round-up of a NaN scale gives byte 0x00.
"""
import sys
import traceback

import torch

# The image is r6k's, so the package's indexer module is r6k's quantizer.
from vllm.models.deepseek_v4_1.tilelang import indexer as r6k

DEVICE = "cuda"
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
GROUP = 32


def decode(packed, scale_bytes):
    u = packed.to(torch.uint8).long()
    codes = torch.stack([u & 15, u >> 4], -1).reshape(*packed.shape[:-1], -1)
    table = torch.tensor(E2M1 + tuple(-v for v in E2M1), device=packed.device)
    scale = torch.exp2(scale_bytes.float() - 127.0)
    scale = torch.where(scale_bytes == 0xFF, torch.nan, scale).repeat_interleave(GROUP, -1)
    return codes, table[codes] * scale


def report(label, codes, values, scale_bytes, cases):
    print(f"{label}", flush=True)
    for row, col, what in cases:
        g = col // GROUP
        group = values[row, g * GROUP : (g + 1) * GROUP]
        print(f"  row {row} ({what} at {col}): scale byte 0x{int(scale_bytes[row, g]):02X}, "
              f"code {int(codes[row, col])}, decoded {values[row, col].item()}, "
              f"group NaN {int(group.isnan().sum())}/{GROUP}, finite {int(group.isfinite().sum())}/{GROUP}, "
              f"group |max| {group[group.isfinite()].abs().max().item() if group.isfinite().any() else 'n/a'}",
              flush=True)


torch.manual_seed(5)
x = torch.randn(3, 128, device=DEVICE).bfloat16()
cases = ((0, 5, "clean"), (1, 5, "NaN"), (2, 70, "+inf"))
x[1, 5] = float("nan")
x[2, 70] = float("inf")

try:
    q = torch.empty(3, 64, dtype=torch.uint8, device=DEVICE)
    s = torch.empty(3, 4, dtype=torch.uint8, device=DEVICE)
    r6k.quantize_mxfp4(x, q, s)
    torch.cuda.synchronize()
    codes, values = decode(q, s)
    report("r6k quantize_mxfp4 (B12X's arithmetic)", codes, values, s, cases)
except Exception:
    print(f"r6k quantizer ERROR\n{traceback.format_exc(limit=3)}", flush=True)

try:
    from tile_kernels.torch.cast import cast

    packed, sf = cast(x, "e2m1", block_size=(1, GROUP), round_sf=True)
    scale_bytes = (sf.contiguous().view(torch.int32) >> 23) & 0xFF
    codes, values = decode(packed.view(torch.uint8), scale_bytes)
    report("TileKernels torch cast(e2m1, (1, 32), round_sf=True)", codes, values, scale_bytes, cases)
except Exception:
    print(f"TileKernels reference ERROR\n{traceback.format_exc(limit=3)}", flush=True)

sys.exit(0)
