"""TileKernels' routed-expert SwiGLU + MXFP8 cast (swiglu_forward_and_per_token_cast) at
serving shapes, as vLLM's TileLang MoE calls it (TP4: 576 columns a rank, 384 experts,
32-row alignment, limit 10, route weights, packed UE8M0 scales), on synthetic dispatch
layouts: expert segments start on 32-row boundaries and psum[e] marks where expert e's
real rows end, as the kernel's boundary walk expects.

1. Time per call of the original kernel at decode rows (6 routes a row) and prefill rows:
   where does its 23-30 us at decode go?
2. The same kernel with its two clamp sites as compare-and-select
   (tk_swiglu_select.py): bits on finite inputs against the original, time per call in
   three alternating rounds, and whether a NaN gate value now reaches the output (the
   original's fminf clamp turns it into 10).
Every shape is tried on its own; an error is printed, not fatal.
"""
import importlib.util
import statistics
import sys
import traceback

import torch

sys.path.insert(0, "/b")
from kbench import DEVICE, per_call  # noqa: E402

import tile_kernels  # noqa: E402
from tilelang import language as T  # noqa: E402
from tile_kernels.quant.common import get_cast_output_config, get_sf_shape  # noqa: E402
from tile_kernels.quant.swiglu_forward_cuda import get_swiglu_forward_and_per_token_cast_kernel_cuda as original  # noqa: E402

spec = importlib.util.spec_from_file_location("tk_swiglu_select", "/b/tk_swiglu_select.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
patched = module.get_swiglu_forward_and_per_token_cast_kernel_cuda

HIDDEN, EXPERTS, ALIGN, LIMIT, SF, TOPK = 576, 384, 32, 10.0, 32, 6
tile_kernels.config.set_token_alignment(ALIGN)
config = get_cast_output_config("e4m3", (1, SF), False, True, True)


def build(fn):
    return fn(HIDDEN, EXPERTS, ALIGN, True, False, LIMIT, False, in_dtype=T.dtype(torch.bfloat16),
              out_config=config, num_sms=tile_kernels.config.get_num_sms(), use_pdl=False)


def layout(routes, gen):
    """Random routing of ``routes`` slots over the experts, laid out with aligned segments."""
    experts = torch.randint(0, EXPERTS, (routes,), generator=gen, device="cpu")
    counts = torch.bincount(experts, minlength=EXPERTS).tolist()
    psum, start = [], 0
    for c in counts:
        psum.append(start + c)
        start = (start + c + ALIGN - 1) // ALIGN * ALIGN
    rows = max(start, ALIGN)
    real = torch.zeros(rows, dtype=torch.bool)
    begin = 0
    for c in counts:
        real[begin:begin + c] = True
        begin = (begin + c + ALIGN - 1) // ALIGN * ALIGN
    return torch.tensor(psum, dtype=torch.int32, device=DEVICE), rows, real.to(DEVICE)


def outputs(rows):
    out = torch.zeros((rows, HIDDEN), dtype=torch.float8_e4m3fn, device=DEVICE)
    shape = get_sf_shape((rows, HIDDEN), config)
    sf = torch.zeros(shape, dtype=torch.uint8, device=DEVICE)  # the kernel takes uint8 scale storage
    return out, sf


gen = torch.Generator(device="cpu").manual_seed(29)
k_orig, k_new = build(original), build(patched)
print("rows routes padded  original us (warm/cold)  select/original median ratio (warm/cold)  bits  NaN reaches output (orig/select)", flush=True)
for rows in (1, 2, 6, 12, 24, 48, 96, 512, 2048, 8192):
    try:
        routes = rows * TOPK
        psum, padded, real = layout(routes, gen)
        x = (torch.randn((padded, 2 * HIDDEN), device=DEVICE) * 4).bfloat16()
        w = torch.rand((padded,), device=DEVICE)
        o1, s1 = outputs(padded)
        o2, s2 = outputs(padded)
        run1 = lambda: k_orig(x, o1, s1, psum, w, 1.0, None)
        run2 = lambda: k_new(x, o2, s2, psum, w, 1.0, None)
        run1()
        run2()
        torch.cuda.synchronize()
        bits = torch.equal(o1[real].view(torch.uint8), o2[real].view(torch.uint8)) and torch.equal(s1[real], s2[real])
        warm, cold, base = [], [], None
        for r in range(3):
            if r % 2:
                b, a = per_call(run2, rows), per_call(run1, rows)
            else:
                a, b = per_call(run1, rows), per_call(run2, rows)
            base = base or a
            warm.append(b[0] / a[0])
            cold.append(b[1] / a[1])
        # NaN probe: one real route's first gate value.
        first = int(torch.nonzero(real)[0])
        xn = x.clone()
        xn[first, 0] = float("nan")
        k_orig(xn, o1, s1, psum, w, 1.0, None)
        k_new(xn, o2, s2, psum, w, 1.0, None)
        torch.cuda.synchronize()
        nan1 = bool(o1[first].float().isnan().any())
        nan2 = bool(o2[first].float().isnan().any())
        print(f"{rows:5d} {routes:6d} {padded:6d}  {base[0]:9.2f} / {base[1]:9.2f}  "
              f"{statistics.median(warm):.3f} / {statistics.median(cold):.3f}  {'equal' if bits else 'DIFFER'}  {nan1} / {nan2}",
              flush=True)
    except Exception:
        print(f"{rows:5d} ERROR\n{traceback.format_exc(limit=4)}", flush=True)
