"""Indexer ReLU forms that keep a NaN, against r6k's kernel (``max(dot, 0)``, which
drops it):
- isnan: ``isnan(dot) ? dot : max(dot, 0)``, vLLM kernel-contracts 0006. w1 found it
  4% slower at an 8K prefill chunk: TileLang lowered the loop serially;
- ne: ``dot != dot ? dot : max(dot, 0)``, a comparison in place of the call;
- sel: ``dot > 0 ? dot : (dot != dot ? dot : 0)``, no max.

Each variant compiles after a header line, so a TileLang vectorization warning in the
output names the variant that caused it.
1. Bits: a variant must write r6k's bits on finite inputs (integers compared).
2. Time per call under CUDA graphs, warm and cold, in three rounds alternating which
   kernel runs first; the median of the rounds' ratios is reported.
The shared SwiGLU and Engram gate fixes are re-timed the same way, since w1 flagged
single measurements of 1-17 us calls there.

Exits 0 when at least one variant keeps r6k's bits everywhere and is within 1% of r6k's
time at every shape (warm and cold, median ratio); the verdict names it.
"""
import importlib.util
import statistics
import sys

import torch

sys.path.insert(0, "/b")
from kbench import CAPACITY, DECODE, DEVICE, PREFILL, per_call  # noqa: E402

import vllm.models.deepseek_v4_1.tilelang as package  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang import activation, engram  # noqa: E402

ROUNDS = 3
BUDGET = 1.01
VARIANTS = ("isnan", "ne", "sel")


class _Op:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def register_fake(self, fn):
        return fn


def load(path, name):
    spec = importlib.util.spec_from_file_location(f"{package.__name__}.{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    custom_op = torch.library.custom_op
    torch.library.custom_op = lambda *args, **kwargs: _Op
    try:
        spec.loader.exec_module(module)
    finally:
        torch.library.custom_op = custom_op
    return module


def same_bits(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def rounds(old, new, rows):
    """Median over ROUNDS of new/old (warm, cold), alternating which runs first."""
    warm, cold, times = [], [], []
    for r in range(ROUNDS):
        if r % 2:
            n, o = per_call(new, rows), per_call(old, rows)
        else:
            o, n = per_call(old, rows), per_call(new, rows)
        warm.append(n[0] / o[0])
        cold.append(n[1] / o[1])
        times.append((o, n))
    return statistics.median(warm), statistics.median(cold), times[0]


gen = torch.Generator(device=DEVICE).manual_seed(23)
results = {v: {"bits": True, "worst": (0.0, 0.0)} for v in VARIANTS}

print("indexer score (median ratio of three alternating rounds, warm / cold)", flush=True)
old_module = load("/b/r6k_indexer.py", "r6k_indexer")
modules = {v: load(f"/b/indexer_{v}.py", f"indexer_{v}") for v in VARIANTS}
PAGE = 64
HEADS = old_module.INDEX_HEADS
for rows, width in ((1, 65536), (16, 65536), (96, 65536), (1, 1 << 20), (16, 1 << 20), (512, 16384), (2048, 8192)):
    pages = width // PAGE
    nbytes = old_module.page_bytes(PAGE)
    q = torch.randint(0, 256, (rows, HEADS, 64), generator=gen, device=DEVICE).to(torch.uint8)
    q_scales = torch.randint(118, 130, (rows, HEADS, 4), generator=gen, device=DEVICE).to(torch.uint8)
    weights = (torch.randn((rows, HEADS), generator=gen, device=DEVICE) / 64).bfloat16()
    pool = torch.randint(0, 256, (pages, nbytes), generator=gen, device=DEVICE).to(torch.uint8)
    pool[:, PAGE * 64 : PAGE * 68] = torch.randint(118, 130, (pages, PAGE * 4), generator=gen, device=DEVICE).to(torch.uint8)
    table = torch.stack([torch.randperm(pages, generator=gen, device=DEVICE) for _ in range(rows)]).to(torch.int32)
    lengths = torch.full((rows,), width, dtype=torch.int32, device=DEVICE)
    active = torch.tensor([width], dtype=torch.int32, device=DEVICE)
    args = (q.view(torch.int8), q_scales.view(torch.uint32).reshape(rows, -1), weights, pool.view(torch.int8),
            pool.view(torch.uint32), table, lengths, active)
    s_old = torch.empty((rows, width), dtype=torch.bfloat16, device=DEVICE)
    old = old_module.indexer_score_v41(width, PAGE)
    old(*args, s_old)
    for variant in VARIANTS:
        print(f"-- compile {variant} for {rows} x {width}", flush=True)
        kernel = modules[variant].indexer_score_v41(width, PAGE)
        s_new = torch.empty_like(s_old)
        kernel(*args, s_new)
        torch.cuda.synchronize()
        bits = same_bits(s_old, s_new)
        results[variant]["bits"] &= bits
        warm, cold, (o, n) = rounds(lambda: old(*args, s_old), lambda k=kernel, s=s_new: k(*args, s), rows)
        worst = results[variant]["worst"]
        results[variant]["worst"] = (max(worst[0], warm), max(worst[1], cold))
        print(f"  {variant:6s} {width >> 10:5d}K rows {rows:5d}  r6k {o[0]:9.2f} / {o[1]:9.2f} us"
              f"  {variant} {n[0]:9.2f} / {n[1]:9.2f}  median ratio {warm:.3f} / {cold:.3f}"
              f"{'' if bits else '  BITS DIFFER'}", flush=True)
        del s_new
    del q, q_scales, weights, pool, table, s_old
    torch.cuda.empty_cache()

print("shared SwiGLU and Engram gate fixes, re-timed (median ratio of three alternating rounds)", flush=True)
old_act, old_engram = load("/b/r6k_activation.py", "r6k_activation"), load("/b/r6k_engram.py", "r6k_engram")
N, LIMIT = 576, 10.0
x = (torch.randn((CAPACITY, 2 * N), generator=gen, device=DEVICE) * 6).bfloat16()
y_old, y_new = (torch.empty((CAPACITY, N), dtype=torch.bfloat16, device=DEVICE) for _ in range(2))
k_old, k_new = old_act.swiglu_clamp_kernel(N, LIMIT), activation.swiglu_clamp_kernel(N, LIMIT)
HC, DIM, EPS = 4, 5120, 1e-6
xs = torch.randn((CAPACITY, HC * DIM), generator=gen, device=DEVICE).bfloat16()
kv = torch.randn((CAPACITY, (HC + 1) * DIM), generator=gen, device=DEVICE).bfloat16()
weight = torch.rand((HC * DIM,), generator=gen, device=DEVICE) + 0.5
image = torch.rand((CAPACITY,), generator=gen, device=DEVICE) < 0.25
g_old, g_new = torch.empty_like(xs), torch.empty_like(xs)
e_old, e_new = old_engram.engram_gate_kernel(HC, DIM, EPS, True), engram.engram_gate_kernel(HC, DIM, EPS, True)


def gate(kernel, out, rows):
    kernel(xs[:rows].view(rows, HC, DIM), kv[:rows].view(rows, HC + 1, DIM), weight.view(HC, DIM),
           image[:rows], out[:rows].view(rows, HC, DIM))


over = []
for rows in DECODE + PREFILL:
    for name, old, new in (("swiglu", lambda r=rows: k_old(x[:r], y_old[:r]), lambda r=rows: k_new(x[:r], y_new[:r])),
                           ("engram", lambda r=rows: gate(e_old, g_old, r), lambda r=rows: gate(e_new, g_new, r))):
        warm, cold, _ = rounds(old, new, rows)
        flag = "  OVER" if max(warm, cold) > 1.03 else ""
        if flag:
            over.append(f"{name} {rows}")
        print(f"  {name:6s} rows {rows:5d}  median ratio {warm:.3f} / {cold:.3f}{flag}", flush=True)

print("summary", flush=True)
good = []
for variant, r in results.items():
    ok = r["bits"] and max(r["worst"]) <= BUDGET
    good.append(variant) if ok else None
    print(f"  {variant:6s} bits {'equal' if r['bits'] else 'DIFFER'}; worst median ratio "
          f"{r['worst'][0]:.3f} / {r['worst'][1]:.3f}{'  OK' if ok else ''}", flush=True)
print(f"  SwiGLU / Engram over 3% after repeats: {over or 'none'}", flush=True)
print(f"verdict: {good[0] if good else 'no variant'} keeps r6k's bits and time", flush=True)
sys.exit(0 if good else 1)
