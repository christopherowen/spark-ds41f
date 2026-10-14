"""Second round of NaN-keeping forms, against r6k's kernels. Round 1 (bundles/relu) found:
- isnan (vLLM 0006) keeps the NaN, but TileLang lowers the per-head loop serially:
  about 3.5% slower at an 8K prefill chunk;
- ``dot != dot`` and compare-and-select without max are vectorized and as fast as r6k,
  but fast math folds the NaN test away: the NaN contract fails (no NaN where the
  reference has 3980).
Here:
- indexer ``bits``: the NaN test on the float's bits (exponent all ones, mantissa nonzero),
  which fast math cannot fold;
- indexer ``fixup``: r6k's vectorized loop unchanged, then a separate loop that writes
  the NaN where the accumulator is NaN;
- SwiGLU ``expr``: the compare-and-select clamps as pure expressions, without the
  reassigned alloc_var scalars TileLang's warning names as a cause of serial loops.
Each kernel compiles after a header line, so TileLang's vectorization warnings in the
output name the kernel that caused them. Bits on finite inputs must equal r6k's; time is
the median ratio of three rounds that alternate which kernel runs first. Every variant
is tried on its own: a variant that fails to build is reported, not fatal.
"""
import importlib.util
import statistics
import sys
import traceback

import torch

sys.path.insert(0, "/b")
from kbench import CAPACITY, DECODE, DEVICE, PREFILL, per_call  # noqa: E402

import vllm.models.deepseek_v4_1.tilelang as package  # noqa: E402

ROUNDS, BUDGET = 3, 1.01


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
    warm, cold = [], []
    for r in range(ROUNDS):
        if r % 2:
            n, o = per_call(new, rows), per_call(old, rows)
        else:
            o, n = per_call(old, rows), per_call(new, rows)
        warm.append(n[0] / o[0])
        cold.append(n[1] / o[1])
    return statistics.median(warm), statistics.median(cold)


gen = torch.Generator(device=DEVICE).manual_seed(23)
results = {}


def record(name, bits, warm, cold):
    r = results.setdefault(name, {"bits": True, "worst": (0.0, 0.0), "error": None})
    r["bits"] &= bits
    r["worst"] = (max(r["worst"][0], warm), max(r["worst"][1], cold))


print("indexer score (median ratio of three alternating rounds, warm / cold)", flush=True)
old_module = load("/b/r6k_indexer.py", "r6k_indexer")
indexers = {}
for variant in ("isnan", "bits", "fixup"):
    try:
        indexers[variant] = load(f"/b/indexer_{variant}.py", f"indexer_{variant}")
    except Exception:
        results[f"indexer {variant}"] = {"bits": False, "worst": (9, 9), "error": traceback.format_exc(limit=2)}
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
    for variant, module in indexers.items():
        name = f"indexer {variant}"
        if results.get(name, {}).get("error"):
            continue
        try:
            print(f"-- compile {name} for {rows} x {width}", flush=True)
            kernel = module.indexer_score_v41(width, PAGE)
            s_new = torch.empty_like(s_old)
            kernel(*args, s_new)
            torch.cuda.synchronize()
            bits = same_bits(s_old, s_new)
            warm, cold = rounds(lambda: old(*args, s_old), lambda k=kernel, s=s_new: k(*args, s), rows)
            record(name, bits, warm, cold)
            print(f"  {variant:6s} {width >> 10:5d}K rows {rows:5d}  median ratio {warm:.3f} / {cold:.3f}"
                  f"{'' if bits else '  BITS DIFFER'}", flush=True)
        except Exception:
            results[name] = {"bits": False, "worst": (9, 9), "error": traceback.format_exc(limit=3)}
            print(f"  {variant}: ERROR\n{results[name]['error']}", flush=True)
    del q, q_scales, weights, pool, table, s_old
    torch.cuda.empty_cache()

print("shared SwiGLU (median ratio of three alternating rounds, warm / cold)", flush=True)
old_act = load("/b/r6k_activation.py", "r6k_activation")
N, LIMIT = 576, 10.0
x = (torch.randn((CAPACITY, 2 * N), generator=gen, device=DEVICE) * 6).bfloat16()
y_old = torch.empty((CAPACITY, N), dtype=torch.bfloat16, device=DEVICE)
print("-- compile swiglu r6k", flush=True)
k_old = old_act.swiglu_clamp_kernel(N, LIMIT)
for variant in ("fixed", "expr"):
    name = f"swiglu {variant}"
    try:
        print(f"-- compile {name}", flush=True)
        k_new = load(f"/b/activation_{variant}.py", f"activation_{variant}").swiglu_clamp_kernel(N, LIMIT)
        y_new = torch.empty_like(y_old)
        for rows in DECODE + PREFILL:
            k_old(x[:rows], y_old[:rows])
            k_new(x[:rows], y_new[:rows])
            torch.cuda.synchronize()
            bits = same_bits(y_old[:rows], y_new[:rows])
            warm, cold = rounds(lambda r=rows: k_old(x[:r], y_old[:r]), lambda r=rows: k_new(x[:r], y_new[:r]), rows)
            record(name, bits, warm, cold)
            print(f"  {variant:6s} rows {rows:5d}  median ratio {warm:.3f} / {cold:.3f}"
                  f"{'' if bits else '  BITS DIFFER'}", flush=True)
    except Exception:
        results[name] = {"bits": False, "worst": (9, 9), "error": traceback.format_exc(limit=3)}
        print(f"  {variant}: ERROR\n{results[name]['error']}", flush=True)

print("summary (bits on finite inputs; worst median ratio over shapes, warm / cold)", flush=True)
for name, r in results.items():
    ok = r["bits"] and not r["error"] and max(r["worst"]) <= BUDGET
    r["ok"] = ok
    print(f"  {name:15s} {'ERROR' if r['error'] else ('bits equal' if r['bits'] else 'BITS DIFFER')}; "
          f"worst {r['worst'][0]:.3f} / {r['worst'][1]:.3f}{'  OK' if ok else ''}", flush=True)
best = [n for n, r in results.items() if r["ok"] and n.startswith("indexer")]
print(f"verdict: indexer forms within {BUDGET - 1:.0%} of r6k with its bits: {best or 'none'}", flush=True)
sys.exit(0 if best else 1)
