"""Fifth round (relu5): the index MXFP4 quantizer keeps a NaN, against r6k's quantizer.

r6k takes the group amax with ``fmaxf`` (drops a NaN) and codes ``|x|`` through a chain of
threshold tests (all false for NaN), so a NaN query or key value becomes a zero code under
a finite scale and never reaches the score. ``quant`` (indexer_quant.py):
- the group amax is an integer max on ``|x|``'s FP32 bits: for non-negative values integer
  order is float order with NaN above infinity, so it keeps the NaN; finite groups get
  fmaxf's value;
- the UE8M0 exponent is clamped at 255, so a NaN amax takes the NaN scale 0xFF (it would
  round up to 256 and wrap to 0x00), as an infinite amax already does;
- a non-finite group's codes are zeros, so its block's partial dot is zero and the 0xFF
  scale makes the dot NaN whether the MMA reads 0xFF as NaN or as 2^128.
Here, against r6k's quantizer:
- bytes on finite inputs (random, ties, zeros, signed zeros, BF16 extremes, subnormals),
  contiguous (queries) and paged (index-cache keys); they must be equal;
- time per call, median of three rounds that alternate which runs first, at query and key
  sizes; the owner's bar is no penalty, judged at 1%;
- report only: how the MMA reads a query group's scale 0xFF with nonzero codes and with
  zero codes, through r6k's score kernel.
"""
import importlib.util
import statistics
import sys
import traceback

import torch

sys.path.insert(0, "/b")
from kbench import DEVICE, per_call  # noqa: E402

import vllm.models.deepseek_v4_1.tilelang as package  # noqa: E402
from tests.kernels.attention.test_deepseek_v41_tilelang import (  # noqa: E402
    _dense_extent,
    _index_inputs,
    _mxfp4_inputs,
    _page_buffer,
)

ROUNDS, BUDGET = 3, 1.01
PAGE = 64


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


def rounds(old, new, n):
    warm, cold = [], []
    for r in range(ROUNDS):
        if r % 2:
            b, a = per_call(new, n), per_call(old, n)
        else:
            a, b = per_call(old, n), per_call(new, n)
        warm.append(b[0] / a[0])
        cold.append(b[1] / a[1])
    return statistics.median(warm), statistics.median(cold)


def finite_inputs(n, seed):
    x = _mxfp4_inputs(n, seed).float()
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    x[7] = 3.3895313892515355e38  # BF16 max
    x[8, ::2] = -3.3895313892515355e38
    x[9] = 2.0**-133 * torch.randint(1, 128, (128,), generator=gen, device=DEVICE)  # subnormals
    x[10] = 6.0 * 2.0**-126  # the amax floor
    x[11] = torch.randn(128, generator=gen, device=DEVICE) * 2.0**-120
    x[12] = torch.randn(128, generator=gen, device=DEVICE) * 2.0**100
    return x.bfloat16()


old = load("/b/r6k_indexer.py", "r6k_indexer")
new = load("/b/indexer_quant.py", "indexer_quant")
results = {}


def record(name, bits, warm, cold):
    r = results.setdefault(name, {"bits": True, "worst": (0.0, 0.0)})
    r["bits"] &= bits
    r["worst"] = (max(r["worst"][0], warm), max(r["worst"][1], cold))


print("contiguous quantizer, queries (vectors = tokens x 64 heads)", flush=True)
for n in (64, 1024, 6144, 131072, 524288):
    try:
        x = finite_inputs(n, 31)
        q0 = torch.empty((n, 64), dtype=torch.uint8, device=DEVICE)
        s0 = torch.empty((n, 4), dtype=torch.uint8, device=DEVICE)
        q1, s1 = torch.empty_like(q0), torch.empty_like(s0)
        old.quantize_mxfp4(x, q0, s0)
        new.quantize_mxfp4(x, q1, s1)
        torch.cuda.synchronize()
        bits = torch.equal(q0, q1) and torch.equal(s0, s1)
        warm, cold = rounds(lambda: old.quantize_mxfp4(x, q0, s0), lambda: new.quantize_mxfp4(x, q1, s1), n)
        record("quantize", bits, warm, cold)
        print(f"  {n:7d} vectors  median ratio {warm:.3f} / {cold:.3f}{'' if bits else '  BYTES DIFFER'}", flush=True)
    except Exception:
        results["quantize"] = {"bits": False, "worst": (9, 9), "error": traceback.format_exc(limit=3)}
        print(f"  {n}: ERROR\n{results['quantize']['error']}", flush=True)

print("paged quantizer, index-cache keys (tokens)", flush=True)
for n in (1, 16, 96, 2048, 8192):
    try:
        pages = (n + PAGE - 1) // PAGE + 1
        x = finite_inputs(max(n, 16), 37)[:n]
        gen = torch.Generator(device=DEVICE).manual_seed(n)
        slots = torch.randperm(pages * PAGE, generator=gen, device=DEVICE)[:n].to(torch.int64)
        pool0 = _page_buffer(pages, old.page_bytes(PAGE))
        pool1 = _page_buffer(pages, old.page_bytes(PAGE))
        old.write_index_keys(x, pool0, slots, PAGE)
        new.write_index_keys(x, pool1, slots, PAGE)
        torch.cuda.synchronize()
        bits = torch.equal(pool0._base, pool1._base)
        warm, cold = rounds(lambda: old.write_index_keys(x, pool0, slots, PAGE),
                            lambda: new.write_index_keys(x, pool1, slots, PAGE), n)
        record("write_index_keys", bits, warm, cold)
        print(f"  {n:7d} keys     median ratio {warm:.3f} / {cold:.3f}{'' if bits else '  BYTES DIFFER'}", flush=True)
    except Exception:
        results["write_index_keys"] = {"bits": False, "worst": (9, 9), "error": traceback.format_exc(limit=3)}
        print(f"  {n}: ERROR\n{results['write_index_keys']['error']}", flush=True)

print("report only: r6k's score with a query group's scale 0xFF", flush=True)
try:
    rows, context = 4, 4096
    names = ("q_packed", "q_scales", "weights", "pool", "page_table", "lengths", "active")
    for codes in ("nonzero", "zero"):
        inp = _index_inputs(rows, context, PAGE)
        width = context
        plan = old.IndexerV41Plan(width, PAGE)
        r, h = 1, 3
        inp["q_scales"][r, h, 0] = 0xFF
        if codes == "zero":
            inp["q_packed"][r, h, :16] = 0
        else:
            inp["q_packed"][r, h, :16] |= 0x11  # every code of the group nonzero
        _, scores = plan.run(*(inp[k] for k in names))
        e = _dense_extent(inp, width)[r]
        row = scores[r, :e].float()
        print(f"  {codes} codes: row {r} NaN {int(row.isnan().sum())}/{e}, +inf {int((row == float('inf')).sum())}, "
              f"finite {int(row.isfinite().sum())}", flush=True)
except Exception:
    print(f"  ERROR\n{traceback.format_exc(limit=3)}", flush=True)

print("summary (bytes on finite inputs; worst median ratio over sizes, warm / cold)", flush=True)
for name, r in results.items():
    ok = r["bits"] and not r.get("error") and max(r["worst"]) <= BUDGET
    r["ok"] = ok
    print(f"  {name:17s} {'ERROR' if r.get('error') else ('bytes equal' if r['bits'] else 'BYTES DIFFER')}; "
          f"worst {r['worst'][0]:.3f} / {r['worst'][1]:.3f}{'  OK' if ok else ''}", flush=True)
passed = len(results) == 2 and all(r["ok"] for r in results.values())
print(f"verdict: NaN-keeping quantizer {'within' if passed else 'NOT within'} {BUDGET - 1:.0%} of r6k with its bytes", flush=True)
sys.exit(0 if passed else 1)
