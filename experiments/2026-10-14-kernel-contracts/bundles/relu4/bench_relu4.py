"""Fourth round (relu4): NaN-keeping ReLUs from TileLang's existing language surface, against
r6k's ``max(dot, 0)``. Rounds 1-3 (bundles/relu, relu2, relu3) found:
- ``isnan(dot) ? dot : max(dot, 0)`` keeps the NaN, but TileLang lowers the per-head loop
  serially (``tirx.isnan`` is pure but not marked TVectorizable): about 3.5% slower at an 8K
  prefill chunk;
- the NaN test on the float's bits passes at up to 2.2%;
- ``dot != dot ? dot : max(dot, 0)`` vectorizes at r6k's speed, but its NaN test is gone even
  with fast math off (round 3), so TileLang or TVM removes it, not nvcc.
Here:
- ``clamp``: ``T.clamp(dot, 0, inf)``. TileLang documents its clamp as NaN-propagating, matching
  ``torch.clamp`` (upstream #3205, in r6k's TileLang); the reference's ``relu_`` is
  ``clamp(min=0)``. The op is pure and TVectorizable and lowers to ``tl::clamp``;
- ``lt``: ``dot < 0 ? 0 : dot``. No self-comparison; ``NaN < 0`` is false, so a NaN dot is its
  own result.
Also printed, to find what removes ``dot != dot`` before an upstream report:
- TVM's analyzer on ``x != x`` and the new forms (CPU only);
- the parsed TIR's ReLU lines and the generated CUDA of the score kernel for r6k, ne (round 1),
  clamp and lt at one shape, between markers.
Bits on finite inputs must equal r6k's; time is the median ratio of three rounds that alternate
which kernel runs first. Every variant is tried on its own: one that fails to build is reported,
not fatal.
"""
import importlib.util
import re
import statistics
import sys
import traceback

import torch

sys.path.insert(0, "/b")
from kbench import DEVICE, per_call  # noqa: E402

import vllm.models.deepseek_v4_1.tilelang as package  # noqa: E402

ROUNDS, BUDGET = 3, 1.01
TIMED = ("clamp", "lt")
DUMPED = ("r6k", "ne", "clamp", "lt")
DUMP_SHAPE = (16, 65536)


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


def analyzer_probe():
    """What TVM's analyzer makes of the NaN tests, on a Var and on the kernel's expression shape."""
    import tvm
    from tvm import arith, tirx

    print("TVM analyzer (CPU)", flush=True)
    buf = tirx.decl_buffer((4,), "float32", name="acc")
    zero = tirx.const(0.0, "float32")
    for label, x in (("var", tirx.Var("x", "float32")),
                     ("cast(cast(acc[0], bf16), f32)",
                      tirx.Cast("float32", tirx.Cast("bfloat16", tirx.BufferLoad(buf, [0]))))):
        ne = tirx.NE(x, x)
        analyzer = arith.Analyzer()
        cases = (
            ("x != x", ne),
            ("!(x != x)", tirx.Not(ne)),
            ("x == x", tirx.EQ(x, x)),
            ("if_then_else(x != x, x, max(x, 0))", tirx.if_then_else(ne, x, tirx.Max(x, zero))),
            ("if_then_else(x < 0, 0, x)", tirx.if_then_else(tirx.LT(x, zero), zero, x)),
        )
        print(f"  x = {label}", flush=True)
        for name, expr in cases:
            print(f"    simplify({name}) = {analyzer.simplify(expr)}", flush=True)
        print(f"    can_prove(!(x != x)) = {analyzer.can_prove(tirx.Not(ne))}", flush=True)
        print(f"    Python bool(x != x) = {bool(x != x)}  (TVM's deferred NotEqualOp)", flush=True)
    del tvm


try:
    analyzer_probe()
except Exception:
    print(f"analyzer probe ERROR\n{traceback.format_exc(limit=3)}", flush=True)

gen = torch.Generator(device=DEVICE).manual_seed(23)
results = {}


def record(name, bits, warm, cold):
    r = results.setdefault(name, {"bits": True, "worst": (0.0, 0.0), "error": None})
    r["bits"] &= bits
    r["worst"] = (max(r["worst"][0], warm), max(r["worst"][1], cold))


old_module = load("/b/r6k_indexer.py", "r6k_indexer")
modules = {"r6k": old_module}
for variant in DUMPED[1:]:
    try:
        modules[variant] = load(f"/b/indexer_{variant}.py", f"indexer_{variant}")
    except Exception:
        results[f"indexer {variant}"] = {"bits": False, "worst": (9, 9), "error": traceback.format_exc(limit=2)}

PAGE = 64
HEADS = old_module.INDEX_HEADS
RELU_LINE = re.compile(r"dot|clamp|isnan|if_then_else|T\.max\(")

print("indexer score (median ratio of three alternating rounds, warm / cold)", flush=True)
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
    if (rows, width) == DUMP_SHAPE:
        for variant in DUMPED:
            module = modules.get(variant)
            if module is None:
                continue
            try:
                tir = str(module.indexer_score_v41.get_tir(width, PAGE).script())
                print(f"==== parsed TIR ReLU lines: {variant}", flush=True)
                for line in tir.splitlines():
                    if RELU_LINE.search(line):
                        print(f"  {line.strip()}", flush=True)
                kernel = old if variant == "r6k" else module.indexer_score_v41(width, PAGE)
                print(f"==== BEGIN CUDA {variant} {rows} x {width}", flush=True)
                print(kernel.get_kernel_source(), flush=True)
                print(f"==== END CUDA {variant}", flush=True)
            except Exception:
                print(f"==== dump {variant} ERROR\n{traceback.format_exc(limit=3)}", flush=True)
    for variant in TIMED:
        module = modules.get(variant)
        name = f"indexer {variant}"
        if module is None or results.get(name, {}).get("error"):
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

print("summary (bits on finite inputs; worst median ratio over shapes, warm / cold)", flush=True)
for name, r in results.items():
    ok = r["bits"] and not r["error"] and max(r["worst"]) <= BUDGET
    r["ok"] = ok
    print(f"  {name:15s} {'ERROR' if r['error'] else ('bits equal' if r['bits'] else 'BITS DIFFER')}; "
          f"worst {r['worst'][0]:.3f} / {r['worst'][1]:.3f}{'  OK' if ok else ''}", flush=True)
best = [n for n, r in results.items() if r["ok"] and n.startswith("indexer")]
print(f"verdict: indexer forms within {BUDGET - 1:.0%} of r6k with its bits: {best or 'none'}", flush=True)
sys.exit(0 if best else 1)
