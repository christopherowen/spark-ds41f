"""Port B7: the Engram gate (4 streams x 5120), B12X's run_engram_mix against the
TileLang kernel in tilelang/engram.py, with TileKernels' engram_gate_fwd kernel (one
warp per token and stream) and other block shapes of ours for comparison. A quarter
of the tokens are image tokens (B12X takes their complement as its mask).

1. Error against FP64, no worse than B12X's (max and RMS); image tokens pass through.
2. Batch invariance: every row count gives the same rows of the 8192-row batch.
3. Time per call under CUDA graphs, warm and cold, at the decode capture sizes and
   prefill rows.

Exits 1 if a check fails or the serving kernel (256 threads x 4) is more than 3%
slower than B12X at any shape.
"""
import sys

import torch

sys.path.insert(0, "/b")
from kbench import (  # noqa: E402
    CAPACITY, DECODE, DEVICE, PREFILL, errors, no_worse, per_call, slower, timing_line,
)

from b12x.norm import hyperconnection  # noqa: E402
from b12x.norm.hyperconnection._impl import run_engram_mix_impl  # noqa: E402
from b12x.preparation import PreparationSession, PreparedCall  # noqa: E402
from tile_kernels.config import get_num_sms, get_pdl  # noqa: E402
from tile_kernels.engram.engram_gate_fwd_cuda import get_engram_gate_fwd_kernel_cuda  # noqa: E402

from vllm.models.deepseek_v4_1.tilelang.engram import CLAMP, engram_gate, engram_gate_kernel  # noqa: E402

HC, DIM, EPS = 4, 5120, 1e-6
failures = []

gen = torch.Generator(device=DEVICE).manual_seed(71)
x = torch.randn((CAPACITY, HC * DIM), generator=gen, device=DEVICE).bfloat16()
kv = torch.randn((CAPACITY, (HC + 1) * DIM), generator=gen, device=DEVICE).bfloat16()
weight = torch.rand((HC * DIM,), generator=gen, device=DEVICE) + 0.5
image = torch.rand((CAPACITY,), generator=gen, device=DEVICE) < 0.25
text = ~image
outs = {name: torch.empty_like(x) for name in ("b12x", "tilelang", "variant", "tilekernels")}


def tilelang(rows):
    engram_gate(x[:rows], kv[:rows], weight, image[:rows], outs["tilelang"][:rows], EPS, HC)


def variant(threads, vec):
    kernel = engram_gate_kernel(HC, DIM, EPS, True, threads, vec)

    def call(rows):
        kernel(x[:rows].view(rows, HC, DIM), kv[:rows].view(rows, HC + 1, DIM), weight.view(HC, DIM),
               image[:rows], outs["variant"][:rows].view(rows, HC, DIM))
    return call


tk_kernel = get_engram_gate_fwd_kernel_cuda(DIM, EPS, DIM**-0.5, get_num_sms(), CLAMP, HC, False, True, False,
                                            get_pdl())


def tilekernels(rows):
    tk_kernel(x[:rows].view(rows, HC, DIM), kv[:rows].view(rows, HC + 1, DIM), weight.view(HC, DIM), image[:rows],
              outs["tilekernels"][:rows].view(rows, HC, DIM), None, None, None, None)


# B12X as the Engram layer prepares it: masked plan at the capacity.
plan = hyperconnection.plan(
    hyperconnection.Caps(device=DEVICE, max_tokens=CAPACITY, hidden_size=DIM, streams=HC),
    invocation={"operation": "engram_mix", "eps": EPS, "token_mask": True})


def prepare(state):
    return PreparedCall(run=lambda: run_engram_mix_impl(x, kv, weight, eps=EPS, plan=state, out=outs["b12x"],
                                                        token_mask=text))


def b12x(rows):
    hyperconnection.run_engram_mix(x[:rows], kv[:rows], weight, eps=EPS, plan=plan, out=outs["b12x"][:rows],
                                   token_mask=text[:rows])


with PreparationSession(device="cuda", autotune=False, compile_workers=2) as session:
    session.prepare((plan.request(name="engram-mix/True", prepare_call=prepare),))

    # 1. Error against FP64 at the full batch.
    for call in (tilelang, b12x, tilekernels):
        call(CAPACITY)
    torch.cuda.synchronize()
    xs = x.double().view(CAPACITY, HC, DIM)
    kvs = kv.double().view(CAPACITY, HC + 1, DIM)
    k, v = kvs[:, :HC], kvs[:, HC:]
    dot = (xs * weight.double().view(HC, DIM) * k).sum(-1)
    dot = dot * torch.rsqrt(xs.square().mean(-1) + EPS) * torch.rsqrt(k.square().mean(-1) + EPS) * DIM**-0.5
    gate = torch.where(image[:, None], 0.0, torch.sigmoid(dot.sign() * dot.abs().clamp(min=CLAMP).sqrt()))
    ref = (xs + gate[..., None] * v).view(CAPACITY, HC * DIM)
    del xs, kvs, k, v
    result = {name: errors(outs[name], ref) for name in ("tilelang", "b12x", "tilekernels")}
    ok = no_worse(result["tilelang"], result["b12x"])
    print("error vs fp64 max/rms  " + "  ".join(f"{n} {e[0]:.3e}/{e[1]:.3e}" for n, e in result.items())
          + f"  {'ok' if ok else 'WORSE'}", flush=True)
    if not ok:
        failures.append("error")
    if not torch.equal(outs["tilelang"][image], x[image]):
        failures.append("image tokens changed")
    differ = (outs["tilelang"] != outs["tilekernels"]).sum().item()
    print(f"elements differing from TileKernels: {differ} of {x.numel()}; from B12X: "
          f"{(outs['tilelang'] != outs['b12x']).sum().item()}", flush=True)
    del ref
    full, b_full = outs["tilelang"].clone(), outs["b12x"].clone()

    # 2. Batch invariance.
    tl_varies, b_varies = [], []
    for rows in DECODE + PREFILL[:-1]:
        tilelang(rows)
        b12x(rows)
        torch.cuda.synchronize()
        if not torch.equal(outs["tilelang"][:rows], full[:rows]):
            tl_varies.append(rows)
        if not torch.equal(outs["b12x"][:rows], b_full[:rows]):
            b_varies.append(rows)
    if tl_varies:
        failures.append(f"TileLang differs from its full batch at {tl_varies}")
    print(f"TileLang differs from its full batch at {tl_varies or 'no'} sizes; B12X at {b_varies or 'no'}",
          flush=True)

    # 3. Time.
    shapes = [(t, v) for t, v in ((128, 8), (320, 8), (640, 8)) if DIM % (t * v) == 0]
    variants = {f"t{t}v{v}": variant(t, v) for t, v in shapes}
    print("us per call, warm / cold; then TileKernels and other block shapes", flush=True)
    for rows in DECODE + PREFILL:
        base = per_call(lambda: b12x(rows), rows)
        port = per_call(lambda: tilelang(rows), rows)
        line = timing_line(rows, base, port)
        for name, call in (("tilekernels", tilekernels), *variants.items()):
            w, c = per_call(lambda: call(rows), rows)
            line += f"  {name} {w:7.2f} / {c:7.2f}"
        print(line, flush=True)
        if slower(base, port):
            failures.append(f"rows {rows}: TileLang slower")

for failure in failures:
    print("FAIL", failure)
sys.exit(1 if failures else 0)
