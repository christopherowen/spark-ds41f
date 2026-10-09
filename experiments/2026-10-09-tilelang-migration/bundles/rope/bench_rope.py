"""Port B2: RoPE on the last 64 columns, B12X's rotary.rotate (out of place, copying the
leading columns) against the TileLang kernel in tilelang/rope.py (in place), at the TP4
roles: query (16 x 512), index query (32 x 128, replicated), KV (1 x 512), latent (1 x 512, ratio 4)
and index key (1 x 128, ratio 4), with the model's FP32 table.

1. Error against FP64, no worse than B12X's (max and RMS); leading columns unchanged.
2. Rows are independent: a prefix rotates to the same bits.
3. Time per call under CUDA graphs, warm and cold, at the decode capture sizes and
   prefill rows.

Exits 1 if a check fails or TileLang is more than 3% slower than B12X at any shape.
"""
import sys

import torch

sys.path.insert(0, "/b")
from kbench import (  # noqa: E402
    CAPACITY, DECODE, DEVICE, PREFILL, errors, no_worse, per_call, slower, timing_line,
)

from b12x.attention.compressed_sparse_mla import rotary  # noqa: E402
from b12x.preparation import PreparationSession, PreparedCall  # noqa: E402

from vllm.models.deepseek_v4_1.tilelang.rope import rope_  # noqa: E402

ROPE, POSITIONS = 64, 1 << 20
ROLES = {"q": (16, 512, 1), "index_query": (32, 128, 1), "kv": (1, 512, 1), "latent": (1, 512, 4),
         "index_key": (1, 128, 4)}
failures = []

inv = 1.0 / (10000 ** (torch.arange(0, ROPE, 2, device=DEVICE).double() / ROPE))
angles = torch.arange(POSITIONS, device=DEVICE).double()[:, None] * inv
table = torch.cat((angles.cos(), angles.sin()), dim=-1).float()
del angles
gen = torch.Generator(device=DEVICE).manual_seed(81)
positions = torch.randint(0, POSITIONS, (CAPACITY,), generator=gen, device=DEVICE)

with PreparationSession(device="cuda", autotune=False, compile_workers=2) as session:
    for role, (heads, dim, ratio) in ROLES.items():
        x = torch.randn((CAPACITY, heads, dim), generator=gen, device=DEVICE).bfloat16()
        work = x.clone()
        b_out = torch.empty_like(x)
        plan = rotary.plan(rotary.Query(max_rows=CAPACITY, heads=heads, dim=dim, ratio=ratio,
                                        cos_sin_dtype="float32"), device=DEVICE)

        def prepare(state, x=x, b_out=b_out):
            return PreparedCall(run=lambda: state.run(x[:1], positions[:1], table, out=b_out[:1]))

        session.prepare((plan.request(name=f"rope.{role}", prepare_call=prepare),))

        def b12x(rows, x=x, b_out=b_out, plan=plan):
            rotary.rotate(x[:rows], positions[:rows], table, out=b_out[:rows], plan=plan)

        def tilelang(rows, work=work, ratio=ratio):
            rope_(work[:rows], positions[:rows], table, ratio, False)

        # 1. Error against FP64 (one rotation of the original rows).
        b12x(CAPACITY)
        tilelang(CAPACITY)
        torch.cuda.synchronize()
        floored = positions // ratio * ratio
        cos, sin = table.double()[floored].chunk(2, dim=-1)
        pairs = x[..., -ROPE:].double().unflatten(-1, (ROPE // 2, 2))
        first, second = pairs[..., 0], pairs[..., 1]
        cos, sin = cos[:, None], sin[:, None]
        ref = torch.stack((first * cos - second * sin, first * sin + second * cos), -1).flatten(-2)
        port, base = errors(work[..., -ROPE:], ref), errors(b_out[..., -ROPE:], ref)
        ok = no_worse(port, base)
        same = torch.equal(work, b_out)
        print(f"{role} {heads}x{dim} ratio {ratio}: error vs fp64 max/rms  tilelang {port[0]:.3e}/{port[1]:.3e}"
              f"  b12x {base[0]:.3e}/{base[1]:.3e}  {'ok' if ok else 'WORSE'}; bits equal to B12X: {same}",
              flush=True)
        if not ok:
            failures.append(f"{role} error")
        if not torch.equal(work[..., :-ROPE], x[..., :-ROPE]):
            failures.append(f"{role}: leading columns changed")
        del ref, pairs, first, second, cos, sin

        # 2. Rows are independent.
        prefix = x[:7].clone()
        rope_(prefix, positions[:7], table, ratio, False)
        if not torch.equal(prefix, work[:7]):
            failures.append(f"{role}: a prefix differs")

        # 3. Time (repeated in-place rotation costs the same as the first).
        print(f"{role}: us per call, warm / cold", flush=True)
        for rows in DECODE + PREFILL:
            base = per_call(lambda: b12x(rows), rows)
            port = per_call(lambda: tilelang(rows), rows)
            print(timing_line(rows, base, port), flush=True)
            if slower(base, port):
                failures.append(f"{role} rows {rows}: TileLang slower")
        del x, work, b_out

for failure in failures:
    print("FAIL", failure)
sys.exit(1 if failures else 0)
