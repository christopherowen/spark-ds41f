"""Port B1: the attention's WO projection at TP4 (16 heads x 512 in 2 groups, rank 1024,
hidden 5120), B12X's fused wo_projection.run_inv_rope (prepared per decode row count and
for prefill, as the attention prepares it) against TileLangWOProjection (inverse RoPE in
place, grouped WO-A, WO-B).

1. Error against FP64 (inverse rotation in FP64, unquantized activations, dequantized
   weights), no worse than B12X's (max and RMS).
2. Batch invariance: every row count gives the same rows of the 8192-row batch.
3. Time per call under CUDA graphs, warm and cold, at the decode capture sizes and
   prefill rows.

Exits 1 if a check fails or TileLang is more than 3% slower than B12X at any shape.
"""
import dataclasses
import sys

import torch
from torch import nn

sys.path.insert(0, "/b")
from kbench import (  # noqa: E402
    CAPACITY, DECODE, DEVICE, PREFILL, errors, no_worse, per_call, slower, timing_line,
)

from b12x.gemm import wo_projection  # noqa: E402
from b12x.preparation import PreparationSession, PreparedCall  # noqa: E402

from vllm.models.deepseek_v4_1.tilelang.wo import TileLangWOProjection  # noqa: E402
from vllm.v1.worker.workspace import use_preallocated_workspace  # noqa: E402

HEADS, HEAD_DIM, GROUPS, RANK, HIDDEN, ROPE, POSITIONS = 16, 512, 2, 1024, 5120, 64, 1 << 20
WIDTH = HEADS // GROUPS * HEAD_DIM
failures = []
gen = torch.Generator(device=DEVICE).manual_seed(101)


def block32(n, k):
    layer = nn.Module()
    layer.weight = (torch.randn((n, k), generator=gen, device=DEVICE) * 2).to(torch.float8_e4m3fn)
    exps = torch.randint(118, 123, (n // 32, k // 32), generator=gen, device=DEVICE).to(torch.uint8)
    layer.weight_scale_inv = exps.view(torch.float8_e8m0fnu)
    return layer


def dequant(layer):
    exps = layer.weight_scale_inv.view(torch.uint8).double()
    return layer.weight.double() * torch.exp2(exps - 127).repeat_interleave(32, 0).repeat_interleave(32, 1)


wo_a, wo_b = block32(GROUPS * RANK, WIDTH), block32(HIDDEN, GROUPS * RANK)
inv = 1.0 / (10000 ** (torch.arange(0, ROPE, 2, device=DEVICE).double() / ROPE))
angles = torch.arange(POSITIONS, device=DEVICE).double()[:, None] * inv
table = torch.cat((angles.cos(), angles.sin()), dim=-1).float()
del angles
o = torch.randn((CAPACITY, HEADS, HEAD_DIM), generator=gen, device=DEVICE).bfloat16()
positions = torch.randint(0, POSITIONS, (CAPACITY,), generator=gen, device=DEVICE)
work = o.clone()  # TileLang rotates in place

tilelang_wo = TileLangWOProjection(wo_a, wo_b, groups=GROUPS)
t_out = torch.empty((CAPACITY, HIDDEN), dtype=torch.bfloat16, device=DEVICE)
scratch = torch.empty(256 << 20, dtype=torch.uint8, device=DEVICE)  # serving binds a workspace


def tilelang(rows):
    with use_preallocated_workspace(scratch):
        tilelang_wo(work[:rows], positions[:rows], table, t_out[:rows])


weights = wo_projection.pack_weights(wo_a.weight, wo_a.weight_scale_inv, wo_b.weight, wo_b.weight_scale_inv,
                                     groups=GROUPS, group_width=WIDTH, rank=RANK, hidden=HIDDEN,
                                     block_size=(32, 32))
invocation = dict(operation="inv_rope", heads_per_group=HEADS // GROUPS, nope_dim=HEAD_DIM - ROPE, rope_dim=ROPE,
                  positions_dtype="int64", cos_sin_dtype="float32")
plans, scratches = {}, {}
for key in (*DECODE, "prefill"):
    rows = CAPACITY if key == "prefill" else key
    plans[key] = wo_projection.plan(
        wo_projection.Caps(device=DEVICE, max_tokens=rows, groups=GROUPS, group_width=WIDTH, rank=RANK,
                           hidden=HIDDEN),
        invocation=dict(invocation, dynamic_tokens=key == "prefill"))


def prepare(state):
    rows = state.query.max_tokens
    source = torch.empty((rows, HEADS, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
    pos = torch.arange(rows, dtype=torch.int64, device=DEVICE)
    scratch = tuple(torch.empty(s.shape, dtype=s.dtype, device=DEVICE) for s in state._scratch_state.scratch_specs())
    binding = state.bind_inv_rope(scratch=scratch, o=source, positions=pos, cos_sin_cache=table, weights=weights,
                                  heads_per_group=HEADS // GROUPS, nope_dim=HEAD_DIM - ROPE, rope_dim=ROPE)
    return PreparedCall(run=lambda: state.run_inv_rope(binding), produce=lambda: source.normal_(std=0.25))


b_out = torch.empty((CAPACITY, HIDDEN), dtype=torch.bfloat16, device=DEVICE)


def b12x(rows):
    key = rows if rows <= DECODE[-1] else "prefill"
    plan = plans[key]
    if key not in scratches:
        scratches[key] = tuple(torch.empty(s.shape, dtype=s.dtype, device=DEVICE) for s in plan.scratch_specs())
    binding = wo_projection.bind_inv_rope(plan, scratch=scratches[key], o=o[:rows], positions=positions[:rows],
                                          cos_sin_cache=table, weights=weights, heads_per_group=HEADS // GROUPS,
                                          nope_dim=HEAD_DIM - ROPE, rope_dim=ROPE)
    output = b_out[:rows].as_strided((rows, HIDDEN, 1), (HIDDEN, 1, rows * HIDDEN))
    binding = dataclasses.replace(binding, output=output)
    wo_projection.run_inv_rope(binding=binding, plan=plan, stream=torch.cuda.current_stream().cuda_stream)


with PreparationSession(device="cuda", autotune=False, compile_workers=2) as session:
    session.prepare(tuple(plan.request(name=f"wo.{key}", prepare_call=prepare) for key, plan in plans.items()))

    # 1. Error against FP64 (one projection of the original rows).
    b12x(CAPACITY)
    tilelang(CAPACITY)
    torch.cuda.synchronize()
    cos, sin = table.double()[positions].chunk(2, dim=-1)
    pairs = o[..., -ROPE:].double().unflatten(-1, (ROPE // 2, 2))
    first, second = pairs[..., 0], pairs[..., 1]
    cos, sin = cos[:, None], -sin[:, None]  # inverse
    x = o.double().clone()
    x[..., -ROPE:] = torch.stack((first * cos - second * sin, first * sin + second * cos), -1).flatten(-2)
    x = x.view(CAPACITY, -1)
    del pairs, first, second, cos, sin
    w_a = dequant(wo_a)
    a = torch.cat([x[:, g * WIDTH:(g + 1) * WIDTH] @ w_a[g * RANK:(g + 1) * RANK].T for g in range(GROUPS)], dim=1)
    del x, w_a
    ref = a @ dequant(wo_b).T
    del a
    port, base = errors(t_out, ref), errors(b_out, ref)
    ok = no_worse(port, base)
    print(f"error vs fp64 max/rms  tilelang {port[0]:.3e}/{port[1]:.3e}  b12x {base[0]:.3e}/{base[1]:.3e}"
          f"  {'ok' if ok else 'WORSE'}", flush=True)
    if not ok:
        failures.append("error")
    del ref
    full = t_out.clone()

    # 2. Batch invariance (fresh unrotated rows each time).
    varies = []
    for rows in DECODE + PREFILL[:-1]:
        work[:rows].copy_(o[:rows])
        tilelang(rows)
        torch.cuda.synchronize()
        if not torch.equal(t_out[:rows], full[:rows]):
            varies.append(rows)
    if varies:
        failures.append(f"TileLang differs from its full batch at {varies}")
    print(f"TileLang differs from its full batch at {varies or 'no'} sizes", flush=True)

    # 3. Time.
    print("us per call, warm / cold", flush=True)
    for rows in DECODE + PREFILL:
        base = per_call(lambda: b12x(rows), rows)
        port = per_call(lambda: tilelang(rows), rows)
        print(timing_line(rows, base, port), flush=True)
        if slower(base, port):
            failures.append(f"rows {rows}: TileLang slower")

for failure in failures:
    print("FAIL", failure)
sys.exit(1 if failures else 0)
