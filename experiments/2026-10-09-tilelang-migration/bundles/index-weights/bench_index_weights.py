"""Port B8: the indexer's head weights (32 heads, 5120 -> 32 BF16) and DeepSeek's
(heads * head_dim) ** -0.5 = 1/64 scale. Today: the TileLang projection, then B12X's
scale_index_weights pass. The port: TileLangScaledLinearMethod, the scale folded into
the weight at load, one launch.

1. Bits: the folded projection equals the projection plus B12X's pass at every row
   count, exactly.
2. Time per step under CUDA graphs, warm and cold, at the decode capture sizes and
   prefill rows.

Exits 1 if any bit differs or the folded projection is slower at any shape.
"""
import sys

import torch
from torch import nn

sys.path.insert(0, "/b")
from kbench import CAPACITY, DECODE, DEVICE, PREFILL, per_call, slower, timing_line  # noqa: E402

from b12x.attention.compressed_sparse_mla import weight_scale  # noqa: E402
from b12x.preparation import PreparationSession, PreparedCall  # noqa: E402

from vllm.models.deepseek_v4_1.tilelang.linear import (  # noqa: E402
    TileLangLinearMethod,
    TileLangScaledLinearMethod,
)
from vllm.v1.worker.workspace import use_preallocated_workspace  # noqa: E402

HEADS, HIDDEN, SCALE = 32, 5120, (32 * 128) ** -0.5
failures = []
gen = torch.Generator(device=DEVICE).manual_seed(111)
weight = (torch.randn((HEADS, HIDDEN), generator=gen, device=DEVICE) * 0.02).bfloat16()
x = torch.randn((CAPACITY, HIDDEN), generator=gen, device=DEVICE).bfloat16()
plain, folded = nn.Module(), nn.Module()
plain.weight = nn.Parameter(weight.clone(), requires_grad=False)
folded.weight = nn.Parameter(weight.clone(), requires_grad=False)
plain_method, folded_method = TileLangLinearMethod(), TileLangScaledLinearMethod(SCALE)
plain_method.process_weights_after_loading(plain)
folded_method.process_weights_after_loading(folded)
scratch = torch.empty(16 << 20, dtype=torch.uint8, device=DEVICE)
plan = weight_scale.plan(weight_scale.Query(max_elements=CAPACITY * HEADS), device=DEVICE)
b_out = torch.empty((CAPACITY, HEADS), dtype=torch.bfloat16, device=DEVICE)
outs = {}


def prepare(state):
    source = torch.ones((1, HEADS), dtype=torch.bfloat16, device=DEVICE)
    out = torch.empty_like(source)
    return PreparedCall(run=lambda: state.run(source, out=out))


def today(rows):
    with use_preallocated_workspace(scratch):
        weights = plain_method.apply(plain, x[:rows])
    weight_scale.scale_index_weights(weights, out=b_out[:rows], plan=plan)


def port(rows):
    with use_preallocated_workspace(scratch):
        outs[rows] = folded_method.apply(folded, x[:rows])


with PreparationSession(device="cuda", autotune=False, compile_workers=2) as session:
    session.prepare((plan.request(name="index_weights", prepare_call=prepare),))
    differ = []
    for rows in DECODE + PREFILL:
        today(rows)
        port(rows)
        torch.cuda.synchronize()
        if not torch.equal(outs[rows], b_out[:rows]):
            differ.append(rows)
    if differ:
        failures.append(f"bits differ at {differ}")
    print(f"folded projection differs from projection + B12X scale at {differ or 'no'} sizes", flush=True)
    print("us per step, warm / cold (b12x = TileLang projection + B12X scale)", flush=True)
    for rows in DECODE + PREFILL:
        base = per_call(lambda: today(rows), rows)
        folded_time = per_call(lambda: port(rows), rows)
        print(timing_line(rows, base, folded_time), flush=True)
        if slower(base, folded_time):
            failures.append(f"rows {rows}: folded slower")

for failure in failures:
    print("FAIL", failure)
sys.exit(1 if failures else 0)
