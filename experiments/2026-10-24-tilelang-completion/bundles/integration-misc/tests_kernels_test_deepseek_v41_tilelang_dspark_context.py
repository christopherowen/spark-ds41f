# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark's context-graph input launch against the torch prologue it replaces
(``DSparkContextCudaGraphs.run``: three strided copies into ``aux``, then the
tail of ``aux``, ``positions`` and every group's slots scrubbed), bit for bit:
every buffer the launch may write is compared whole, starting from identical
garbage, with sources that are views of wider buffers (row strides past the
width) and longer than the step, NaN and infinite values copied as bits."""

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")

from vllm.models.deepseek_v4_1.tilelang import dspark_context  # noqa: E402
from vllm.v1.attention.backends.utils import PAD_SLOT_ID  # noqa: E402

DEVICE = torch.device("cuda")
WIDTH = 5120  # DeepSeek-V4.1 Flash's hidden size
LIMIT = 96  # 16 requests of 1 + 5 rows: the largest context graph
GROUPS = 2


def _reference(aux, sources, positions, slots, num_tokens, capacity):
    for i, source in enumerate(sources):
        aux[:num_tokens, i * WIDTH : (i + 1) * WIDTH].copy_(source[:num_tokens])
    aux[num_tokens:capacity].zero_()
    positions[num_tokens:capacity].zero_()
    slots[:, num_tokens:capacity].fill_(PAD_SLOT_ID)


@pytest.mark.parametrize(
    "num_tokens,capacity",
    [(1, 1), (5, 8), (6, 8), (31, 32), (64, 64), (90, LIMIT), (LIMIT, LIMIT)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_context_inputs(num_tokens, capacity, dtype):
    gen = torch.Generator(device=DEVICE).manual_seed(num_tokens * 131 + capacity)
    # Each source a view of a wider, longer buffer, as the target's outputs are.
    sources = []
    for i in range(dspark_context.NUM_SOURCES):
        wide = torch.randn(
            LIMIT + 3, WIDTH + 8 * (i + 1), generator=gen, device=DEVICE
        ).to(dtype)
        wide[0, :3] = torch.tensor([float("nan"), float("inf"), -float("inf")])
        sources.append(wide[:, 8 * i : 8 * i + WIDTH])
    garbage = torch.randint(
        -(2**15),
        2**15,
        (LIMIT, dspark_context.NUM_SOURCES * WIDTH),
        generator=gen,
        device=DEVICE,
        dtype=torch.int16,
    ).view(dtype)
    positions = torch.randint(0, 1 << 20, (LIMIT + 4,), generator=gen, device=DEVICE)
    slots = torch.randint(0, 1 << 20, (GROUPS, LIMIT + 4), generator=gen, device=DEVICE)
    results = []
    for kind in ("ref", "ours"):
        buffers = (garbage.clone(), positions.clone(), slots.clone())
        args = (buffers[0], sources, buffers[1], buffers[2], num_tokens, capacity)
        if kind == "ref":
            _reference(*args)
        else:
            dspark_context.context_inputs(*args)
        results.append(buffers)
    torch.accelerator.synchronize()
    for r, o in zip(*results):
        assert torch.equal(
            r.view(torch.int16) if r.dtype == dtype else r,
            o.view(torch.int16) if o.dtype == dtype else o,
        )


def test_context_inputs_rejects_misaligned_rows():
    aux = torch.zeros(
        8, dspark_context.NUM_SOURCES * WIDTH, device=DEVICE, dtype=torch.bfloat16
    )
    wide = torch.zeros(8, WIDTH + 1, device=DEVICE, dtype=torch.bfloat16)
    sources = [wide[:, 1:]] * dspark_context.NUM_SOURCES
    positions = torch.zeros(8, dtype=torch.int64, device=DEVICE)
    slots = torch.zeros(1, 8, dtype=torch.int64, device=DEVICE)
    with pytest.raises(ValueError, match="16-byte aligned"):
        dspark_context.context_inputs(aux, sources, positions, slots, 4, 8)
