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


@pytest.mark.parametrize("capacity", [1, 6, 32, LIMIT])
def test_context_graph_norm_writes_the_context_buffer(monkeypatch, capacity):
    """The context graph's combine writes main_x straight into the speculator's
    context buffer (the norm's ``out``), which the context KV projections then read:
    no copy between the norm and their per-token casts, the buffer's bits the
    combine's own (TileKernels' norm of a stand-in projection of the joined
    sources), the rows past the capacity untouched."""
    from types import SimpleNamespace

    from vllm.models.deepseek_v4_1.nvidia.dspark import (
        DSparkContextCudaGraphs,
        DSparkDeepseekV4Model,
    )
    from vllm.models.deepseek_v4_1.tilelang.norm import TileLangRMSNorm

    gen = torch.Generator(device=DEVICE).manual_seed(capacity)
    norm = TileLangRMSNorm(WIDTH, 1e-6).to(DEVICE)
    norm.weight.data.copy_(torch.rand(WIDTH, generator=gen, device=DEVICE) + 0.5)
    joined = dspark_context.NUM_SOURCES * WIDTH

    def main_proj(x):  # a stand-in projection, elementwise, so deterministic
        return x[:, :WIDTH] * 2 + x[:, WIDTH : 2 * WIDTH]

    model = SimpleNamespace(main_proj=main_proj, main_norm=norm)
    aux = torch.randn(LIMIT, joined, generator=gen, device=DEVICE).to(torch.bfloat16)
    aux[0, :8] = float("nan")
    hidden_states = torch.randn(LIMIT + 4, WIDTH, generator=gen, device=DEVICE)
    hidden_states = hidden_states.to(torch.bfloat16)
    before = hidden_states.clone()
    expected = DSparkDeepseekV4Model.combine_hidden_states(model, aux[:capacity])
    projected = []

    def combine(x, out):
        return DSparkDeepseekV4Model.combine_hidden_states(model, x, out)

    def project(states, positions, slots):
        projected.append((states, slots))

    graphs = DSparkContextCudaGraphs.__new__(DSparkContextCudaGraphs)
    graphs.model = SimpleNamespace(
        combine_hidden_states=combine, precompute_and_store_context_kv=project
    )
    graphs.aux = aux
    graphs.hidden_states = hidden_states
    graphs.positions = torch.zeros(LIMIT, dtype=torch.int64, device=DEVICE)
    graphs.slot_mappings = torch.zeros(1, LIMIT, dtype=torch.int64, device=DEVICE)
    graphs.layer_group_idx = None

    def refuse(self, *args, **kwargs):
        raise AssertionError("the context graph copied a tensor")

    monkeypatch.setattr(torch.Tensor, "copy_", refuse)
    graphs._forward(capacity)
    monkeypatch.undo()
    torch.accelerator.synchronize()
    ((states, slots),) = projected
    assert states.data_ptr() == hidden_states.data_ptr()
    assert states.shape == (capacity, WIDTH)
    assert torch.equal(
        hidden_states[:capacity].view(torch.int16), expected.view(torch.int16)
    )
    assert torch.equal(
        hidden_states[capacity:].view(torch.int16), before[capacity:].view(torch.int16)
    )


def test_precompute_and_store_context_kv_casts_once_and_fuses_front(monkeypatch):
    from types import SimpleNamespace

    from vllm.models.deepseek_v4_1 import kernels
    from vllm.models.deepseek_v4_1.nvidia.dspark import DSparkDeepseekV4Model

    casts = []
    orig_cast = kernels.cast_mxfp8

    def record_cast(x):
        out = orig_cast(x)
        casts.append(out)
        return out

    monkeypatch.setattr(kernels, "cast_mxfp8", record_cast)
    inserts = []

    class Proj:
        tilelang_mxfp8_input = True

        def __init__(self, idx):
            self.idx = idx

        def __call__(self, x):
            return x[:, :512] + self.idx

    model = DSparkDeepseekV4Model.__new__(DSparkDeepseekV4Model)
    torch.nn.Module.__init__(model)
    model._context_kv_projections = [Proj(i) for i in range(3)]
    model.layers = [
        SimpleNamespace(
            attn=SimpleNamespace(
                insert_raw_context_kv=lambda kv_raw, pos, slots, i=i: inserts.append(
                    (i, kv_raw, pos, slots)
                )
            )
        )
        for i in range(3)
    ]
    main_x = torch.randn(6, WIDTH, dtype=torch.bfloat16, device=DEVICE)
    positions = torch.arange(6, dtype=torch.int64, device=DEVICE)
    slots = [torch.arange(6, dtype=torch.int64, device=DEVICE) + i * 10 for i in range(3)]
    model.precompute_and_store_context_kv(main_x, positions, slots)
    assert len(casts) == 1
    assert len(inserts) == 3
    for i, (layer_idx, kv_raw, pos, slot) in enumerate(inserts):
        assert layer_idx == i
        assert torch.equal(kv_raw, main_x[:, :512] + i)
        assert pos is positions and slot is slots[i]
