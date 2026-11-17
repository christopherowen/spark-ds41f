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


@pytest.mark.parametrize("rows", [1, 6, 32, LIMIT])
def test_rmsnorm_forward_with_cast_matches_separate_cast(rows):
    from vllm.models.deepseek_v4_1 import kernels
    from vllm.models.deepseek_v4_1.tilelang.norm import TileLangRMSNorm

    gen = torch.Generator(device=DEVICE).manual_seed(rows * 97 + 3)
    norm = TileLangRMSNorm(WIDTH, 1e-6).to(DEVICE)
    norm.weight.data.copy_(torch.rand(WIDTH, generator=gen, device=DEVICE) + 0.5)
    x = torch.randn(rows, WIDTH, generator=gen, device=DEVICE).to(torch.bfloat16)
    ref_out = torch.empty_like(x)
    norm(x, out=ref_out)
    ref_cast = kernels.cast_mxfp8(ref_out)

    fused_out = torch.empty_like(x)
    fused_cast = norm.forward_with_cast(x, out=fused_out)
    torch.accelerator.synchronize()

    assert fused_cast.rows.data_ptr() == fused_out.data_ptr()
    assert torch.equal(fused_out.view(torch.int16), ref_out.view(torch.int16))
    assert torch.equal(
        fused_cast.values[:rows].view(torch.uint8),
        ref_cast.values[:rows].view(torch.uint8),
    )
    assert torch.equal(
        fused_cast.scales[:rows].view(torch.uint8),
        ref_cast.scales[:rows].view(torch.uint8),
    )


def test_combine_and_precompute_context_kv_fuses_cast_into_main_norm(monkeypatch):
    from types import SimpleNamespace

    from vllm.models.deepseek_v4_1 import kernels
    from vllm.models.deepseek_v4_1.nvidia.dspark import DSparkDeepseekV4Model
    from vllm.models.deepseek_v4_1.tilelang.linear import provided_mxfp8
    from vllm.models.deepseek_v4_1.tilelang.norm import TileLangRMSNorm

    def forbid_cast(x):
        raise AssertionError("standalone cast_mxfp8 should not run after main_norm")

    monkeypatch.setattr(kernels, "cast_mxfp8", forbid_cast)
    seen_casts = []
    inserts = []

    class Proj:
        tilelang_mxfp8_input = True

        def __init__(self, idx):
            self.idx = idx

        def __call__(self, x):
            seen_casts.append(provided_mxfp8(x))
            return x[:, :512] + self.idx

    gen = torch.Generator(device=DEVICE).manual_seed(42)
    norm = TileLangRMSNorm(WIDTH, 1e-6).to(DEVICE)
    norm.weight.data.copy_(torch.rand(WIDTH, generator=gen, device=DEVICE) + 0.5)
    joined = dspark_context.NUM_SOURCES * WIDTH

    model = DSparkDeepseekV4Model.__new__(DSparkDeepseekV4Model)
    torch.nn.Module.__init__(model)
    model.hidden_size = WIDTH
    model.main_proj = lambda x: x[:, :WIDTH] * 2 + x[:, WIDTH : 2 * WIDTH]
    model.main_norm = norm
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

    aux = torch.randn(6, joined, generator=gen, device=DEVICE).to(torch.bfloat16)
    out = torch.empty(6, WIDTH, dtype=torch.bfloat16, device=DEVICE)
    main_x = model.combine_hidden_states(aux, out=out)
    expected_norm = norm(model.main_proj(aux))
    positions = torch.arange(6, dtype=torch.int64, device=DEVICE)
    slots = [
        torch.arange(6, dtype=torch.int64, device=DEVICE) + i * 10 for i in range(3)
    ]
    model.precompute_and_store_context_kv(main_x, positions, slots)
    torch.accelerator.synchronize()

    assert main_x.data_ptr() == out.data_ptr()
    assert torch.equal(main_x.view(torch.int16), expected_norm.view(torch.int16))
    assert len(seen_casts) == 3
    assert all(c is not None and c is seen_casts[0] for c in seen_casts)
    assert len(inserts) == 3
    for i, (layer_idx, kv_raw, pos, slot) in enumerate(inserts):
        assert layer_idx == i
        assert torch.equal(kv_raw, main_x[:, :512] + i)
        assert pos is positions and slot is slots[i]

def test_combine_aux_hidden_states_into_fuses_join_and_cast_mxfp8(monkeypatch):
    from types import SimpleNamespace

    from vllm.models.deepseek_v4_1 import kernels
    from vllm.models.deepseek_v4_1.nvidia.dspark import DSparkDeepseekV4Model
    from vllm.models.deepseek_v4_1.tilelang.linear import provided_mxfp8
    from vllm.models.deepseek_v4_1.tilelang.norm import TileLangRMSNorm

    gen = torch.Generator(device=DEVICE).manual_seed(42)
    norm = TileLangRMSNorm(WIDTH, 1e-6).to(DEVICE)
    norm.with_cast = True
    norm.weight.data.copy_(torch.rand(WIDTH, generator=gen, device=DEVICE) + 0.5)

    seen_provided = []

    class MainProj:
        tilelang_mxfp8_input = True

        def __call__(self, x):
            seen_provided.append(provided_mxfp8(x))
            return x[:, :WIDTH] * 0.5 + x[:, WIDTH : 2 * WIDTH] * 0.25

    class KvProj:
        tilelang_mxfp8_input = True

        def __call__(self, x):
            seen_provided.append(provided_mxfp8(x))
            return x[:, :512]

    model = DSparkDeepseekV4Model.__new__(DSparkDeepseekV4Model)
    torch.nn.Module.__init__(model)
    model.hidden_size = WIDTH
    model.main_proj = MainProj()
    model.main_norm = norm
    model._context_kv_projections = [KvProj()]
    model._context_main_cast = None
    model.layers = [
        SimpleNamespace(
            attn=SimpleNamespace(insert_raw_context_kv=lambda kv_raw, pos, slots: None)
        )
    ]

    casts = []
    monkeypatch.setattr(kernels, "cast_mxfp8", lambda x: casts.append(x))

    rows = 12
    aux_states = [
        torch.randn(rows, WIDTH, generator=gen, device=DEVICE, dtype=torch.bfloat16)
        for _ in range(3)
    ]
    out = torch.empty(rows, WIDTH, dtype=torch.bfloat16, device=DEVICE)
    model.combine_aux_hidden_states_into(aux_states, out)
    assert model._context_main_cast is not None
    assert model._context_main_cast.rows.data_ptr() == out.data_ptr()
    assert seen_provided[0].values.shape[1] == 3 * WIDTH

    positions = torch.arange(rows, dtype=torch.int64, device=DEVICE)
    slots = [torch.arange(rows, dtype=torch.int64, device=DEVICE)]
    model.precompute_and_store_context_kv(out, positions, slots)
    assert not casts
    assert seen_provided[1].values.shape[1] == WIDTH


def _recipe_ladder(rows):
    # The deployment recipe's capture ladder (scripts/recipe.py capture_sizes).
    ladder = [1, 2, 3, 4, 6, 8, *range(12, 33, 4), *range(40, rows + 1, 8)]
    return [*(size for size in ladder if size < rows), rows]


def _decode_config(max_num_seqs, capture_sizes, drafts=5, verification=5):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from vllm.config import (
        CompilationConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )

    compilation_config = CompilationConfig(
        cudagraph_mode="FULL_AND_PIECEWISE", cudagraph_capture_sizes=capture_sizes
    )
    compilation_config.max_cudagraph_capture_size = capture_sizes[-1]
    compilation_config.post_init_cudagraph_sizes()
    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.compilation_config = compilation_config
    vllm_config.scheduler_config = SchedulerConfig.default_factory(
        max_num_seqs=max_num_seqs
    )
    vllm_config.parallel_config = ParallelConfig()
    vllm_config.cache_config = SimpleNamespace(use_kda_recoverssm=False)
    vllm_config.num_speculative_tokens = drafts
    vllm_config.uniform_decode_query_len = 1 + drafts
    vllm_config.num_verification_tokens = verification
    speculative_config = MagicMock()
    speculative_config.uses_acceptance_length_adaptation.return_value = False
    speculative_config.uses_batch_size_dynamic_speculative_decoding.return_value = (
        False
    )
    vllm_config.speculative_config = speculative_config
    return vllm_config


def _no_graph_pool(monkeypatch):
    from types import SimpleNamespace

    from vllm.v1.worker.gpu import cudagraph_utils

    monkeypatch.setattr(
        cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        cudagraph_utils.current_platform, "get_global_graph_pool", lambda: object()
    )


def _target_manager(vllm_config, decode_query_len):
    """The target's graphs as the model runner plans them under adaptive
    verification: FULL variable-length decode graphs beside piecewise mixed ones."""
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

    manager = CudaGraphManager(
        vllm_config,
        torch.device("cpu"),
        CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=decode_query_len,
        varlen_decode=True,
    )
    manager._graphs_captured = True
    return manager


def _context_graphs(vllm_config, limit):
    from types import SimpleNamespace

    from vllm.models.deepseek_v4_1.nvidia.dspark import DSparkContextCudaGraphs

    model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=8, dspark_target_layer_ids=[0, 1, 2])
    )
    graphs = DSparkContextCudaGraphs(
        model,
        vllm_config,
        torch.zeros(limit + 4, 8, dtype=torch.bfloat16),
        torch.zeros(limit + 4, dtype=torch.int64),
        torch.zeros(1, limit + 4, dtype=torch.int64),
        None,
        limit,
    )
    graphs.manager._graphs_captured = True
    return graphs


@pytest.mark.parametrize(
    "max_num_seqs,rows,verification",
    [(8, 48, 5), (16, 96, 5), (8, 48, 8), (8, 59, 16), (16, 107, 16)],
    ids=["tp3-triangle", "tp4-ring", "copy-drafts", "copy-tp3", "copy-tp4"],
)
def test_context_capture_sizes_are_the_target_decode_graph_sizes(
    monkeypatch, max_num_seqs, rows, verification
):
    """The context graphs capture exactly the row counts of the target's FULL
    decode graphs below the largest decode step, derived from the same capture
    ladder and dense decode shapes: one source, so the two cannot drift."""
    from vllm.config.compilation import CUDAGraphMode
    from vllm.models.deepseek_v4_1.nvidia.dspark import _context_capture_sizes

    _no_graph_pool(monkeypatch)
    vllm_config = _decode_config(
        max_num_seqs, _recipe_ladder(rows), verification=verification
    )
    target = _target_manager(vllm_config, 1 + verification)
    target_sizes = {
        desc.num_tokens for desc in target._capture_descs[CUDAGraphMode.FULL]
    }
    for limit in (rows, rows - 7):
        sizes = _context_capture_sizes(vllm_config, limit)
        assert sizes == sorted({n for n in target_sizes if n <= limit} | {limit})
    assert _context_capture_sizes(vllm_config, 0) == []
    assert all(
        n * 6 in _context_capture_sizes(vllm_config, rows)
        for n in range(1, max_num_seqs + 1)
        if n * 6 <= rows
    )
    if (max_num_seqs, verification) == (8, 5):
        # TP3: 23 graphs, every one- and two-request total and every full-depth
        # step exact, where powers of two captured 7.
        assert _context_capture_sizes(vllm_config, rows) == [
            *range(1, 13), 16, 18, 20, 24, 28, 30, 32, 36, 40, 42, 48
        ]


@pytest.mark.parametrize(
    "max_num_seqs,rows", [(8, 48), (16, 96)], ids=["tp3-triangle", "tp4-ring"]
)
def test_context_graphs_pad_decode_rows_no_further_than_the_target(
    monkeypatch, max_num_seqs, rows
):
    """Every decode step the target runs as a FULL graph replays a context graph
    no wider than the target's: full-depth steps (6 rows a request) and every one-
    and two-request total (adaptive verification trims drafts) exactly."""
    from vllm.config.compilation import CUDAGraphMode

    _no_graph_pool(monkeypatch)
    vllm_config = _decode_config(max_num_seqs, _recipe_ladder(rows))
    target = _target_manager(vllm_config, 6)
    graphs = _context_graphs(vllm_config, rows)
    sizes = graphs.manager.compilation_config.cudagraph_capture_sizes
    assert graphs.manager.compilation_config.max_cudagraph_capture_size == rows
    assert graphs.aux.shape[0] == rows

    def context_rows(num_tokens):
        desc = graphs.manager.dispatch(1, num_tokens, None, 0)
        assert desc.cg_mode == CUDAGraphMode.FULL
        return desc.num_tokens

    for num_reqs in range(1, max_num_seqs + 1):
        assert context_rows(6 * num_reqs) == 6 * num_reqs
        for num_tokens in range(num_reqs, 6 * num_reqs + 1):
            desc = target.dispatch(num_reqs, num_tokens, None, 0, max_query_len=6)
            assert desc.cg_mode == CUDAGraphMode.FULL
            assert num_tokens <= context_rows(num_tokens) <= desc.num_tokens
            if num_reqs <= 2:
                assert context_rows(num_tokens) == num_tokens
    assert graphs.manager.dispatch(1, rows + 1, None, 0).cg_mode == (
        CUDAGraphMode.NONE
    )
    assert len(graphs.manager._capture_descs[CUDAGraphMode.FULL]) == len(sizes)


def test_context_graphs_warm_largest_first_and_profiling_tracks_nested_manager(
    monkeypatch,
):
    from types import SimpleNamespace

    from vllm.v1.worker.gpu.cudagraph_utils import _profiling_cudagraph_managers
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    _no_graph_pool(monkeypatch)
    vllm_config = _decode_config(8, _recipe_ladder(48))
    graphs = _context_graphs(vllm_config, 48)
    graphs.model._context_main_cast = "stale"
    events = []
    monkeypatch.setattr(graphs, "_forward", lambda cap: events.append(("forward", cap)))
    monkeypatch.setattr(
        graphs.manager,
        "capture",
        lambda *_a, **_kw: events.append(("capture", graphs.model._context_main_cast)),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: events.append(("sync", None)))
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: events.append(("empty", None)))

    graphs.capture()
    expected_capacities = list(
        reversed(graphs.manager.compilation_config.cudagraph_capture_sizes)
    )
    assert [cap for kind, cap in events if kind == "forward"] == expected_capacities
    assert events[-3:] == [("sync", None), ("empty", None), ("capture", None)]

    runner = SimpleNamespace(
        cudagraph_manager=graphs.manager,
        speculator=SimpleNamespace(_context_preparer=graphs),
    )
    assert _profiling_cudagraph_managers(runner) == [graphs.manager]

    closed = []
    spec = DFlashSpeculator.__new__(DFlashSpeculator)
    spec._context_preparer = SimpleNamespace(close=lambda: closed.append(True))
    spec.reset_attn()
    assert closed == [True]
    assert spec._context_preparer is None


def test_grouped_context_kv_projection_matches_per_layer_bits():
    from types import SimpleNamespace

    from torch import nn

    from vllm.models.deepseek_v4_1.nvidia.dspark import (
        DSparkDeepseekV4Model,
        _grouped_context_kv_projection,
    )
    from vllm.models.deepseek_v4_1.tilelang.linear import (
        TileLangBlock32Rows,
        cast_mxfp8,
        provide_mxfp8,
    )
    from vllm.v1.worker.workspace import use_preallocated_workspace

    gen = torch.Generator(device=DEVICE).manual_seed(42)
    projs = []
    for _ in range(3):
        layer = nn.Module()
        layer.weight = (torch.randn((512, WIDTH), generator=gen, device=DEVICE) * 2).to(
            torch.float8_e4m3fn
        )
        layer.weight_scale_inv = torch.randint(
            120, 125, (512 // 32, WIDTH // 32), generator=gen, device=DEVICE
        ).to(torch.uint8)
        proj = TileLangBlock32Rows(layer, 0)
        proj.tilelang_mxfp8_input = True
        proj.declare_kernels()
        projs.append(proj)

    assert _grouped_context_kv_projection(projs[:1]) is None
    diff = nn.Module()
    diff.weight = projs[0].weight[:256]
    diff.weight_scale_inv = projs[0].weight_scale_inv[:8]
    diff_proj = TileLangBlock32Rows(diff, 0)
    assert _grouped_context_kv_projection([projs[0], diff_proj]) is None

    grouped = _grouped_context_kv_projection(projs)
    assert grouped is not None
    assert grouped.tilelang_mxfp8_input is True
    assert not getattr(grouped, "tilelang_weight_packed", False)
    grouped.declare_kernels()

    recorded = []
    model = DSparkDeepseekV4Model.__new__(DSparkDeepseekV4Model)
    nn.Module.__init__(model)
    model._context_kv_projections = projs
    model._grouped_context_kv_projection = grouped
    model._context_main_cast = None
    object.__setattr__(
        model,
        "layers",
        [
            SimpleNamespace(
                attn=SimpleNamespace(
                    insert_raw_context_kv=lambda kv, pos, slot, idx=i: recorded.append(
                        (idx, kv, pos, slot)
                    )
                )
            )
            for i in range(3)
        ],
    )

    scratch = torch.empty(16 << 20, dtype=torch.uint8, device=DEVICE)
    with use_preallocated_workspace(scratch):
        for rows in (1, 6, 24, 48, 256):
            x = torch.randn((rows, WIDTH), generator=gen, device=DEVICE).bfloat16()
            mxfp8 = cast_mxfp8(x)
            with provide_mxfp8(mxfp8):
                expected = [p(x) for p in projs]
            recorded.clear()
            positions = torch.arange(rows, device=DEVICE, dtype=torch.int64)
            slots = torch.arange(rows, device=DEVICE, dtype=torch.int64)
            DSparkDeepseekV4Model.precompute_and_store_context_kv(
                model, x, positions, slots
            )
            assert len(recorded) == 3
            for idx, (rec_idx, got, rec_pos, rec_slot) in enumerate(recorded):
                assert rec_idx == idx
                assert rec_pos is positions
                assert rec_slot is slots
                assert torch.equal(got.view(torch.int16), expected[idx].view(torch.int16))
