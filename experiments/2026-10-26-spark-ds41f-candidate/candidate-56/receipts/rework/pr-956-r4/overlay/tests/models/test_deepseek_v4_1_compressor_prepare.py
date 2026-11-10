# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek V4.1 compressor owes nothing to B12X preparation.

B12X prepares its units' plans while memory profiling, a distributed forward, so a
B12X unit would compile its CuTe kernel there on a rank whose cache is cold. The
compressor offers none; binding its declared TileLang kernels allocates the serving
storage before profiling instead.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4_1.compressor import DeepseekCompressor
from vllm.models.deepseek_v4_1.tilelang import prepare

TOKENS, REQUESTS = 64, 4


def _compressor(ratio: int) -> DeepseekCompressor:
    compressor = DeepseekCompressor.__new__(DeepseekCompressor)  # no weights or TP
    torch.nn.Module.__init__(compressor)
    compressor.prefix, compressor.compress_ratio = "layers.0.compressor", ratio
    compressor.capacity, compressor.requests = TOKENS, REQUESTS
    compressor.fused_wkv_wgate = SimpleNamespace(weight=torch.empty(512 * ratio, 32))
    compressor._values = compressor._latent = compressor._emitted = None
    compressor._slots = compressor._gates = compressor._pending_values = None
    compressor._pending_gates = compressor._pending_tags = None
    compressor._local_ids = compressor._row_ids = None
    return compressor


def _bind(monkeypatch, compressor: DeepseekCompressor) -> None:
    declared = {}
    monkeypatch.setattr(
        prepare,
        "declare",
        lambda calls=None, bind=None, warm=None: declared.update(bind=bind),
    )
    compressor.declare_kernels()
    declared["bind"]()


@pytest.mark.parametrize("ratio", [1, 2])
def test_offers_b12x_preparation_nothing(ratio):
    compressor = _compressor(ratio)
    assert getattr(compressor, "b12x_preparation_provider", None) is None
    assert not hasattr(compressor, "get_b12x_preparation_units")


def test_binding_allocates_the_serving_storage(monkeypatch):
    compressor = _compressor(2)
    _bind(monkeypatch, compressor)
    assert compressor._values.shape == (TOKENS, 512)
    assert compressor._values.dtype == torch.float32
    assert compressor._gates.shape == (TOKENS, 512)
    assert compressor._latent.shape == (TOKENS, 512)
    assert compressor._latent.dtype == torch.bfloat16
    assert compressor._emitted.shape == compressor._slots.shape == (TOKENS,)
    assert compressor._pending_values.shape == (REQUESTS, 512)
    assert compressor._pending_tags.shape == compressor._local_ids.shape == (REQUESTS,)
    assert compressor._row_ids.tolist() == list(range(REQUESTS))


def test_norm_weight_loads_unchanged_without_a_norm_module(monkeypatch):
    # The TileLang compression applies the norm's weight itself, so no norm module
    # (and no preparation unit) is built; the checkpoint loads the same parameter.
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear

    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=TOKENS, max_num_seqs=REQUESTS
        ),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(rms_norm_eps=1e-6)),
    )
    compressor = DeepseekCompressor(config, 1, 32, 512, prefix="layers.0.compressor")
    weight = dict(compressor.named_parameters())["norm.weight"]
    assert weight is compressor.norm.weight
    assert weight.shape == (512,) and weight.dtype == torch.float32
    assert not weight.requires_grad and not hasattr(weight, "weight_loader")
    assert type(compressor.norm) is torch.nn.Module
    assert not hasattr(compressor.norm, "b12x_preparation_provider")


def test_forwarding_before_binding_fails_by_name():
    compressor = _compressor(1)
    with pytest.raises(RuntimeError, match="bound"):
        compressor.forward(torch.empty(1, 32), metadata=None)


def test_sequential_compressor_layers_share_capacity_scratch(monkeypatch):
    """Multiple compressor layers of the same geometry and ratio share one set
    of transient serving scratch buffers instead of duplicating them per layer."""
    first, second = _compressor(2), _compressor(2)
    other_ratio = _compressor(1)
    _bind(monkeypatch, first)
    _bind(monkeypatch, second)
    _bind(monkeypatch, other_ratio)
    for attr in (
        "_values",
        "_latent",
        "_emitted",
        "_slots",
        "_row_ids",
        "_gates",
        "_pending_values",
        "_pending_gates",
        "_pending_tags",
        "_local_ids",
    ):
        assert getattr(first, attr) is getattr(second, attr)
    assert first._values is not other_ratio._values


@pytest.mark.parametrize("projection_tp", [1, 3, 4, 5])
def test_sp_projection_binding_preserves_decode_weight(monkeypatch, projection_tp):
    from vllm.models.deepseek_v4_1 import compressor as module
    from vllm.models.deepseek_v4_1.tilelang.linear import TileLangSplitLinearMethod

    compressor = _compressor(1)
    compressor._sp_enabled = True
    full_weight = torch.arange(512 * 32, dtype=torch.float32).reshape(512, 32)
    part_cols = module._compressor_part_cols(projection_tp)
    padded_weight = torch.zeros(projection_tp * part_cols, 32, dtype=torch.float32)
    padded_weight[:512].copy_(full_weight)
    weight = padded_weight[:part_cols].clone()
    compressor.fused_wkv_wgate = SimpleNamespace(
        weight=weight, tp_size=projection_tp, quant_method=TileLangSplitLinearMethod(1)
    )
    calls, gathered = {}, []
    monkeypatch.setattr(
        prepare,
        "declare",
        lambda declaration, **kw: calls.update(kernels=declaration(), bind=kw["bind"]),
    )
    kernel = object()
    monkeypatch.setattr(prepare, "launch", lambda call: kernel)

    def gather(value, dim):
        assert value.data_ptr() == weight.data_ptr() and dim == 0
        gathered.append(value.shape)
        return padded_weight

    monkeypatch.setattr(module, "tensor_model_parallel_all_gather", gather)
    compressor.declare_kernels()
    call = compressor._sp_projection_call()
    assert call in calls["kernels"]
    calls["bind"]()
    assert compressor.fused_wkv_wgate.weight is weight
    assert torch.equal(compressor._sp_weight, full_weight)
    assert compressor._sp_kernel is kernel
    assert len(gathered) == (projection_tp > 1)
    if projection_tp == 1:
        assert compressor._sp_weight.data_ptr() == weight.data_ptr()


@pytest.mark.parametrize("world_size", [3, 4])
@pytest.mark.parametrize("tokens", [5, 13, 31, 32, 33])
def test_sp_compressor_projects_only_local_rows_and_gathers_512_columns(
    monkeypatch, world_size, tokens
):
    from tests.models.test_deepseek_v4_1_sp_prefill import _World
    from vllm.models.deepseek_v4_1 import sp_prefill

    world = _World(world_size)
    monkeypatch.setattr(
        sp_prefill, "tensor_model_parallel_all_gather", world.all_gather
    )
    full_input = torch.arange(tokens * 32, dtype=torch.float32).reshape(tokens, 32)
    # Exact powers-of-two arithmetic makes this an orchestration oracle; native
    # BF16 projection equality is covered separately on GPU.
    expected = full_input[:, :1].expand(tokens, 512) * 0.5
    metadata = object()

    def run(rank):
        sp = sp_prefill.SPRows(tokens, world_size, rank)
        compressor = _compressor(1)
        compressor._values = torch.empty(tokens + world_size, 512)
        compressor._latent = torch.empty_like(compressor._values)
        compressor._slots = torch.empty(tokens + world_size, dtype=torch.int64)
        compressor._sp_weight = object()
        seen = []

        def project(inputs, weight, out):
            assert inputs.shape[0] == sp.local_rows and weight is compressor._sp_weight
            assert out.shape == (sp.local_rows, 512)
            out.copy_(inputs[:, :1].expand(sp.local_rows, 512) * 0.5)

        def normalize(rows, main, state, *, values, **kwargs):
            assert rows == tokens and main is metadata and state is None
            assert values.shape == (tokens, 512)
            assert torch.equal(values, expected)
            seen.append(values.shape)
            compressor._latent[:rows].copy_(values)
            compressor._slots[:rows].copy_(torch.arange(rows))

        compressor._sp_kernel = project
        compressor._forward_tilelang = normalize
        output, slots = compressor.forward(
            sp.local_slice(full_input), metadata, sp_rows=sp
        )
        assert seen == [(tokens, 512)]
        assert torch.equal(output, expected)
        assert torch.equal(slots, torch.arange(tokens))
        with pytest.raises(ValueError, match="local rows"):
            compressor.forward(full_input[:0], metadata, sp_rows=sp)
        return output.clone()

    assert len(world.run(run)) == world_size


def test_attention_global_kv_keeps_full_metadata_after_local_projection(monkeypatch):
    from vllm.models.deepseek_v4_1 import attention

    tokens, local_rows = 17, 6
    positions = torch.arange(tokens)
    values = torch.arange(tokens * 512, dtype=torch.float32).reshape(tokens, 512)
    main, index_meta = object(), SimpleNamespace(slot_mapping=torch.arange(tokens))
    compressor = _compressor(1)
    compressor.state_cache = None
    compressor._values = torch.empty(tokens, 512)
    compressor._latent = torch.empty(tokens, 512)
    compressor._slots = torch.arange(tokens)
    compressor._sp_weight = object()
    compressor._sp_kernel = lambda x, weight, out: out.zero_()
    seen = []

    def normalize(rows, metadata, state, *, values, **kwargs):
        assert rows == tokens and metadata is main and state is None
        compressor._latent.copy_(values)

    compressor._forward_tilelang = normalize

    def gather(local):
        assert local.shape == (local_rows, 512)
        seen.append("gather-projection")
        return values

    sp = SimpleNamespace(num_tokens=tokens, local_rows=local_rows, gather=gather)
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"global": main, "index": index_meta}),
    )

    def write_index(key, slots, *, positions):
        assert key.shape[0] == tokens and len(slots) == tokens
        assert len(positions) == tokens
        seen.append("index-full")

    def write_kv(latent, cache, slots, **kwargs):
        assert torch.equal(latent, values) and len(slots) == tokens
        assert len(kwargs["positions"]) == tokens
        seen.append("kv-full")

    target = SimpleNamespace(
        _ready=True,
        compressor=compressor,
        indexer=SimpleNamespace(
            wk=lambda x: x[:, :4], k_cache=SimpleNamespace(prefix="index")
        ),
        _owner=lambda: SimpleNamespace(prefix="global"),
        _main_page=64,
        kv_cache=torch.empty(0),
        _issue_wo_prefetch=lambda rows, **kw: None,
        _write_index_keys=write_index,
        _write_v41_cache=write_kv,
    )
    attention.DeepseekV4Attention._prepare_global_kv(
        target, positions, torch.ones(local_rows, 32), sp_rows=sp
    )
    assert seen == ["gather-projection", "index-full", "kv-full"]


@pytest.mark.parametrize(
    "world,tokens,replay,expected",
    [
        (3, 385, 128, False),
        (3, 390, 128, False),
        (3, 391, 128, True),
        (3, 6144, 128, True),
        (4, 512, 128, False),
        (4, 513, 128, True),
        (4, 16384, 128, True),
        (3, 3, 1, False),
        (3, 4, 1, True),
    ],
)
def test_ced_sp_dispatch_counts_replicated_projection_gather(
    world, tokens, replay, expected
):
    from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention
    from vllm.models.deepseek_v4_1.sp_prefill import SPRows

    compressor = _compressor(1)
    compressor.fused_wkv_wgate.tp_size = 1 if world == 3 else world
    attention = SimpleNamespace(compressor=compressor, hidden_size=5120)
    assert (
        DeepseekV4Attention.ced_sp_saves_bytes(
            attention, SPRows(tokens, world, 0), replay, 4
        )
        is expected
    )


def test_ced_sp_dispatch_keeps_unsupported_compressor_on_full_path():
    from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention
    from vllm.models.deepseek_v4_1.sp_prefill import SPRows

    for compressor in (None, _compressor(2), SimpleNamespace(compress_ratio=1)):
        assert not DeepseekV4Attention.ced_sp_saves_bytes(
            SimpleNamespace(compressor=compressor, hidden_size=5120),
            SPRows(6144, 3, 0),
            1,
            4,
        )


@pytest.mark.parametrize("world_size", [3, 4])
@pytest.mark.parametrize("tokens", [5, 13, 31, 32, 33])
def test_sp_ratio2_compressor_projects_local_rows_and_gathers_values_and_gates(
    monkeypatch, world_size, tokens
):
    from tests.models.test_deepseek_v4_1_sp_prefill import _World
    from vllm.models.deepseek_v4_1 import sp_prefill

    world = _World(world_size)
    monkeypatch.setattr(
        sp_prefill, "tensor_model_parallel_all_gather", world.all_gather
    )
    full_input = torch.arange(tokens * 32, dtype=torch.float32).reshape(tokens, 32)
    expected_values = full_input[:, :1].expand(tokens, 512) * 0.5
    expected_gates = full_input[:, 1:2].expand(tokens, 512) * 0.25
    metadata = object()
    state_meta = object()

    def run(rank):
        sp = sp_prefill.SPRows(tokens, world_size, rank)
        compressor = _compressor(2)
        compressor.fused_wkv_wgate.tp_size = world_size
        compressor.state_cache = SimpleNamespace(prefix="state")
        compressor._values = torch.empty(tokens + world_size, 512, dtype=torch.float32)
        compressor._gates = torch.empty(tokens + world_size, 512, dtype=torch.float32)
        compressor._latent = torch.empty(tokens + world_size, 512, dtype=torch.bfloat16)
        compressor._slots = torch.empty(tokens + world_size, dtype=torch.int64)
        compressor._sp_weight = object()
        seen = []

        def project_two(inputs, weight, out_values, out_gates):
            assert inputs.shape[0] == sp.local_rows and weight is compressor._sp_weight
            assert out_values.shape == (sp.local_rows, 512)
            assert out_gates.shape == (sp.local_rows, 512)
            out_values.copy_(inputs[:, :1].expand(sp.local_rows, 512) * 0.5)
            out_gates.copy_(inputs[:, 1:2].expand(sp.local_rows, 512) * 0.25)

        def normalize(rows, main, state, *, values, gates, **kwargs):
            assert rows == tokens and main is metadata and state is state_meta
            assert values.shape == (tokens, 512) and gates.shape == (tokens, 512)
            assert torch.equal(values, expected_values)
            assert torch.equal(gates, expected_gates)
            seen.append((values.shape, gates.shape))
            compressor._latent[:rows].copy_(values.to(torch.bfloat16))
            compressor._slots[:rows].copy_(torch.arange(rows))

        compressor._sp_kernel = project_two
        compressor._forward_tilelang = normalize
        output, slots = compressor.forward(
            sp.local_slice(full_input),
            metadata,
            state_metadata=state_meta,
            sp_rows=sp,
        )
        assert seen == [((tokens, 512), (tokens, 512))]
        assert torch.equal(output, expected_values.to(torch.bfloat16))
        assert torch.equal(slots, torch.arange(tokens))
        return output.clone()

    assert len(world.run(run)) == world_size
