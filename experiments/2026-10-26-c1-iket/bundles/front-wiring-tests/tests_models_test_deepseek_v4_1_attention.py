# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The TileLang attention layer's staging, binding and context KV writes."""

from types import SimpleNamespace

import pytest
import torch


def _tilelang_layer(layer_id):
    """A TileLang attention layer with the geometry its bind reads (no weights)."""
    from vllm.models.deepseek_v4_1.tilelang.attention import (
        DeepseekV41TileLangAttention,
    )

    module = DeepseekV41TileLangAttention.__new__(DeepseekV41TileLangAttention)
    torch.nn.Module.__init__(module)
    module.prefix = f"model.layers.{layer_id}.attn"
    module.layer_id, module.candidate_source_layer = layer_id, 3
    module.capacity, module.swa_width, module.compress_ratio = 300, 128, 1
    module._main_page = module._index_page = 64
    module._main_width = module._index_width = 16
    module.is_ced_decoder, module.is_index_source = False, True
    module.topk_indices_buffer = None
    module.indexer = SimpleNamespace(heads=32)
    module.compressor = None
    module.config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        speculative_config=None,
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=24),
    )
    module.rotary_emb = SimpleNamespace(cos_sin_cache=torch.zeros(16, 64))
    module._ready = False
    return module


def test_attention_offers_b12x_preparation_nothing():
    # B12X's preparation used to compile and launch helper plans for this layer
    # while memory profiling; the layer declares TileLang kernels only.
    module = _tilelang_layer(layer_id=3)
    assert getattr(module, "b12x_preparation_provider", None) is None
    assert not hasattr(module, "get_b12x_preparation_units")
    assert not module._ready


@pytest.mark.parametrize(
    "layer_id,staging",
    [
        (2, ["topk_indices_buffer", "_index_pages", "_active", "_short_index_pages"]),
        (
            3,
            [
                "topk_indices_buffer",
                "_index_pages",
                "_active",
                "_candidates",
                "_candidate_lens",
            ],
        ),
    ],
)
def test_tilelang_binding_prepares_the_attention_staging(
    monkeypatch, layer_id, staging
):
    # The layer's kernel binding allocates the staging, before memory profiling.
    module = _tilelang_layer(layer_id)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    module._build_tilelang_plans = lambda: "plans"
    module._attention_geometry()  # declare_kernels() runs it before binding
    with torch.inference_mode():  # the worker prepares kernels inside it
        module._bind_plans()
    assert module._ready and module._tilelang_plans == "plans"
    assert [name for name, _, _ in module._staging_specs] == staging
    for name, shape, dtype in module._staging_specs:
        tensor = getattr(module, name)
        assert tuple(tensor.shape) == shape and tensor.dtype == dtype
        assert not tensor.is_inference()  # updated in place while serving
    assert module._active.tolist() == [16 * 64]


def test_context_kv_write_uses_the_tilelang_writer(monkeypatch):
    from vllm.models.deepseek_v4_1.tilelang import cache_writer, rope

    module = _tilelang_layer(layer_id=3)
    writes = []
    monkeypatch.setattr(rope, "rope_", lambda *args: None)
    monkeypatch.setattr(
        cache_writer, "write_cache", lambda *args, **kwargs: writes.append(kwargs)
    )
    module.swa_cache_layer = SimpleNamespace(
        kv_cache=torch.empty(2, 8, dtype=torch.uint8), block_size=64
    )
    module._ready = True
    kv = torch.zeros(1, 512, dtype=torch.bfloat16)
    slots = torch.zeros(1, dtype=torch.int64)
    module.insert_context_kv(kv, slots, slots)
    assert writes == [dict(page_size=64, cache_kind="swa")]


def test_global_preparation_dependency_survives_functionalization(monkeypatch):
    from vllm.models.deepseek_v4_1 import attention

    device = "cuda" if torch.cuda.is_available() else "cpu"
    source = torch.arange(16, dtype=torch.float32, device=device).view(4, 4)
    positions = torch.arange(4, device=device, dtype=torch.int64)
    backing = torch.zeros((8, 4), device=device)

    class CacheConsumer(torch.nn.Module):
        prepare_global_kv = attention.DeepseekV4Attention.prepare_global_kv
        forward = attention.DeepseekV4Attention.forward

        def __init__(self):
            super().__init__()
            self.prefix = "functionalized_ced"
            self.compressor = SimpleNamespace(state_cache=None)
            self.kv_cache = backing[:4]
            self.indexer = SimpleNamespace(
                k_cache=SimpleNamespace(kv_cache=backing[4:])
            )

        def _prepare_global_kv(self, positions, hidden):
            self.kv_cache.copy_(hidden * 2)
            self.indexer.k_cache.kv_cache.copy_(hidden * 3)

        def _forward(self, positions, hidden):
            return self.kv_cache + self.indexer.k_cache.kv_cache

    layer = CacheConsumer()
    context = SimpleNamespace(no_compile_layers={layer.prefix: layer})
    monkeypatch.setattr(attention, "get_forward_context", lambda: context)

    def run(hidden):
        ready = layer.prepare_global_kv(positions, hidden)
        return layer(positions, hidden, global_kv_ready=ready)

    compiled = torch.compile(run, backend="aot_eager", fullgraph=True)
    for factor in (1, -2):
        hidden = source * factor
        output = compiled(hidden)
        torch.testing.assert_close(output, hidden * 5)
        # Preparation must remain visible to later consumers as well; mutable
        # alias-list functionalization used to copy stale clones back here.
        torch.testing.assert_close(backing, torch.cat((hidden * 2, hidden * 3)))


def test_decode_front_is_one_kernel_at_decode_rows(monkeypatch):
    """At decode rows with cache pages, the front after the fused projection is
    ``attention_front``'s one launch, at the SWA slots and page size, and Q-B gets its
    cast; sequence-parallel rows, rows past a decode tile and memory profiling (no
    cache pages) run the separate kernels."""
    from vllm.models.deepseek_v4_1 import attention
    from vllm.models.deepseek_v4_1.tilelang import front

    module = _tilelang_layer(layer_id=3)
    module.q_lora_rank = 1280
    module.q_norm = SimpleNamespace(weight=torch.ones(1280), variance_epsilon=1e-6)
    module.kv_norm = SimpleNamespace(weight=torch.ones(512), variance_epsilon=1e-6)
    module.swa_cache_layer = SimpleNamespace(
        prefix="swa", kv_cache=torch.empty(2, 8, dtype=torch.uint8), block_size=128
    )
    launches = []
    monkeypatch.setattr(front, "attention_front", lambda *args: launches.append(args))
    slots = torch.arange(100, dtype=torch.int64)
    metadata = {"swa": SimpleNamespace(slot_mapping=slots)}
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=metadata),
    )
    positions = torch.zeros(6, dtype=torch.int64)
    qr, kv, cast = module._front(torch.zeros(6, 1792), positions, None)
    (args,) = launches
    assert args[7].tolist() == list(range(6)) and args[8] == 128
    assert args[9] is qr and args[12] is kv
    assert cast.rows is qr and args[10] is cast.values and args[11] is cast.scales
    assert qr.shape == (6, 1280) and kv.shape == (6, 512)
    assert module._front(torch.zeros(6, 1792), positions, object()) is None
    assert module._front(torch.zeros(65, 1792), positions, None) is None
    metadata = None
    assert module._front(torch.zeros(6, 1792), positions, None) is None
    assert len(launches) == 1
