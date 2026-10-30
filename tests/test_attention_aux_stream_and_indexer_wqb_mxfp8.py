# SPDX-License-Identifier: Apache-2.0
"""Tests for indexer Q-B MXFP8 cast handover and auxiliary-stream overlap in
DeepSeek-V4.1 attention (``_index_query`` and ``_prepare_global_kv``)."""

from types import SimpleNamespace

import pytest
import torch


def _make_attention_shell(layer_id: int = 3):
    from vllm.models.deepseek_v4_1.tilelang.attention import (
        DeepseekV41TileLangAttention,
    )

    module = DeepseekV41TileLangAttention.__new__(DeepseekV41TileLangAttention)
    torch.nn.Module.__init__(module)
    module.prefix = f"model.layers.{layer_id}.attn"
    module.layer_id, module.candidate_source_layer = layer_id, 3
    module.capacity, module.swa_width, module.compress_ratio = 300, 128, 2
    module._main_page = module._index_page = 32
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
    module._ready = True
    return module


def test_index_query_mxfp8_handover_and_aux_stream_routing(monkeypatch):
    from vllm.models.deepseek_v4_1 import attention, kernels
    from vllm.models.deepseek_v4_1.tilelang.linear import MXFP8Rows, provided_mxfp8

    module = _make_attention_shell()
    seen_casts = []

    def fake_wq_b(x):
        seen_casts.append(provided_mxfp8(x))
        return torch.zeros(x.shape[0], 32 * 128, dtype=x.dtype, device=x.device)

    fake_wq_b.tilelang_mxfp8_input = True
    module.indexer = SimpleNamespace(
        heads=32,
        wq_b=fake_wq_b,
        weights_proj=lambda h: torch.ones(
            h.shape[0], 32, dtype=h.dtype, device=h.device
        ),
    )
    module._rope = lambda x, pos, role: x
    module._quantize_index_query = lambda mode, iq, data, scale: (
        data.fill_(5),
        scale.fill_(2),
    )

    parallel_streams = []
    monkeypatch.setattr(
        attention,
        "maybe_execute_in_parallel",
        lambda fn0, fn1, ev0, ev1, stream=None: (
            parallel_streams.append(stream),
            fn0(),
            fn1(),
        )[1:],
    )
    qr = torch.zeros(4, 1280)
    hidden = torch.zeros(4, 5120)
    positions = torch.arange(4, dtype=torch.int64)
    q_cast = MXFP8Rows(
        qr,
        torch.ones(64, 1280, dtype=torch.float8_e4m3fn),
        torch.ones(64, 160, dtype=torch.uint8),
    )
    iq_data, iq_scale, weights = module._index_query(
        qr,
        positions,
        hidden,
        None,
        0,
        4,
        SimpleNamespace(is_decode=True),
        q_cast=q_cast,
    )
    assert seen_casts == [q_cast]
    assert iq_data.shape == (4, 32, 64) and torch.all(iq_data == 5)
    assert iq_scale.shape == (4, 32, 4) and torch.all(iq_scale == 2)
    assert weights.shape == (4, 32) and torch.all(weights == 1)
    assert len(parallel_streams) == 1

    # Sequence-parallel row slice casts only [first:last].
    sliced = []
    monkeypatch.setattr(
        kernels,
        "cast_mxfp8",
        lambda x: (
            sliced.append(x.shape),
            MXFP8Rows(
                x,
                torch.ones(64, 1280, dtype=torch.float8_e4m3fn),
                torch.ones(64, 160, dtype=torch.uint8),
            ),
        )[1],
    )
    _, _, split_weights = module._index_query(
        qr,
        positions,
        hidden,
        None,
        1,
        3,
        SimpleNamespace(is_decode=False),
        q_cast=q_cast,
    )
    assert sliced == [torch.Size([2, 1280])]
    assert split_weights.shape == (2, 32)


def test_prepare_global_kv_parallel_ordering(monkeypatch):
    from vllm.models.deepseek_v4_1 import attention

    module = _make_attention_shell()
    module._owner = lambda: SimpleNamespace(prefix="main")
    module.kv_cache = torch.empty(2, 8, dtype=torch.uint8)
    events = []

    def compressor(hidden, main, state):
        events.append("compressor")
        return torch.zeros(4, 512), torch.arange(4, dtype=torch.int64)

    compressor.state_cache = None
    module.compressor = compressor
    module.indexer = SimpleNamespace(
        k_cache=SimpleNamespace(prefix="idx_k"),
        wk=lambda latent: events.append("wk") or torch.zeros(4, 128),
        k_norm=lambda key: events.append("k_norm") or key,
    )
    module._rope = lambda x, pos, role: events.append(f"rope:{role}") or x
    module._write_index_keys = lambda key, slots: events.append("write_index_keys")
    module._write_v41_cache = lambda latent, cache, slots, **kw: events.append(
        f"write_cache:{kw['cache_kind']}"
    )
    metadata = {
        "main": SimpleNamespace(),
        "idx_k": SimpleNamespace(slot_mapping=torch.arange(8, dtype=torch.int64)),
    }
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=metadata),
    )
    monkeypatch.setattr(
        attention,
        "maybe_execute_in_parallel",
        lambda fn0, fn1, ev0, ev1, stream=None: (fn0(), fn1()),
    )
    module._prepare_global_kv(torch.arange(4, dtype=torch.int64), torch.zeros(4, 5120))
    assert events == [
        "compressor",
        "wk",
        "k_norm",
        "rope:index_key",
        "write_index_keys",
        "rope:latent",
        "write_cache:indexed",
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [1, 4, 16, 48, 64, 96])
def test_gpu_index_query_and_prepare_global_kv_bit_identical(monkeypatch, rows):
    from vllm.models.deepseek_v4_1 import attention
    from vllm.models.deepseek_v4_1.tilelang import linear, prepare
    from vllm.models.deepseek_v4_1.tilelang.cache_writer import write_cache, writer_call
    from vllm.models.deepseek_v4_1.tilelang.indexer import (
        quantize_call,
        quantize_mxfp4,
        write_index_keys,
    )
    from vllm.models.deepseek_v4_1.tilelang.norm import TileLangRMSNorm
    from vllm.models.deepseek_v4_1.tilelang.rope import rope_, rope_call
    from vllm.v1.worker.workspace import use_preallocated_workspace

    monkeypatch.setattr(linear, "_max_rows", lambda: 96)
    device = "cuda"
    torch.manual_seed(42 + rows)

    def make_fp8_layer(n, k, mxfp8_in):
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(
            torch.randn(n, k, device=device).to(torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.weight_scale_inv = torch.nn.Parameter(
            torch.randint(
                120, 128, (n // 32, k // 32), dtype=torch.uint8, device=device
            ).view(torch.float8_e8m0fnu),
            requires_grad=False,
        )
        layer.tilelang_mxfp8_input = mxfp8_in
        method = linear.TileLangFP8LinearMethod(
            SimpleNamespace(weight_block_size=[32, 32])
        )
        method.process_weights_after_loading(layer)
        method.declare_layer_kernels(layer)
        layer.forward = lambda x: method.apply(layer, x)
        return layer

    def make_bf16_layer(n, k, scale=None):
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(
            torch.randn(n, k, dtype=torch.bfloat16, device=device) * 0.02,
            requires_grad=False,
        )
        method = (
            linear.TileLangScaledLinearMethod(scale)
            if scale is not None
            else linear.TileLangLinearMethod()
        )
        method.process_weights_after_loading(layer)
        method.declare_layer_kernels(layer)
        layer.forward = lambda x: method.apply(layer, x)
        return layer

    iwq_b_ref = make_fp8_layer(4096, 1280, mxfp8_in=False)
    iwq_b_opt = make_fp8_layer(4096, 1280, mxfp8_in=True)
    iwq_b_opt.weight.data.copy_(iwq_b_ref.weight.data)
    iwq_b_opt.tilelang_weight_sf.copy_(iwq_b_ref.tilelang_weight_sf)
    wproj = make_bf16_layer(32, 5120, scale=(32 * 128) ** -0.5)
    wk = make_bf16_layer(128, 512)
    k_norm = TileLangRMSNorm(128, 1e-6).to(device)
    k_norm.declare_kernels()

    prepare.declare(
        lambda: [
            rope_call(32, 128, 64, 1, False),
            rope_call(1, 128, 64, 2, False),
            rope_call(1, 512, 64, 2, False),
            quantize_call(),
            quantize_call(32),
            writer_call("indexed", 32, torch.int64),
        ],
        warm=lambda: linear._warm_casts(1280),
    )

    cos_sin = torch.randn(4096, 64, dtype=torch.float32, device=device)
    qr = torch.randn(rows, 1280, dtype=torch.bfloat16, device=device)
    hidden = torch.randn(rows, 5120, dtype=torch.bfloat16, device=device)
    positions = torch.arange(rows, dtype=torch.int64, device=device)
    q_cast = linear.cast_mxfp8(qr)
    scratch = torch.empty(4 * 1024 * 1024, dtype=torch.uint8, device=device)

    module = _make_attention_shell()
    module.rotary_emb = SimpleNamespace(cos_sin_cache=cos_sin)
    module._quantize_index_query = lambda mode, iq, data, scale: quantize_mxfp4(
        iq, data, scale
    )
    swa = SimpleNamespace(is_decode=(rows <= 64))

    # Reference sequential without q_cast handover or aux_stream
    module.indexer = SimpleNamespace(
        heads=32, wq_b=iwq_b_ref.forward, weights_proj=wproj.forward
    )
    with use_preallocated_workspace(scratch):
        ref_iq_d, ref_iq_s, ref_w = module._index_query(
            qr, positions, hidden, None, 0, rows, swa, q_cast=None
        )

    # Optimized with q_cast handover + aux_stream
    iwq_b_opt.forward.tilelang_mxfp8_input = True
    module.indexer = SimpleNamespace(
        heads=32, wq_b=iwq_b_opt.forward, weights_proj=wproj.forward
    )
    with use_preallocated_workspace(scratch):
        opt_iq_d, opt_iq_s, opt_w = module._index_query(
            qr, positions, hidden, None, 0, rows, swa, q_cast=q_cast
        )

    assert torch.equal(ref_iq_d, opt_iq_d)
    assert torch.equal(ref_iq_s, opt_iq_s)
    assert torch.equal(ref_w, opt_w)

    # Now test _prepare_global_kv bit-identity (eager and CUDA graph replay)
    latent_base = torch.randn(rows, 512, dtype=torch.bfloat16, device=device)
    slots = torch.arange(rows, dtype=torch.int64, device=device)
    idx_slots = torch.arange(rows, dtype=torch.int64, device=device)
    metadata = {
        "main": SimpleNamespace(),
        "idx_k": SimpleNamespace(slot_mapping=idx_slots),
    }
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=metadata),
    )
    module._owner = lambda: SimpleNamespace(prefix="main")

    def run_kv(use_aux: bool):
        latent_buf = latent_base.clone()
        k_cache = torch.zeros(16, 32 * 68, dtype=torch.uint8, device=device)
        kv_cache = torch.zeros(16, 32 * 288, dtype=torch.uint8, device=device)
        module.kv_cache = kv_cache
        comp = lambda h, m, s: (latent_buf, slots)
        comp.state_cache = None
        module.compressor = comp
        module.indexer = SimpleNamespace(
            k_cache=SimpleNamespace(prefix="idx_k", kv_cache=k_cache),
            wk=wk.forward,
            k_norm=k_norm,
        )
        module._write_index_keys = lambda key, sl: write_index_keys(
            key, k_cache, sl, page_size=32
        )
        module._write_v41_cache = lambda lat, cache, sl, **kw: write_cache(
            lat, cache, sl, **kw
        )
        orig_aux = attention.aux_stream
        if not use_aux:
            monkeypatch.setattr(attention, "aux_stream", lambda: None)
        try:
            with use_preallocated_workspace(scratch):
                module._prepare_global_kv(positions, hidden)
        finally:
            monkeypatch.setattr(attention, "aux_stream", orig_aux)
        return k_cache, kv_cache, latent_buf

    ref_kc, ref_kvc, ref_lat = run_kv(use_aux=False)
    opt_kc, opt_kvc, opt_lat = run_kv(use_aux=True)
    assert torch.equal(ref_kc, opt_kc)
    assert torch.equal(ref_kvc, opt_kvc)
    assert torch.equal(ref_lat, opt_lat)

    if rows <= 64:
        # Also verify inside a captured CUDA graph
        latent_cg = latent_base.clone()
        k_cache_cg = torch.zeros(16, 32 * 68, dtype=torch.uint8, device=device)
        kv_cache_cg = torch.zeros(16, 32 * 288, dtype=torch.uint8, device=device)
        module.kv_cache = kv_cache_cg
        comp_cg = lambda h, m, s: (latent_cg, slots)
        comp_cg.state_cache = None
        module.compressor = comp_cg
        module.indexer = SimpleNamespace(
            k_cache=SimpleNamespace(prefix="idx_k", kv_cache=k_cache_cg),
            wk=wk.forward,
            k_norm=k_norm,
            heads=32,
            wq_b=iwq_b_opt.forward,
            weights_proj=wproj.forward,
        )
        module._write_index_keys = lambda key, sl: write_index_keys(
            key, k_cache_cg, sl, page_size=32
        )
        module._write_v41_cache = lambda lat, cache, sl, **kw: write_cache(
            lat, cache, sl, **kw
        )

        # Warmup before capture
        with use_preallocated_workspace(scratch):
            module._prepare_global_kv(positions, hidden)
            module._index_query(
                qr, positions, hidden, None, 0, rows, swa, q_cast=q_cast
            )
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), use_preallocated_workspace(scratch):
            module._prepare_global_kv(positions, hidden)
            cg_iq_d, cg_iq_s, cg_w = module._index_query(
                qr, positions, hidden, None, 0, rows, swa, q_cast=q_cast
            )

        latent_cg.copy_(latent_base)
        k_cache_cg.zero_()
        kv_cache_cg.zero_()
        cg_iq_d.zero_()
        cg_iq_s.zero_()
        cg_w.zero_()
        g.replay()
        torch.cuda.synchronize()

        assert torch.equal(ref_kc, k_cache_cg)
        assert torch.equal(ref_kvc, kv_cache_cg)
        assert torch.equal(ref_lat, latent_cg)
        assert torch.equal(ref_iq_d, cg_iq_d)
        assert torch.equal(ref_iq_s, cg_iq_s)
        assert torch.equal(ref_w, cg_w)
