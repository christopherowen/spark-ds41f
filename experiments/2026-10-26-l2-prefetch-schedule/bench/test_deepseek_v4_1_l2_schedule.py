# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where each layer type issues its L2 prefetch windows (``l2_prefetch``), on fake
layers: the windows are cache hints, so what matters is the point and the plan."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4_1 import l2_prefetch
from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention
from vllm.models.deepseek_v4_1.common import engram as engram_module
from vllm.models.deepseek_v4_1.nvidia.model import DeepseekV4DecoderLayer
from vllm.models.deepseek_v4_1.tilelang import wo


@pytest.fixture
def issued(monkeypatch):
    """Every plan ``l2_prefetch.issue`` is given (None, a window the layer lacks, is
    not recorded)."""
    calls = []

    def issue(plan, rows):
        if plan is not None:
            calls.append((plan, rows))

    monkeypatch.setattr(l2_prefetch, "issue", issue)
    return calls


@pytest.mark.parametrize(
    "indexer, compressor, expected",
    [
        # window-only and reuse layers: WO behind NEXT's fill
        (None, None, {"q-a/kv": ["wo"], "q-b": [], "compressor": []}),
        # owner layers: IDX in WO's old place, WO behind it after the compressor
        (
            "indexer",
            "compressor",
            {"q-a/kv": [], "q-b": ["idx"], "compressor": ["wo"]},
        ),
        # indexer-only layers keep WO after Q-B
        ("indexer", None, {"q-a/kv": [], "q-b": ["wo"], "compressor": []}),
    ],
)
def test_wo_and_idx_points_by_layer_type(issued, indexer, compressor, expected):
    layer = SimpleNamespace(
        indexer=indexer,
        compressor=compressor,
        _l2pf_wo="wo",
        _l2pf_idx="idx" if compressor is not None else None,
    )
    for point, plans in expected.items():
        issued.clear()
        DeepseekV4Attention._issue_wo_prefetch(layer, 6, after=point)
        assert issued == [(plan, 6) for plan in plans], point


def test_ffn_window_between_wo_a_and_wo_b(monkeypatch):
    events = []

    class WOA:
        columns = 8

        def partial_specs(self, rows):
            return ()

        def __call__(self, x, xq, sf, partials):
            events.append("wo-a")

    monkeypatch.setattr(wo, "current_preallocated_workspace", lambda: None)
    monkeypatch.setattr(wo, "_activation_specs", lambda rows, k, decode_rows: ())
    monkeypatch.setattr(
        wo, "_scratch", lambda specs, scratch: [torch.empty(8), torch.empty(8)]
    )
    monkeypatch.setattr(
        wo, "_block32_gemm", lambda layer, xq, sf, out, rows: events.append("wo-b")
    )
    projection = wo.TileLangWOProjection.__new__(wo.TileLangWOProjection)
    projection.a, projection.b = WOA(), SimpleNamespace(tilelang_decode_rows=16)
    projection(
        torch.zeros(6, 2, 4), torch.empty(6, 8), after_a=lambda: events.append("ffn")
    )
    assert events == ["wo-a", "ffn", "wo-b"]
    events.clear()
    projection(torch.zeros(6, 2, 4), torch.empty(6, 8))
    assert events == ["wo-a", "wo-b"]


def test_next_window_between_engram_projection_and_all_gather(monkeypatch):
    events = []

    def all_gather(x):
        events.append("all-gather")
        return torch.cat([x, x], dim=-1)

    def local_product(x):
        events.append("gemm")
        return torch.ones(x.shape[0], 4)

    monkeypatch.setattr(engram_module, "tensor_model_parallel_all_gather", all_gather)
    cls = engram_module._PaddedColumnParallelLinear
    linear = cls.__new__(cls)
    torch.nn.Module.__init__(linear)
    linear.gather_output, linear.tp_size = True, 2
    linear.logical_output_size, linear.padded_output_size = 6, 8
    linear.local_product = local_product
    module = SimpleNamespace(
        wkv=linear,
        l2_prefetch_after_projection=lambda rows: events.append(("next", rows)),
    )
    kv = engram_module.Engram._project(module, torch.zeros(5, 3))
    assert events == ["gemm", ("next", 5), "all-gather"]
    assert kv.shape == (5, 6)

    # A replicated projection has no all-gather: the window follows the GEMM.
    events.clear()
    module.wkv = lambda x: (events.append("gemm"), torch.ones(x.shape[0], 6))[1]
    engram_module.Engram._project(module, torch.zeros(5, 3))
    assert events == ["gemm", ("next", 5)]


class _Plan(tuple):
    def describe(self):
        return ", ".join(self)


def _layer(*, owner=False, indexer=False, engram=False):
    """A decoder layer's attributes ``build_l2_prefetch`` reads; projections are their
    segment names."""
    index = None
    if owner or indexer:
        index = SimpleNamespace(
            wk="wk" if owner else None, wq_b="wq_b", weights_proj="hw"
        )
    attn = SimpleNamespace(
        _wo_projection_weights=SimpleNamespace(l2_segments=lambda: ["wo"]),
        compressor=SimpleNamespace(fused_wkv_wgate="wkv_wgate") if owner else None,
        indexer=index,
        fused_wqa_wkv="qa_kv",
        wq_b="q_b",
        _l2pf_idx=None,
    )
    return SimpleNamespace(
        attn=attn,
        hc_attn_fn=torch.empty(1),
        ffn_norm=SimpleNamespace(),
        attn_norm=SimpleNamespace(),
        ffn=SimpleNamespace(
            gate="gate",
            shared_experts=SimpleNamespace(gate_up_proj="gate_up", down_proj="down"),
            experts=SimpleNamespace(),
        ),
        engram=SimpleNamespace(wkv="engram_wkv", l2_prefetch_after_projection=None)
        if engram
        else None,
    )


@pytest.fixture
def segments(monkeypatch):
    monkeypatch.setattr(
        l2_prefetch,
        "param_segments",
        lambda name, owner, attrs: [f"{name}.{attr}" for attr in attrs],
    )
    monkeypatch.setattr(
        l2_prefetch,
        "linear_segments",
        lambda name, linear: [] if linear is None else [name],
    )
    monkeypatch.setattr(
        l2_prefetch,
        "make_plan",
        lambda segs, budget, device: _Plan(segs) if segs else None,
    )


def test_owner_layer_gets_an_idx_window_in_run_order(segments):
    layer, nxt = _layer(owner=True), _layer()
    DeepseekV4DecoderLayer.build_l2_prefetch(layer, nxt)
    assert layer.attn._l2pf_idx == (
        "compressor",
        "indexer.wk",
        "indexer.wq_b",
        "indexer.weights_proj",
    )
    plain = _layer(indexer=True)
    DeepseekV4DecoderLayer.build_l2_prefetch(plain, nxt)
    assert plain.attn._l2pf_idx is None


def test_engram_layer_moves_next_after_its_projection(segments, issued):
    layer, nxt = _layer(), _layer(engram=True)
    DeepseekV4DecoderLayer.build_l2_prefetch(layer, nxt)
    layer.ffn.experts._l2_prefetch_pre_reduce_hook(6)
    assert issued == [(("engram.wkv",), 6)]
    issued.clear()
    nxt.engram.l2_prefetch_after_projection(6)
    (plan, rows), *rest = issued
    assert not rest and rows == 6
    assert plan[:4] == (
        "mhc.hc_attn_fn_broadcast",
        "mhc.hc_attn_fn",
        "mhc.hc_attn_scale",
        "mhc.hc_attn_base",
    )
    assert plan[-2:] == ("wqa_wkv", "wq_b")

    # Any other next layer: NEXT on the MoE hook, no ENGRAM window.
    issued.clear()
    other = _layer()
    DeepseekV4DecoderLayer.build_l2_prefetch(other, _layer(indexer=True))
    other.ffn.experts._l2_prefetch_pre_reduce_hook(6)
    (plan, _), *rest = issued
    assert not rest and plan[-2:] == ("indexer.wq_b", "indexer.weights_proj")
