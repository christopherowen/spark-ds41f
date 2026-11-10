# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where each layer type issues its L2 prefetch windows (``l2_prefetch``), on fake
layers: the windows are cache hints, so what matters is the point and the plan."""

import os
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4_1 import l2_prefetch
from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention
from vllm.models.deepseek_v4_1.common import engram as engram_module
from vllm.models.deepseek_v4_1.nvidia.model import DeepseekV4DecoderLayer


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
        # window-only and reuse layers: WO after Q-B
        (None, None, {"q-b": ["wo"], "compressor": []}),
        # owner layers: COMP in WO's place, WO behind it after the compressor
        ("indexer", "compressor", {"q-b": ["comp"], "compressor": ["wo"]}),
        # indexer-only layers: WO after Q-B
        ("indexer", None, {"q-b": ["wo"], "compressor": []}),
    ],
)
def test_wo_and_idx_points_by_layer_type(issued, indexer, compressor, expected):
    layer = SimpleNamespace(
        indexer=indexer,
        compressor=compressor,
        _l2pf_wo="wo",
        _l2pf_comp="comp" if compressor is not None else None,
    )
    for point, plans in expected.items():
        issued.clear()
        DeepseekV4Attention._issue_wo_prefetch(layer, 6, after=point)
        assert issued == [(plan, 6) for plan in plans], point


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

    def hook(rows):
        events.append(("next", rows))

    kv = engram_module.project_rows(linear, torch.zeros(5, 3), hook)
    assert events == ["gemm", ("next", 5), "all-gather"]
    assert kv.shape == (5, 6)

    # A replicated projection has no all-gather: the window follows the GEMM.
    events.clear()

    def replicated(x):
        events.append("gemm")
        return torch.ones(x.shape[0], 6)

    engram_module.project_rows(replicated, torch.zeros(5, 3), hook)
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
        _l2pf_comp=None,
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


def test_owner_layer_gets_a_compressor_window(segments):
    layer, nxt = _layer(owner=True), _layer()
    DeepseekV4DecoderLayer.build_l2_prefetch(layer, nxt)
    assert layer.attn._l2pf_comp == ("compressor",)
    plain = _layer(indexer=True)
    DeepseekV4DecoderLayer.build_l2_prefetch(plain, nxt)
    assert plain.attn._l2pf_comp is None


def test_compressor_window_has_its_own_budget(segments, monkeypatch):
    budgets = {}

    def make_plan(segs, budget, device):
        if segs:
            budgets[segs[0]] = budget
        return _Plan(segs) if segs else None

    monkeypatch.setattr(l2_prefetch, "make_plan", make_plan)
    DeepseekV4DecoderLayer.build_l2_prefetch(_layer(owner=True), _layer())
    assert budgets["compressor"] == l2_prefetch.BUDGET_COMP
    assert budgets["wo"] == l2_prefetch.BUDGET_WO


def test_compressor_budget_fills_a_shard():
    """The default COMP budget takes a TP rank's whole 12-bit shard (TP5 and up),
    and only a shard's worth of the replicated BF16 weight (TP4 and below)."""
    from vllm.models.glm5next.nvidia import l2_prefetch as base

    if os.getenv("VLLM_DS41_L2_PREFETCH_COMP_MB") is not None:
        pytest.skip("VLLM_DS41_L2_PREFETCH_COMP_MB overrides the default")
    budget = l2_prefetch.BUDGET_COMP
    shard = 256 * 7680  # ratio two at TP5: 2 x 128 columns, 12-bit, H=5120
    taken, rest = base.take_budget([("compressor.weight", 0, shard)], budget)
    assert taken == [("compressor.weight", 0, shard)] and not rest
    replicated = 1024 * 5120 * 2  # ratio two through TP4: BF16, every column
    taken, rest = base.take_budget([("compressor.weight", 0, replicated)], budget)
    assert len(taken) == 1 and shard <= taken[0][2] <= 2_000_000
    assert rest[0][1] == taken[0][2]


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


def test_decode_shared_experts_launch_after_router_gate():
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExpertsOrder,
    )

    events = []
    experts = SimpleNamespace(fold_shared=True, combine_addend=None)

    class _Router:
        routes_hidden_states = True
        _post_gate_hook = None

        def select_experts(self, **kwargs):
            events.append("gate")
            if self._post_gate_hook is not None:
                hook = self._post_gate_hook
                self._post_gate_hook = None
                hook()
            events.append("topk")
            return torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.int64)

    class _Shared:
        output = "shared_out"

        def workspace_size(self, x):
            return 0

        def __call__(self, x, order, workspace=None):
            return None

        def _determine_shared_experts_order(self, x):
            return SharedExpertsOrder.MULTI_STREAM_OVERLAPPED

        def maybe_forward_async(self, x, workspace=None):
            events.append("shared_async")
            return True

        def wait_output(self):
            return "shared_out"

        def wait(self):
            events.append("shared_wait")

    runner = SimpleNamespace(
        _shared_experts=_Shared(),
        _maybe_apply_shared_experts=lambda *a, **k: None,
        _quant_method=SimpleNamespace(
            modular_experts=experts,
            topk_indices_dtype=torch.int64,
        ),
        routed_experts=SimpleNamespace(
            quant_method=SimpleNamespace(is_monolithic=False),
            forward_modular=lambda **kwargs: (events.append("routed"), "routed_out")[1],
        ),
        router=_Router(),
    )

    x6 = torch.zeros(6, 16)
    MoERunner._apply_quant_method(runner, x6, x6, x6)
    assert events == ["gate", "shared_async", "topk", "routed", "shared_wait"]

    events.clear()
    x48 = torch.zeros(48, 16)
    MoERunner._apply_quant_method(runner, x48, x48, x48)
    assert events == ["shared_async", "gate", "topk", "routed", "shared_wait"]
