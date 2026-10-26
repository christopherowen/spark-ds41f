# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A modular MoE kernel's output outlives the call, so it never lives in the
workspace arena it was handed: the arena's views are reused by the next lease.

DeepSeek V4.1 at TP1 (no all-reduce to copy the output) returned the routed
experts' output from ``workspace[2]``; the next decoder layer's mHC entry leased its
scratch from the same bytes and overwrote that output while reading it (NaN gate
logits with three or more requests in a batch).

Since the experts write the caller's output, a lease prepared for it holds no output
buffer either: the arena keeps only the experts' scratch.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe import modular_kernel as mk


def _impl():
    impl = object.__new__(mk.FusedMoEKernelModularImpl)
    impl.fused_experts = SimpleNamespace(output_dtype=torch.bfloat16)

    def prepare(hidden, weights, ids, *rest):
        return hidden, None, None, ids, weights

    impl._prepare = prepare

    def fused_experts(*, output_alias, **kwargs):
        # The experts write the caller's output directly, as the aliasing path does.
        output_alias.fill_(1.0)
        return output_alias

    impl._fused_experts = fused_experts
    impl._finalize = lambda output, fused_out, *rest, **kwargs: output
    return impl


def test_output_is_not_a_workspace_view():
    rows, hidden = 6, 64
    states = torch.randn(rows, hidden, dtype=torch.bfloat16)
    size = rows * hidden * 2
    arena = torch.zeros(2 * size, dtype=torch.uint8)
    fused_out = arena[size:].view(torch.bfloat16).view(rows, hidden)
    workspace = (
        arena[:0].view(torch.bfloat16),
        arena[:size].view(torch.bfloat16),
        fused_out,
    )
    out = _impl().apply(
        states,
        torch.empty(4, 1),
        torch.empty(4, 1),
        torch.zeros(rows, 2, dtype=torch.int64),
        torch.ones(rows, 2),
        workspace=workspace,
    )
    assert out.shape == states.shape and bool((out == 1).all())
    low, high = arena.data_ptr(), arena.data_ptr() + arena.numel()
    assert not low <= out.data_ptr() < high
    # A later lease of the arena leaves the output as it was.
    arena.fill_(0xFF)
    assert bool((out == 1).all())


ROWS, HIDDEN, SCRATCH = 6, 64, 96


class _Experts:
    """Experts with no workspace13, SCRATCH elements of workspace2 and a
    (rows, hidden) BF16 output, as DeepSeek V4.1's TileLang experts lay them out."""

    output_dtype = torch.bfloat16
    a2_scale = None

    def workspace_dtype(self, dtype):
        return dtype

    def workspace_shapes(self, M, N, K, *rest):
        return (0,), (SCRATCH,), (M, K)

    def moe_problem_size(self, a1q, w1, w2, topk_ids):
        return w1.shape[0], a1q.shape[0], w1.shape[1], a1q.shape[1], topk_ids.shape[1]

    def apply(self, *, output, **kwargs):
        output.fill_(1.0)


def _lease(monkeypatch, output_shape):
    """A prepared routed and shared lease on a CPU arena: its routed views and the
    arena's size in bytes."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.v1.worker import workspace as ws

    impl = object.__new__(mk.FusedMoEKernelModularImpl)
    impl.fused_experts = _Experts()
    manager = ws.WorkspaceManager(torch.device("cpu"))
    monkeypatch.setattr(ws, "dbo_current_ubatch_id", lambda: 0)
    monkeypatch.setattr(mk, "current_workspace_manager", lambda: manager)
    monkeypatch.setattr(mk.current_platform, "is_cpu", lambda: False)
    monkeypatch.setattr(mk.current_platform, "is_rocm", lambda: False)
    views, shared = impl._allocate_buffers(
        torch.bfloat16,
        torch.bfloat16,
        torch.device("cpu"),
        ROWS,
        ROWS,
        128,
        HIDDEN,
        2,
        4,
        4,
        None,
        MoEActivation.SILU,
        512,
        output_shape=output_shape,
    )
    assert shared is not None and shared.numel() == 512
    return impl, views, manager.available_bytes()


def test_a_lease_for_the_callers_output_holds_no_output_buffer(monkeypatch):
    _, views, size = _lease(monkeypatch, (ROWS, HIDDEN))
    assert views[2] is None
    _, leased, leased_size = _lease(monkeypatch, None)
    assert leased[2].shape == (ROWS, HIDDEN)
    # The output buffer (already 256-byte aligned) is the whole difference.
    assert leased_size - size == ROWS * HIDDEN * 2


def test_an_output_the_experts_cannot_write_keeps_its_buffer(monkeypatch):
    # Another shape than the experts' output, or experts that write a buffer of
    # their own (ROCm without aiter), lease the output buffer as before.
    _, views, _ = _lease(monkeypatch, (ROWS, HIDDEN + 1))
    assert views[2] is not None and views[2].shape == (ROWS, HIDDEN)
    monkeypatch.setattr(
        mk.FusedMoEKernelModularImpl,
        "_experts_write_output_alias",
        staticmethod(lambda: False),
    )
    _, views, _ = _lease(monkeypatch, (ROWS, HIDDEN))
    assert views[2] is not None


def test_the_experts_write_the_callers_output_into_a_lease_without_one(monkeypatch):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    impl, views, _ = _lease(monkeypatch, (ROWS, HIDDEN))
    states = torch.randn(ROWS, HIDDEN, dtype=torch.bfloat16)
    kwargs = dict(
        in_dtype=torch.bfloat16,
        a1q=states,
        a1q_scale=None,
        w1=torch.empty(4, 128),
        w2=torch.empty(4, HIDDEN),
        topk_weights=torch.ones(ROWS, 2),
        topk_ids=torch.zeros(ROWS, 2, dtype=torch.int64),
        activation=MoEActivation.SILU,
        global_num_experts=4,
        local_num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
        expert_tokens_meta=None,
        workspace=views,
    )
    output = torch.zeros(ROWS, HIDDEN, dtype=torch.bfloat16)
    assert impl._fused_experts(**kwargs, output_alias=output) is output
    assert bool((output == 1).all())
    # Without the caller's output the lease has nowhere to write: refused.
    with pytest.raises(ValueError, match="holds no output buffer"):
        impl._fused_experts(**kwargs, output_alias=None)
