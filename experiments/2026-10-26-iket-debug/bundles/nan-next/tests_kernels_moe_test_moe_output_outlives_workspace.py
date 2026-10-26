# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A modular MoE kernel's output outlives the call, so it never lives in the
workspace arena it was handed: the arena's views are reused by the next lease.

DeepSeek V4.1 at TP1 (no all-reduce to copy the output) returned the routed
experts' output from ``workspace[2]``; the next decoder layer's mHC entry leased its
scratch from the same bytes and overwrote that output while reading it (NaN gate
logits with three or more requests in a batch).
"""

from types import SimpleNamespace

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
