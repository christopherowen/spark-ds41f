"""GPU regression for the two-request CED failure observed on the r6 TP3 ring.

Run inside an r6 image with patch 0045 mounted/applied, before loading weights:
    python3 -m pytest --noconftest -q test_two_request_routing.py

No checkpoint, user conversation, or images are needed. On unpatched r6 the
mask assertion fails before NaN logits are sent to the device-side assertion.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("tile_kernels")
if not torch.cuda.is_available():
    pytest.skip("requires CUDA and the recipe's TileKernels image", allow_module_level=True)

from vllm.models.deepseek_v4_1.ced import CEDState, gather_rows  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang import router as routing  # noqa: E402


@pytest.mark.parametrize("prefill_rows", [369, 1631])
@pytest.mark.parametrize("actual_decode_rows", [1, 4, 5, 6])
def test_two_request_upper_bound_tail(monkeypatch, prefill_rows, actual_decode_rows):
    """Keep real prefill rows; skip dead drafts and the CPU plan's spare rows."""
    scheduled_decode_rows = 6
    actual_rows = actual_decode_rows + prefill_rows
    state = CEDState(2, 4096, (256, 512, 1024, 2048, 4096), "cuda")
    starts = torch.tensor([0, actual_decode_rows, actual_rows], dtype=torch.int32, device="cuda")
    seq = torch.tensor([118000, 105000], dtype=torch.int32, device="cuda")
    state.stage(starts, seq, [scheduled_decode_rows, prefill_rows], [False, False], [0, 0])
    indices = state.get_indices()
    assert indices is not None
    valid_count = actual_decode_rows + 128
    expected_indices = (
        list(range(actual_decode_rows))
        + list(range(actual_rows - 128, actual_rows))
        + [-1] * (scheduled_decode_rows - actual_decode_rows)
    )
    assert indices.tolist() == expected_indices

    padding = torch.zeros(actual_rows, dtype=torch.bool, device="cuda")
    dead_drafts = [row for row in (2, 3, 4) if row < actual_decode_rows]
    padding[dead_drafts] = True
    compact = (indices >= 0) & ~gather_rows(padding, indices)
    expected = torch.zeros(134, dtype=torch.bool, device="cuda")
    expected[:valid_count] = True
    expected[dead_drafts] = False
    assert torch.equal(compact, expected)

    context = SimpleNamespace(
        is_padding=padding,
        attn_metadata={"layer": SimpleNamespace(decoder=SimpleNamespace(routing_mask=compact))},
    )
    monkeypatch.setattr(routing, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(routing, "get_forward_context", lambda: context)
    monkeypatch.setattr(routing.envs, "VLLM_MOE_SKIP_PADDING", True)

    # Exercise the installed router, including r6's original implementation.
    # A baseline fails here cleanly, without poisoning its CUDA context.
    kwargs = {"ced_decoder": True} if "ced_decoder" in inspect.signature(routing._routing_mask).parameters else {}
    actual = routing._routing_mask(134, **kwargs)
    assert torch.equal(actual, expected), "CED padding was sliced instead of gathered"
    # Equal row counts must not make an encoder use the decoder's layout.
    assert torch.equal(routing._routing_mask(134), (~padding)[:134])

    bias = torch.linspace(-0.2, 0.2, 384, device="cuda")
    router = routing.TileKernelsRouter(
        top_k=6, global_num_experts=384, e_score_correction_bias=bias,
        renormalize=True, routed_scaling_factor=1.5, scoring_func="sqrtsoftplus",
    )
    router._ds41_is_ced_decoder = True
    finite_logits = torch.linspace(-4, 4, 134 * 384, device="cuda").reshape(134, 384)
    reference_weights, reference_ids = router._compute_routing(None, finite_logits, None)
    logits = finite_logits.clone()
    logits[~expected] = float("nan")
    weights, ids = router._compute_routing(None, logits, None)
    torch.cuda.synchronize()
    assert (ids[~expected] == -1).all()
    assert (weights[~expected] == 0).all()
    assert torch.equal(ids[expected], reference_ids[expected])
    assert torch.equal(weights[expected].view(torch.int32), reference_weights[expected].view(torch.int32))
    assert torch.isfinite(weights).all()
