import torch
import pytest
from torch import nn

from vllm.models.deepseek_v4_1.tilelang import vocab as vocab_mod
from vllm.models.deepseek_v4_1.tilelang.vocab import (
    pack_bf16_weight,
    prepare_packed_head,
    prepare_bf16_head,
    project_head,
    project_head_markov_greedy_pairs,
    project_head_screened_markov_greedy_pairs,
    screen_head_base,
)
from vllm.v1.worker.gpu.spec_decode.dspark.greedy import (
    greedy_markov_shard_pair,
    scratch_shape,
)

DEVICE = "cuda"


def _exact_full_vocab_top2_ref(logits: torch.Tensor, vocab_start: int = 0) -> torch.Tensor:
    """Exact full-vocabulary top-2 under vllm_vocab_first_argmax_keep order
    (NaN beats finite, smaller index breaks ties; index 2**30 if vocab < 2)."""
    rows, vocab = logits.shape
    out = torch.empty((rows, 4), dtype=torch.float32, device="cpu")
    lg = logits.float().cpu()
    for r in range(rows):
        row = lg[r]
        # Build exact total-order key: (not isnan, -val, idx)
        items = []
        for idx in range(vocab):
            v = float(row[idx].item())
            is_nan = v != v
            items.append((0 if is_nan else 1, 0.0 if is_nan else -v, idx, v))
        items.sort(key=lambda t: (t[0], t[1], t[2]))
        out[r, 0] = items[0][3]
        out[r, 1] = float(items[0][2] + vocab_start)
        if vocab >= 2:
            out[r, 2] = items[1][3]
            out[r, 3] = float(items[1][2] + vocab_start)
        else:
            out[r, 2] = float("-inf")
            out[r, 3] = float(1 << 30)
    return out.to(logits.device)


def _make_packed_layer(weight: torch.Tensor) -> nn.Module:
    layer = nn.Module()
    packed = pack_bf16_weight(weight.clone())
    layer.weight = nn.Parameter(packed.values, requires_grad=False)
    layer.weight_exception_rows = packed.exception_rows
    layer.weight_exception_entries = packed.exception_entries
    layer.weight_row_offsets = packed.row_offsets
    layer.packed_bf16_exponent_base = packed.exponent_base
    layer.packed_bf16_in_features = weight.shape[1]
    layer.tilelang_max_rows = 64
    layer.tilelang_markov_head = True
    return layer


def _make_bf16_layer(weight: torch.Tensor) -> nn.Module:
    layer = nn.Module()
    layer.weight = nn.Parameter(weight, requires_grad=False)
    layer.tilelang_max_rows = 64
    layer.tilelang_markov_head = True
    return layer


@pytest.mark.parametrize("packed", [True, False])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [1, 7, 16, 24, 48])
def test_exact_full_vocab_top2_all_cases(packed, out_dtype, rows):
    # Overlay check: scratch_shape(top2=True) must double columns
    assert scratch_shape(4, 8192, top2=True) == (4, 8)
    assert scratch_shape(4, 8192, top2=False) == (4, 4)

    n, k = 8192, 256
    vocab, vocab_start = 8192, 4096
    g = torch.Generator(device=DEVICE).manual_seed(1234)
    weight = (torch.randn(n, k, generator=g, device=DEVICE) * 0.05).bfloat16()
    layer = _make_packed_layer(weight) if packed else _make_bf16_layer(weight)
    if packed:
        prepare_packed_head(layer, out_dtype)
    else:
        prepare_bf16_head(layer, out_dtype)

    x = (torch.randn(rows, k, generator=g, device=DEVICE) * 0.25).bfloat16()
    base = (torch.randn(rows, n, generator=g, device=DEVICE) * 0.5).to(out_dtype)

    # Inject critical edge cases in leading rows (zero x so base decides exactly)
    x[: min(rows, 7)] = 0
    # Row 0: same 32/64-col tile! (40: 6.0, 41: 5.5, 6145: 5.0) -> runner-up MUST be (5.5, 41)
    base[0] = -1.0
    base[0, 40], base[0, 41], base[0, 6145] = 6.0, 5.5, 5.0
    if rows > 1:
        # Row 1: same 2048-partition, DIFFERENT tile! (40: 6.0, 500: 5.5, 6145: 5.0) -> (5.5, 500)
        base[1] = -1.0
        base[1, 40], base[1, 500], base[1, 6145] = 6.0, 5.5, 5.0
    if rows > 2:
        # Row 2: different partitions! (40: 6.0, 6145: 5.75) -> (5.75, 6145)
        base[2] = -1.0
        base[2, 40], base[2, 6145] = 6.0, 5.75
    if rows > 3:
        # Row 3: exact tie in SAME tile! (7: 5.0, 13: 5.0, 2053: 5.0) -> best=(5.0, 7), second=(5.0, 13)
        base[3] = -1.0
        base[3, 7], base[3, 13], base[3, 2053] = 5.0, 5.0, 5.0
    if rows > 4:
        # Row 4: exact tie in SAME 2048-partition, different tiles! (7: 5.0, 519: 5.0, 4105: 5.0) -> (5.0, 7), (5.0, 519)
        base[4] = -1.0
        base[4, 7], base[4, 519], base[4, 4105] = 5.0, 5.0, 5.0
    if rows > 5:
        # Row 5: NaN tie (900: NaN, 905: NaN, 6000: 9.0) -> (NaN, 900), (NaN, 905)
        base[5] = -1.0
        base[5, 900], base[5, 905], base[5, 6000] = float("nan"), float("nan"), 9.0
    if rows > 6:
        # Row 6: all -inf -> (-inf, 0), (-inf, 1)
        base[6] = float("-inf")

    bias = project_head(layer, x)
    full_logits = (base.to(out_dtype) + bias.to(out_dtype)).to(out_dtype)
    expected = _exact_full_vocab_top2_ref(full_logits[:, :vocab], vocab_start=vocab_start)

    # 1. Fused dense vocab.py path
    actual_vocab = project_head_markov_greedy_pairs(
        layer, x, base, vocab=vocab, vocab_start=vocab_start, top2=True
    )
    # 2. Unfused tilelang_sampling.py path
    actual_samp = torch.empty((rows, 4), dtype=torch.float32, device=DEVICE)
    s_shape = scratch_shape(rows, vocab, top2=True)
    greedy_markov_shard_pair(
        base[:, :vocab],
        bias[:, :vocab],
        vocab_start,
        actual_samp,
        torch.empty(s_shape, dtype=torch.float32, device=DEVICE),
        torch.empty(s_shape, dtype=torch.int32, device=DEVICE),
        top2=True,
    )

    assert torch.equal(actual_vocab.view(torch.int32), expected.view(torch.int32)), (
        f"vocab mismatch:\nactual={actual_vocab[:7]}\nexpected={expected[:7]}"
    )
    assert torch.equal(actual_samp.view(torch.int32), expected.view(torch.int32)), (
        f"sampling mismatch:\nactual={actual_samp[:7]}\nexpected={expected[:7]}"
    )

    # Also verify top2=False stays bit-identical on cols 0..1 and leaves cols 2..3 untouched
    out_t1 = torch.full((rows, 4), -7.0, dtype=torch.float32, device=DEVICE)
    project_head_markov_greedy_pairs(
        layer, x, base, vocab=vocab, vocab_start=vocab_start, out=out_t1, top2=False
    )
    assert torch.equal(out_t1[:, :2].view(torch.int32), expected[:, :2].view(torch.int32))
    assert (out_t1[:, 2:] == -7.0).all()
