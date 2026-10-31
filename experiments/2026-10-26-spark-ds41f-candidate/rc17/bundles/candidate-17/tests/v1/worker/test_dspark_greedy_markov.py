# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact full-vocabulary DSpark greedy tokens, including graph replay and
online confidence-head calibration."""

from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.acceptance_estimator import (
    OnlineAcceptanceEstimator,
)
from vllm.v1.worker.gpu.spec_decode.dspark.greedy import (
    sample_greedy_markov,
    scratch_shape,
)


@pytest.mark.parametrize("needs_distribution", [False, True])
def test_sequential_sharded_sampling_preserves_each_token(
    needs_distribution, monkeypatch
):
    from vllm.v1.worker.gpu import tilelang_inputs
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    # The heads' TileLang row gather (its own GPU test) as torch, on the host.
    monkeypatch.setattr(
        tilelang_inputs, "gather_hidden_rows", lambda source, indices: source[indices]
    )

    shared = torch.zeros(2, dtype=torch.int64)
    seen_anchors = []

    def embed(ids):
        seen_anchors.append(ids.clone())
        return ids[:, None].float()

    def sample(base, bias):
        return shared.copy_((base + bias).argmax(-1))

    # Distinct position maxima expose overwrites from a reused argmax output.
    base = torch.tensor([[3.0, 0.0], [0.0, 3.0], [3.0, 0.0], [0.0, 3.0]])
    model = SimpleNamespace(
        supports_local_draft_argmax=lambda: True,
        compute_local_draft_logits=Mock(return_value=base),
        markov_embed=embed,
        compute_local_markov_bias=lambda x: x.expand(-1, 2),
        sample_local_draft_logits=Mock(side_effect=sample),
        gather_local_draft_logits=Mock(side_effect=lambda b, m: b + m),
    )
    spec = SimpleNamespace(
        _draft_topk=None,
        num_speculative_steps=2,
        model=model,
        sample_indices=torch.arange(4),
        sample_idx_mapping=torch.tensor([0, 0, 1, 1]),
        sample_pos=torch.arange(4),
        use_confidence_head=False,
        input_buffers=SimpleNamespace(input_ids=torch.tensor([5, 9])),
        _anchor_idx=torch.tensor([0, 1]),
        draft_tokens=torch.full((2, 2), -1, dtype=torch.int64),
        draft_logits=object() if needs_distribution else None,
        acceptance_estimator=None,
        _sample_logits=Mock(side_effect=lambda logits, *args: logits.argmax(-1)),
    )
    spec._vocab_parallel_greedy = MethodType(
        DSparkSpeculator._vocab_parallel_greedy, spec
    )
    DSparkSpeculator._sample_sequential(spec, 2, torch.zeros(4, 8))
    assert spec.draft_tokens.tolist() == [[0, 1], [0, 1]]
    assert [ids.tolist() for ids in seen_anchors] == [[5, 9], [0, 0]]
    assert model.sample_local_draft_logits.call_count == (
        0 if needs_distribution else 2
    )
    assert model.gather_local_draft_logits.call_count == (
        2 if needs_distribution else 0
    )


def test_confidence_head_keeps_vocab_parallel_greedy_and_calibrates_probs(
    monkeypatch,
):
    """When use_confidence_head is True, keeping OnlineAcceptanceEstimator
    enabled must NOT disable _vocab_parallel_greedy or local draft argmax, and
    must calibrate draft_token_confidence_probs in place."""
    from vllm.v1.worker.gpu import tilelang_inputs
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    monkeypatch.setattr(
        tilelang_inputs, "gather_hidden_rows", lambda source, indices: source[indices]
    )

    device = torch.device("cpu")
    estimator = OnlineAcceptanceEstimator.for_confidence_head(4, 2, device)
    estimator.slope.fill_(1.0)
    estimator.intercepts[:] = torch.tensor([0.0, -1.0], device=device)

    raw_conf = torch.tensor([0.80, 0.80, 0.60, 0.60], dtype=torch.float32)
    base = torch.tensor([[3.0, 0.0], [0.0, 3.0], [3.0, 0.0], [0.0, 3.0]])
    model = SimpleNamespace(
        supports_local_draft_argmax=lambda: True,
        vocab_parallel_greedy=True,
        draft_id_to_target_id=None,
        compute_local_draft_logits=Mock(return_value=base),
        markov_embed=lambda ids: ids[:, None].float(),
        compute_local_markov_bias=lambda x: x.expand(-1, 2),
        sample_local_draft_logits=Mock(side_effect=lambda b, m: (b + m).argmax(-1)),
        gather_local_draft_logits=Mock(side_effect=lambda b, m: b + m),
        compute_confidence=Mock(return_value=raw_conf),
    )
    spec = SimpleNamespace(
        _draft_topk=None,
        num_speculative_steps=2,
        model=model,
        sample_indices=torch.arange(4),
        sample_idx_mapping=torch.tensor([2, 2, 3, 3], dtype=torch.int32),
        sample_pos=torch.arange(4),
        use_confidence_head=True,
        input_buffers=SimpleNamespace(input_ids=torch.tensor([5, 9])),
        _anchor_idx=torch.tensor([0, 1]),
        draft_tokens=torch.full((2, 2), -1, dtype=torch.int64),
        draft_token_confidence_probs=torch.zeros((4, 2), dtype=torch.float32),
        draft_logits=None,
        acceptance_estimator=estimator,
        _sample_logits=Mock(side_effect=lambda logits, *args: logits.argmax(-1)),
    )
    spec._vocab_parallel_greedy = MethodType(
        DSparkSpeculator._vocab_parallel_greedy, spec
    )

    # _vocab_parallel_greedy must remain True when use_confidence_head=True
    # even though acceptance_estimator is not None!
    assert spec._vocab_parallel_greedy(local_head=False) is True

    DSparkSpeculator._sample_sequential(spec, 2, torch.zeros(4, 8))
    # Local argmax path was taken without gathering full-vocab logits.
    assert model.sample_local_draft_logits.call_count == 2
    assert model.gather_local_draft_logits.call_count == 0

    # Position 0 has intercept 0.0 (unchanged), position 1 has intercept -1.0.
    expected_pos0 = torch.tensor([0.80, 0.60])
    logits_pos1 = torch.log(expected_pos0 / (1.0 - expected_pos0))
    expected_pos1 = torch.sigmoid(logits_pos1 - 1.0)
    torch.testing.assert_close(
        spec.draft_token_confidence_probs[:2, 0], expected_pos0, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        spec.draft_token_confidence_probs[:2, 1], expected_pos1, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        estimator.predictions[torch.tensor([2, 3]), 1],
        expected_pos1,
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("base_vocab,bias_vocab", [(2049, 2048), (2048, 2049)])
def test_greedy_markov_rejects_mismatched_vocabulary_before_dispatch(
    base_vocab, bias_vocab
):
    """An invalid model head must fail before any GPU buffer is accessed."""
    base = torch.empty((2, base_vocab))
    bias = torch.empty((2, bias_vocab))
    output = torch.empty(2, dtype=torch.int64)
    shape = scratch_shape(2, max(base_vocab, bias_vocab))
    with pytest.raises(ValueError, match="identical shapes"):
        sample_greedy_markov(
            base,
            bias,
            output,
            torch.empty(shape),
            torch.empty(shape, dtype=torch.int32),
        )


@pytest.mark.parametrize("short_values", [False, True])
def test_greedy_markov_rejects_insufficient_scratch_before_dispatch(short_values):
    base = torch.empty((2, 2049))
    bias = torch.empty_like(base)
    output = torch.empty(2, dtype=torch.int64)
    values = torch.empty(3 if short_values else 4)
    indices = torch.empty(4 if short_values else 3, dtype=torch.int32)
    with pytest.raises(ValueError, match="scratch is smaller"):
        sample_greedy_markov(base, bias, output, values, indices)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("vocab", [1, 2047, 2048, 2049, 129280])
@pytest.mark.parametrize("rows", [1, 4])
def test_greedy_markov_exact_replay(dtype, vocab, rows):
    torch.manual_seed(41612)
    storage = torch.randn((rows, 7, vocab), device="cuda", dtype=dtype)
    base = storage[:, 3]
    bias = torch.randn_like(base)
    result = torch.full((rows, 7), -77, device="cuda", dtype=torch.int64)
    output = result[:, 3]
    shape = scratch_shape(rows, vocab)
    values = torch.full(shape, float("nan"), device="cuda")
    indices = torch.full(shape, -1, device="cuda", dtype=torch.int32)
    run = lambda: sample_greedy_markov(base, bias, output, values, indices)
    run()
    torch.testing.assert_close(output, (base + bias).argmax(-1), rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for case in ("mutated", "ties", "negative-infinity", "nan"):
        if case == "mutated":
            base.normal_()
            bias.normal_()
        elif case == "ties":
            base.fill_(-1)
            bias.zero_()
            base[:, 0] = 2
            base[:, -1] = 2
        elif case == "negative-infinity":
            base.fill_(-float("inf"))
            bias.zero_()
        else:
            base.zero_()
            base[:, -1] = float("nan")
            base[:, vocab // 2] = float("nan")
        values.fill_(float("nan"))
        indices.fill_(-1)
        allocated = torch.accelerator.memory_allocated()
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.accelerator.memory_allocated() == allocated
        torch.testing.assert_close(output, (base + bias).argmax(-1), rtol=0, atol=0)
        assert torch.all(result[:, :3] == -77)
        assert torch.all(result[:, 4:] == -77)


def test_parse_max_steps_by_batch_normalizes_and_clamps():
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
        parse_max_steps_by_batch,
    )

    assert parse_max_steps_by_batch("", 5) == ()
    assert parse_max_steps_by_batch("8:4, 16:3", 5) == ((8, 4), (16, 3))
    # Enforces monotonic non-increasing depth as concurrency grows.
    assert parse_max_steps_by_batch({4: 3, 8: 4, 16: 2}, 5) == ((4, 3), (16, 2))
    assert parse_max_steps_by_batch([(2, 9), (8, 0)], 5) == ((8, 1),)


def test_concurrency_bounded_sequential_draft_depth(monkeypatch):
    from vllm.v1.worker.gpu import tilelang_inputs
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
        DSparkSpeculator,
        parse_max_steps_by_batch,
    )

    gathered_rows = []

    def fake_gather(source, indices):
        gathered_rows.append(indices.tolist())
        return source[indices]

    monkeypatch.setattr(tilelang_inputs, "gather_hidden_rows", fake_gather)

    seen_markov = []

    def embed(ids):
        seen_markov.append(ids.clone())
        return ids[:, None].float()

    model = SimpleNamespace(
        supports_local_draft_argmax=lambda: True,
        compute_local_draft_logits=Mock(
            side_effect=lambda h: torch.tensor([[1.0, 5.0]]).expand(h.shape[0], 2)
        ),
        markov_embed=embed,
        compute_local_markov_bias=lambda x: torch.zeros(x.shape[0], 2),
        sample_local_draft_logits=Mock(
            side_effect=lambda b, m: (b + m).argmax(-1)
        ),
        compute_confidence=Mock(
            side_effect=lambda h, m: torch.full((h.shape[0],), 0.75)
        ),
    )
    n_spec = 4
    max_reqs = 4
    spec = SimpleNamespace(
        _draft_topk=None,
        num_speculative_steps=n_spec,
        _max_steps_by_batch=parse_max_steps_by_batch("2:3,4:2", n_spec),
        _active_sample_indices=torch.empty(max_reqs * n_spec, dtype=torch.int64),
        model=model,
        sample_indices=torch.arange(max_reqs * n_spec, dtype=torch.int64),
        sample_idx_mapping=torch.arange(max_reqs * n_spec, dtype=torch.int32) // n_spec,
        sample_pos=torch.arange(max_reqs * n_spec, dtype=torch.int64),
        use_confidence_head=True,
        input_buffers=SimpleNamespace(input_ids=torch.tensor([3, 4, 5, 6])),
        _anchor_idx=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        draft_tokens=torch.full((max_reqs, n_spec), 9, dtype=torch.int64),
        draft_token_confidence_probs=torch.ones((max_reqs, n_spec), dtype=torch.float32),
        draft_logits=None,
        acceptance_estimator=None,
    )
    spec._vocab_parallel_greedy = MethodType(
        DSparkSpeculator._vocab_parallel_greedy, spec
    )
    spec.active_speculative_steps = MethodType(
        DSparkSpeculator.active_speculative_steps, spec
    )

    assert spec.active_speculative_steps(1) == 4
    assert spec.active_speculative_steps(2) == 3
    assert spec.active_speculative_steps(4) == 2

    DSparkSpeculator._sample_sequential(
        spec, 4, torch.zeros(max_reqs * n_spec, 8)
    )
    # Only the first 2 steps per request (8 rows instead of 16) are gathered and
    # projected, and only 2 sequential Markov steps execute.
    assert gathered_rows[-1] == [0, 1, 4, 5, 8, 9, 12, 13]
    assert len(seen_markov) == 2
    assert spec.draft_tokens.tolist() == [
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
    ]
    assert spec.draft_token_confidence_probs.tolist() == [
        [0.75, 0.75, 0.0, 0.0],
        [0.75, 0.75, 0.0, 0.0],
        [0.75, 0.75, 0.0, 0.0],
        [0.75, 0.75, 0.0, 0.0],
    ]

    # Also verify _sample_sequential_topk honors active_speculative_steps.
    topk_model = SimpleNamespace(
        compute_draft_logits=Mock(
            side_effect=lambda h: torch.tensor([[1.0, 2.0, 6.0]]).expand(h.shape[0], 3)
        ),
        markov_embed=lambda ids: ids[:, None].float(),
        apply_markov_bias_gathered=Mock(
            side_effect=lambda emb, base_i, val_i, idx_i: base_i.scatter(
                1, idx_i, val_i
            )
        ),
        compute_confidence=Mock(
            side_effect=lambda h, m: torch.full((h.shape[0],), 0.875)
        ),
    )
    spec._draft_topk = 2
    spec.model = topk_model
    spec._sample_logits = Mock(side_effect=lambda logits_i, *args: logits_i.argmax(-1))
    spec._sample_sequential_topk = DSparkSpeculator._sample_sequential_topk.__get__(spec)
    spec.draft_tokens.fill_(9)
    spec.draft_token_confidence_probs.fill_(1.0)
    DSparkSpeculator._sample_sequential(
        spec, 2, torch.zeros(max_reqs * n_spec, 8)
    )
    assert topk_model.apply_markov_bias_gathered.call_count == 3
    assert spec.draft_tokens[:2].tolist() == [[2, 2, 2, 0], [2, 2, 2, 0]]
    assert spec.draft_token_confidence_probs[:2].tolist() == [
        [0.875, 0.875, 0.875, 0.0],
        [0.875, 0.875, 0.875, 0.0],
    ]
