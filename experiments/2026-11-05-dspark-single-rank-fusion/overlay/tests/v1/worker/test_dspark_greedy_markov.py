# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact full-vocabulary DSpark greedy tokens, including graph replay."""

from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.dspark.greedy import (
    sample_greedy_markov,
    scratch_shape,
)


@pytest.mark.parametrize("sample_from_anchor", [False, True])
@pytest.mark.parametrize("needs_distribution", [False, True])
def test_sequential_sharded_sampling_preserves_each_token(
    sample_from_anchor, needs_distribution, monkeypatch
):
    from vllm.v1.worker.gpu import tilelang_inputs
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    gather_calls = 0

    def gather(source, indices):
        nonlocal gather_calls
        gather_calls += 1
        return source[indices]

    # The heads' TileLang row gather (its own GPU test) as torch, on the host.
    monkeypatch.setattr(tilelang_inputs, "gather_hidden_rows", gather)

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
        sample_from_anchor=sample_from_anchor,
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
    hidden = torch.arange(48, dtype=torch.float32).reshape(6, 8)
    DSparkSpeculator._sample_sequential(spec, 2, hidden)
    assert gather_calls == (0 if sample_from_anchor else 1)
    assert torch.equal(model.compute_local_draft_logits.call_args.args[0], hidden[:4])
    assert spec.draft_tokens.tolist() == [[0, 1], [0, 1]]
    assert [ids.tolist() for ids in seen_anchors] == [[5, 9], [0, 0]]
    assert model.sample_local_draft_logits.call_count == (
        0 if needs_distribution else 2
    )
    assert model.gather_local_draft_logits.call_count == (
        2 if needs_distribution else 0
    )


def _vocab_parallel_speculator(model, rows, n_spec, rank, top2, device):
    """A DSpark speculator's state for the vocabulary-parallel greedy steps, one
    rank's pairs standing in for their all-gather."""
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    generator = torch.Generator(device=device).manual_seed(rows)
    draft_tokens = torch.full((rows, n_spec), -1, dtype=torch.int64, device=device)
    shape = scratch_shape(rows, 129280)
    spec = SimpleNamespace(
        _draft_topk=None,
        num_speculative_steps=n_spec,
        sample_from_anchor=True,
        model=model,
        use_confidence_head=False,
        input_buffers=SimpleNamespace(
            input_ids=torch.randint(
                0, 129280, (rows * n_spec,), generator=generator, device=device,
                dtype=torch.int32,
            )
        ),
        _anchor_idx=torch.arange(rows, device=device) * n_spec,
        sample_idx_mapping=torch.arange(rows * n_spec, device=device),
        sample_pos=torch.arange(rows * n_spec, device=device),
        draft_tokens=draft_tokens,
        draft_logits=None,
        acceptance_estimator=None,
        _enable_top2_branch=top2,
        _top2_branch_margin=0.25 if top2 else 0.0,
        _top2_branch_weight=1.0,
        _markov_conf_weight=torch.randn(
            rank, generator=generator, device=device
        ),
        _greedy_shard_pairs=torch.full((rows, 4), -7.0, device=device),
        _greedy_partial_values=torch.empty(shape, device=device),
        _greedy_partial_indices=torch.empty(shape, dtype=torch.int32, device=device),
        _markov_step_embed=torch.empty(
            (rows, rank), dtype=torch.bfloat16, device=device
        ),
        _markov_confidence_embeds=torch.empty(
            (rows, n_spec, rank), dtype=torch.bfloat16, device=device
        ),
        draft_runner_up_tokens=torch.zeros_like(draft_tokens),
        draft_top2_margins=torch.zeros((rows, n_spec), device=device),
    )
    spec._vocab_parallel_greedy = MethodType(
        DSparkSpeculator._vocab_parallel_greedy, spec
    )
    spec._markov_table = MethodType(DSparkSpeculator._markov_table, spec)
    return spec


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("top2", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "padded,shard_vocab,vocab_start",
    [(129280, 129280, 0), (32320, 32320, 96960), (43136, 43008, 86272)],
    ids=["tp1", "tp4", "tp3"],
)
def test_fused_markov_greedy_steps_match_materialized_bias(
    padded, shard_vocab, vocab_start, dtype, top2, monkeypatch
):
    """Every vocabulary-parallel greedy step through the Markov projection's
    fused reduction draws the tokens, runner-ups, margins and next-step
    embeddings that the materialized bias and ``greedy_markov_shard_pair`` draw."""
    from vllm.platforms import current_platform

    if not current_platform.is_device_capability_family(120):
        pytest.skip("TileLang V4.1 kernels target SM120/SM121")
    pytest.importorskip("tilelang")
    from vllm.models.deepseek_v4_1.nvidia.dspark import DSparkDeepseekV4ForCausalLM
    from vllm.models.deepseek_v4_1.tilelang import vocab
    from vllm.v1.worker.gpu.spec_decode.dspark import speculator
    from vllm.v1.worker.gpu.spec_decode.dspark.greedy import greedy_markov_shard_pair
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    monkeypatch.setattr(
        speculator, "tensor_model_parallel_all_gather", lambda pairs, dim: pairs
    )
    device, n_spec, rank, max_rows = "cuda", 5, 256, 48
    generator = torch.Generator(device=device).manual_seed(padded)
    w2 = torch.randn(padded, rank, generator=generator, device=device)
    w2 = (w2 * 0.05).bfloat16()
    w2[shard_vocab - 1] = w2[3]  # equal biases at both ends of the shard
    packed = vocab.pack_bf16_weight(w2)
    head = SimpleNamespace(
        weight=packed.values,
        weight_exception_rows=packed.exception_rows,
        weight_exception_entries=packed.exception_entries,
        weight_row_offsets=packed.row_offsets,
        packed_bf16_in_features=rank,
        packed_bf16_exponent_base=packed.exponent_base,
        tilelang_max_rows=max_rows,
        tilelang_markov_head=True,
    )
    vocab.prepare_packed_head(head, dtype)
    table = torch.randn(129280, rank, generator=generator, device=device)
    table = (table * 0.5).bfloat16()
    base = torch.randn(
        max_rows * n_spec, padded, generator=generator, device=device
    ).to(dtype)
    base[::7, 3] = base[::7, shard_vocab - 1] = 9.0  # exact ties across the shard

    def materialized(markov_embed, base_logits, pairs, *, top2):
        """The reduction before the fusion: the shard's bias, then its pairs."""
        bias = vocab.project_head(head, markov_embed)
        shape = scratch_shape(pairs.shape[0], shard_vocab)
        greedy_markov_shard_pair(
            base_logits[:, :shard_vocab],
            bias[:, :shard_vocab],
            vocab_start,
            pairs,
            torch.empty(shape, device=device),
            torch.empty(shape, dtype=torch.int32, device=device),
            top2=top2,
        )

    model = SimpleNamespace(
        supports_local_draft_argmax=lambda: False,
        vocab_parallel_greedy=True,
        draft_id_to_target_id=None,
        draft_vocab_shard=(vocab_start, shard_vocab),
        compute_draft_logits_local=lambda hidden: base[: hidden.shape[0]],
        markov_embed_table=lambda: table,
        model=SimpleNamespace(markov_head=SimpleNamespace(markov_w2=head)),
    )
    fused_pairs = MethodType(
        DSparkDeepseekV4ForCausalLM.markov_greedy_pairs_local, model
    )
    hidden = torch.empty((max_rows * n_spec, 8), device=device)
    for rows in (1, 2, 5, 8, 16, 17, 33, 48):
        drafted = []
        for pairs_of in (materialized, fused_pairs):
            model.markov_greedy_pairs_local = pairs_of
            spec = _vocab_parallel_speculator(model, rows, n_spec, rank, top2, device)
            DSparkSpeculator._sample_sequential(spec, rows, hidden)
            drafted.append(
                (
                    spec.draft_tokens,
                    spec.draft_runner_up_tokens,
                    spec.draft_top2_margins.view(torch.int32),
                    spec._markov_step_embed.view(torch.int16),
                    spec._markov_confidence_embeds.view(torch.int16),
                )
            )
        for materialized_out, fused_out in zip(*drafted):
            assert torch.equal(fused_out, materialized_out), (rows, dtype, top2)
        tokens = drafted[0][0]
        in_shard = (tokens >= vocab_start) & (tokens < vocab_start + shard_vocab)
        assert bool(in_shard.all())
        if rows >= 8:
            # The drafts differ: the steps compared more than one token.
            assert tokens.unique().numel() > n_spec


@pytest.mark.parametrize("fuses", [False, True])
def test_vocab_parallel_warmups_need_the_fused_markov_projection(fuses, monkeypatch):
    """Kernel preparation fails for a Markov head without the fused projection,
    and otherwise declares each draft position's selection alone."""
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
    from vllm.v1.worker.gpu.spec_decode.dspark import speculator
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    monkeypatch.setattr(
        DFlashSpeculator, "register_warmups", lambda self, layouts: None
    )
    monkeypatch.setattr(speculator, "get_tensor_model_parallel_world_size", lambda: 3)
    selections = []
    monkeypatch.setattr(
        speculator,
        "register_select_first_argmax_warmup",
        lambda **kwargs: selections.append(kwargs),
    )
    spec = object.__new__(DSparkSpeculator)
    spec.__dict__.update(
        _draft_topk=None,
        num_speculative_steps=5,
        sample_from_anchor=True,
        draft_logits=None,
        acceptance_estimator=None,
        _enable_top2_branch=False,
        draft_tokens=torch.zeros((4, 5), dtype=torch.int64),
        model=SimpleNamespace(
            supports_local_draft_argmax=lambda: False,
            vocab_parallel_greedy=True,
            draft_id_to_target_id=None,
            fuses_markov_greedy=fuses,
            markov_embed_table=lambda: None,
        ),
    )
    if not fuses:
        with pytest.raises(RuntimeError, match="Markov projection"):
            spec.register_warmups([])
        assert selections == []
        return
    spec.register_warmups([])
    assert [kwargs["ranks"] for kwargs in selections] == [3] * 5


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
