# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reported ranks count real vocabulary entries, including ties and nonfinite values."""

import pytest
import torch

pytest.importorskip("tilelang")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA rank kernels", allow_module_level=True)

from vllm.v1.worker.gpu.sample import logprob, tilelang_logprob

DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def reference(logits, ids):
    output = torch.empty(logits.shape[0], device="cuda", dtype=torch.int64)
    logprob._ranks_kernel[(logits.shape[0],)](
        output, logits, logits.stride(0), ids, logits.shape[1], BLOCK_SIZE=8192
    )
    return output


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "rows,vocab,padding",
    [(1, 257, 0), (17, 8192, 0), (3, 8193, 13), (16, 32320, 0), (96, 129280, 0)],
)
def test_ranks_match_triton(dtype, rows, vocab, padding):
    torch.manual_seed(27)
    backing = torch.randn(rows, vocab + padding, device="cuda", dtype=dtype)
    logits = backing[:, :vocab]
    before = backing.clone()
    ids = torch.randint(vocab, (rows,), device="cuda", dtype=torch.int64)
    output = torch.full((rows,), -1, device="cuda", dtype=torch.int64)
    tilelang_logprob.token_ranks(logits, ids, output)
    assert torch.equal(output, reference(logits, ids))
    selected = logits.gather(1, ids[:, None])
    assert torch.equal(output, (logits >= selected).sum(-1))
    assert torch.equal(backing, before)


@pytest.mark.parametrize("dtype", DTYPES)
def test_ranks_keep_ties_nonfinite_and_exclude_padding(dtype):
    logits = torch.zeros(6, 8193, device="cuda", dtype=dtype)
    logits[0, 0] = -torch.inf
    logits[1, 0] = torch.inf
    logits[2, 0] = torch.nan
    logits[3, 1] = torch.nan
    logits[4] = -torch.inf
    logits[5, :2] = torch.tensor([0.0, -0.0], device="cuda", dtype=dtype)
    ids = torch.zeros(6, device="cuda", dtype=torch.int32)
    output = torch.empty(6, device="cuda", dtype=torch.int64)
    tilelang_logprob.token_ranks(logits, ids, output)
    expected = (logits >= logits.gather(1, ids.long()[:, None])).sum(-1)
    assert torch.equal(output, expected)
    assert output.tolist() == [8193, 1, 0, 8192, 8193, 8193]
    # The old kernel's only difference is its 8,191 phantom vocabulary
    # entries when the scored token has logit -inf.
    assert (reference(logits, ids) - output).tolist() == [8191, 0, 0, 0, 8191, 0]


@pytest.mark.parametrize("dtype", DTYPES)
def test_ds41f_zero_probability_token_reports_rank_within_vocabulary(dtype):
    """Forced-token scoring must not count DS41F's 1,792 padded columns."""
    vocab = 129280
    logits = torch.zeros(2, vocab, device="cuda", dtype=dtype)
    ids = torch.tensor([0, vocab - 1], device="cuda", dtype=torch.int64)
    logits[0, 0] = -torch.inf
    logits[1, -1] = -torch.inf
    original_ids = ids.clone()
    result = logprob.compute_topk_scores(logits, 0, ids, logits_mode=True)
    assert result.selected_token_ranks.tolist() == [vocab, vocab]
    assert reference(logits, ids).tolist() == [131072, 131072]
    assert torch.isneginf(result.logprobs[:, 0]).all()
    assert torch.equal(result.logprob_token_ids[:, 0], original_ids)
    assert torch.equal(ids, original_ids)


def test_rank_is_independent_of_batch_and_graph_replay():
    logits = torch.randn(17, 129280, device="cuda")
    ids = torch.randint(129280, (17,), device="cuda", dtype=torch.int64)
    output = torch.empty(17, device="cuda", dtype=torch.int64)
    tilelang_logprob.token_ranks(logits, ids, output)
    expected = output.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            tilelang_logprob.token_ranks(logits, ids, output)
    torch.cuda.current_stream().wait_stream(side)
    output.fill_(-1)
    graph.replay()
    assert torch.equal(output, expected)
    single = torch.empty(1, device="cuda", dtype=torch.int64)
    tilelang_logprob.token_ranks(logits[:1], ids[:1], single)
    assert torch.equal(single, expected[:1])


@pytest.mark.parametrize("logits_mode", [False, True])
def test_reported_topk_scores_keep_original_ranks(logits_mode):
    logits = torch.randn(3, 8193, device="cuda")
    ids = torch.tensor([0, 8192, 17], device="cuda")
    expected = reference(logits, ids)
    result = logprob.compute_topk_scores(logits, 5, ids, logits_mode=logits_mode)
    assert torch.equal(result.selected_token_ranks, expected)


def test_empty_batch():
    logits = torch.empty(0, 129280, device="cuda")
    ids = torch.empty(0, device="cuda", dtype=torch.int64)
    output = torch.empty_like(ids)
    tilelang_logprob.token_ranks(logits, ids, output)
    assert output.numel() == 0


def test_ranks_preserve_subnormal_comparisons():
    tiny = torch.finfo(torch.float32).smallest_normal / 8
    logits = torch.tensor([[-tiny, 0.0, tiny, -0.0, tiny * 2]] * 5, device="cuda")
    ids = torch.arange(5, device="cuda", dtype=torch.int64)
    output = torch.empty(5, device="cuda", dtype=torch.int64)
    tilelang_logprob.token_ranks(logits, ids, output)
    assert torch.equal(output, reference(logits, ids))


def test_rank_calls_are_prepared_for_both_token_id_types():
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
    from vllm.models.deepseek_v4_1.tilelang import prepare

    registry = JitWarmupRegistry(None)
    with registry.activate():
        logprob.register_topk_scores_warmup(
            logits=[(torch.float32, 129280)], vocab_size=129280, logits_mode=True
        )
    calls = prepare.declared_calls(registry)
    assert len(calls) == 2
    prepare.prepare_declared(registry)
    logits = torch.randn(3, 129280, device="cuda")
    output = torch.empty(3, device="cuda", dtype=torch.int64)
    for dtype in (torch.int32, torch.int64):
        ids = torch.zeros(3, device="cuda", dtype=dtype)
        tilelang_logprob.token_ranks(logits, ids, output)
        assert torch.equal(output, reference(logits, ids))
