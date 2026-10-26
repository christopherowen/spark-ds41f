# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12X logits heads (shared and Markov) prepared as DeepSeek V4.1 declares them."""

import pytest
import torch


def _device():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("native b12x integration requires SM12x")
    return torch.device("cuda", torch.cuda.current_device())


@torch.no_grad()
def test_weights_collection_prepares_shared_and_markov_logits_heads(
    default_vllm_config,
    dist_init,
    monkeypatch,
):
    from b12x.preparation import PreparationSession

    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.model_executor.warmup.b12x_prepare import _units_from_modules
    from vllm.utils.b12x import B12xWorkload

    device = _device()
    monkeypatch.setattr(default_vllm_config.kernel_config, "linear_backend", "b12x")
    torch.manual_seed(411)
    target, draft = torch.nn.Module(), torch.nn.Module()
    with torch.device(device):
        target.lm_head = ParallelLMHead(32768, 5120, params_dtype=torch.bfloat16)
    target.lm_head.weight.normal_(std=5120**-0.5)
    target.logits_processor = LogitsProcessor(32768)
    draft.logits_processor = LogitsProcessor(32768)
    draft.lm_head = target.lm_head
    assert not target.logits_processor._b12x_vocab_heads
    assert not draft.logits_processor._b12x_vocab_heads
    with torch.device(device):
        markov = ParallelLMHead(
            32768, 256, params_dtype=torch.bfloat16, disable_tp=True
        )
    markov.weight.normal_(std=256**-0.5)
    draft.logits_processor.prepare_b12x_vocab_projection(markov)
    workload = B12xWorkload(
        stage="weights",
        token_counts=(1, 8),
        fixed_token_counts=(1,),
        output_dtype=torch.bfloat16,
        max_tokens=8,
        max_seqs=8,
        max_model_len=1048576,
    )
    seen = set()
    units = tuple(_units_from_modules(target, workload, seen=seen)) + tuple(
        _units_from_modules(draft, workload, seen=seen)
    )
    processors = (target.logits_processor, draft.logits_processor)
    assert len(units) == 3 and all(unit.stage == "weights" for unit in units)
    source = torch.randn(8, 5120, dtype=torch.bfloat16, device=device)
    markov_source = torch.randn(8, 256, dtype=torch.bfloat16, device=device)
    with PreparationSession(device=device, autotune=True, compile_workers=2) as session:
        session.prepare(tuple(request for unit in units for request in unit.requests))
        session.freeze()
        for rows in (1, 7, 8):
            expected = torch.nn.functional.linear(source[:rows], target.lm_head.weight)
            for processor in processors:
                actual = processor(target.lm_head, source[:rows])
                torch.testing.assert_close(actual, expected, rtol=0.008, atol=1e-4)
                assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
            bias = draft.logits_processor(markov, markov_source[:rows])
            expected_bias = torch.nn.functional.linear(
                markov_source[:rows], markov.weight
            )
            torch.testing.assert_close(bias, expected_bias, rtol=0.008, atol=1e-4)
            assert torch.isfinite(bias).all() and torch.count_nonzero(bias) > 0
        graph = torch.cuda.CUDAGraph()
        try:
            with session.capture(), torch.cuda.graph(graph):
                actual = draft.logits_processor(target.lm_head, source)
                bias = draft.logits_processor(markov, markov_source)
            pointer = actual.data_ptr()
            bias_pointer = bias.data_ptr()
            source.neg_()
            target.lm_head.weight.mul_(0.5)
            markov_source.neg_()
            markov.weight.mul_(0.5)
            bias.fill_(float("nan"))
            actual.fill_(float("nan"))
            allocated = torch.cuda.memory_allocated(device)
            graph.replay()
            torch.cuda.synchronize(device)
            assert (
                torch.cuda.memory_allocated(device) == allocated
                and actual.data_ptr() == pointer
            )
            expected = torch.nn.functional.linear(source, target.lm_head.weight)
            torch.testing.assert_close(actual, expected, rtol=0.008, atol=1e-4)
            assert bias.data_ptr() == bias_pointer
            expected_bias = torch.nn.functional.linear(markov_source, markov.weight)
            torch.testing.assert_close(bias, expected_bias, rtol=0.008, atol=1e-4)
        finally:
            graph.reset()


@torch.no_grad()
def test_packed_shared_logits_head_is_exact_and_batch_invariant(
    default_vllm_config,
    dist_init,
    monkeypatch,
):
    from b12x.gemm import packed_bf16_vocab_projection as projection
    from b12x.preparation import PreparationSession

    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.model_executor.warmup.b12x_prepare import _units_from_modules
    from vllm.utils.b12x import B12xWorkload

    device = _device()
    monkeypatch.setattr(default_vllm_config.kernel_config, "linear_backend", "b12x")
    torch.manual_seed(412)
    target, draft = torch.nn.Module(), torch.nn.Module()
    with torch.device(device):
        target.lm_head = ParallelLMHead(
            32768, 5120, params_dtype=torch.bfloat16, lm_head_quantization="packed_bf16"
        )
    assert target.lm_head.runtime_lm_head_quantization == "packed_bf16"
    target.lm_head.weight.normal_(std=0.02)
    target.lm_head.weight[5, :7] = torch.tensor(
        [0.0, -0.0, 1e-30, 3.0, -2.5, 1e-7, 6e-39], dtype=torch.bfloat16
    )
    original = target.lm_head.weight.detach().clone()
    target.lm_head.quant_method.process_weights_after_loading(target.lm_head)
    assert target.lm_head.weight.dtype == torch.uint8
    assert target.lm_head.weight.shape == (32768, 7680)
    packed = target.lm_head.quant_method.packed_weight(target.lm_head)
    assert packed.exceptions > 0
    assert torch.equal(
        projection.unpack(packed).view(torch.int16), original.view(torch.int16)
    )
    target.logits_processor = LogitsProcessor(32768)
    draft.logits_processor = LogitsProcessor(32768)
    draft.lm_head = target.lm_head
    workload = B12xWorkload(
        stage="weights",
        token_counts=(1, 8),
        fixed_token_counts=(1,),
        output_dtype=torch.bfloat16,
        max_tokens=8,
        max_seqs=8,
        max_model_len=1048576,
    )
    seen = set()
    units = tuple(_units_from_modules(target, workload, seen=seen)) + tuple(
        _units_from_modules(draft, workload, seen=seen)
    )
    assert len(units) == 1 and units[0].stage == "weights"
    source = torch.randn(8, 5120, dtype=torch.bfloat16, device=device)
    processors = (target.logits_processor, draft.logits_processor)
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare(tuple(request for unit in units for request in unit.requests))
        session.freeze()
        first = processors[0](target.lm_head, source[:1])
        for rows in (1, 7, 8):
            expected = torch.nn.functional.linear(source[:rows].float(), original.float())
            for processor in processors:
                actual = processor(target.lm_head, source[:rows])
                torch.testing.assert_close(actual.float(), expected, rtol=0.008, atol=2e-3)
                # A token's logits do not depend on what else is in the batch.
                assert torch.equal(actual[:1].view(torch.int16), first.view(torch.int16))
        expected = draft.logits_processor(target.lm_head, source).clone()
        graph = torch.cuda.CUDAGraph()
        try:
            with session.capture(), torch.cuda.graph(graph):
                actual = draft.logits_processor(target.lm_head, source)
            actual.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        finally:
            graph.reset()
