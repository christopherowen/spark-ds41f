# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("tilelang")
if not torch.cuda.is_available():
    pytest.skip(
        "CUDA required for Model Runner V2 thinking budget tests",
        allow_module_level=True,
    )

from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample import tilelang_thinking
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.thinking_budget import ThinkingBudgetState
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device("cuda")
START = 90
END = 91
END_A = 92
END_B = 93
VOCAB_SIZE = 128

_SM12X = pytest.mark.skipif(
    not current_platform.is_device_capability_family(120),
    reason="TileLang thinking-budget kernels target SM120/SM121",
)


def triton_apply_thinking_budget(
    logits,
    req_ids,
    expanded_idx_mapping,
    thinking_token_budget,
    all_token_ids,
    total_len,
    input_ids,
    expanded_local_pos,
    cached_last_start,
    cached_last_end,
    cached_scan_pos,
    reasoning_start_token_ids,
    natural_reasoning_end_token_ids,
    reasoning_end_token_ids,
):
    """The Triton kernels the TileLang ones replaced, launched as the sampler did."""
    from vllm.v1.worker.gpu.sample.thinking_budget import (
        _COLD_SCAN_BLOCK,
        _thinking_budget_kernel,
        _update_committed_marker_cache_kernel,
    )

    start_len = reasoning_start_token_ids.shape[0]
    natural_end_len = natural_reasoning_end_token_ids.shape[0]
    _update_committed_marker_cache_kernel[(req_ids.shape[0],)](
        req_ids,
        thinking_token_budget,
        all_token_ids,
        all_token_ids.stride(0),
        total_len,
        cached_last_start,
        cached_last_end,
        cached_scan_pos,
        reasoning_start_token_ids,
        natural_reasoning_end_token_ids,
        START_LEN=start_len,
        NATURAL_END_LEN=natural_end_len,
        MAX_LEN=max(start_len, natural_end_len),
        BLOCK=_COLD_SCAN_BLOCK,
    )
    _thinking_budget_kernel[(logits.shape[0],)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        thinking_token_budget,
        all_token_ids,
        all_token_ids.stride(0),
        total_len,
        input_ids,
        expanded_local_pos,
        cached_last_start,
        cached_last_end,
        reasoning_start_token_ids,
        natural_reasoning_end_token_ids,
        reasoning_end_token_ids,
        START_LEN=start_len,
        NATURAL_END_LEN=natural_end_len,
        END_LEN=reasoning_end_token_ids.shape[0],
    )


@pytest.fixture(params=["triton", "tilelang"], autouse=True)
def thinking_implementation(request, monkeypatch):
    # The sampler launches the TileLang kernels; the Triton ones are the reference.
    if request.param == "triton":
        from vllm.v1.worker.gpu.sample import thinking_budget

        monkeypatch.setattr(
            thinking_budget, "apply_thinking_budget", triton_apply_thinking_budget
        )


class MockReasoningConfig:
    reasoning_start_token_ids = [START]
    reasoning_end_token_ids = [END]
    natural_reasoning_end_token_ids = [END]


class MockMultiTokenEndReasoningConfig:
    reasoning_start_token_ids = [START]
    reasoning_end_token_ids = [END_A, END_B]
    natural_reasoning_end_token_ids = [END_A, END_B]


class MockDistinctEndReasoningConfig:
    reasoning_start_token_ids = [START]
    reasoning_end_token_ids = [END_A, END_B]
    natural_reasoning_end_token_ids = [END]


def _make_req_states(tokens: list[int], prompt_len: int = 1) -> RequestState:
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=max(64, len(tokens) + 1),
        max_num_batched_tokens=16,
        num_speculative_steps=4,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    req_states.add_request(
        req_id="req",
        prompt_len=prompt_len,
        all_token_ids=tokens,
        num_computed_tokens=len(tokens),
        max_tokens=32,
    )
    req_states.apply_staged_writes()
    return req_states


def _apply(
    state: ThinkingBudgetState,
    logits: torch.Tensor,
    input_ids: list[int],
    local_pos: list[int],
) -> torch.Tensor:
    idx_mapping = torch.tensor([3], dtype=torch.int32, device=DEVICE)
    expanded_idx_mapping = torch.tensor(
        [3] * len(input_ids), dtype=torch.int32, device=DEVICE
    )
    idx_mapping_np = idx_mapping.cpu().numpy()
    state.apply(
        logits,
        expanded_idx_mapping,
        idx_mapping,
        idx_mapping_np,
        torch.tensor(input_ids, dtype=torch.int32, device=DEVICE),
        torch.tensor(local_pos, dtype=torch.int32, device=DEVICE),
    )
    return logits.cpu()


def test_v2_thinking_budget_forces_end_after_budget_reached():
    req_states = _make_req_states([1, START, 10, 11, 12], prompt_len=1)
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.arange(VOCAB_SIZE, dtype=torch.float32, device=DEVICE).view(1, -1)
    expected = logits.cpu()
    out = _apply(state, logits, input_ids=[12], local_pos=[0])

    expected[0, END] = 1.0e9
    torch.testing.assert_close(out, expected)


def test_v2_thinking_budget_restores_masked_end_token():
    req_states = _make_req_states([1, START, 10, 11, 12], prompt_len=1)
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    logits[0, END] = -float("inf")
    out = _apply(state, logits, input_ids=[12], local_pos=[0])

    assert out[0, END] == pytest.approx(1.0e9)


def test_v2_thinking_budget_allows_tokens_before_budget():
    req_states = _make_req_states([1, START, 10, 11], prompt_len=1)
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[11], local_pos=[0])

    assert torch.all(out == 0)


def test_v2_thinking_budget_continues_multi_token_end_marker():
    req_states = _make_req_states([1, START, 10, 11, 12], prompt_len=1)
    state = ThinkingBudgetState(req_states, MockMultiTokenEndReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((2, VOCAB_SIZE), device=DEVICE)
    out = _apply(
        state,
        logits,
        input_ids=[12, END_A],
        local_pos=[0, 1],
    )

    assert out[0, END_A] == pytest.approx(1.0e9)
    assert out[1, END_B] == pytest.approx(1.0e9)


def test_v2_thinking_budget_uses_distinct_forced_end_marker():
    req_states = _make_req_states([1, START, 10, 11, 12], prompt_len=1)
    state = ThinkingBudgetState(req_states, MockDistinctEndReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((2, VOCAB_SIZE), device=DEVICE)
    out = _apply(
        state,
        logits,
        input_ids=[12, END_A],
        local_pos=[0, 1],
    )

    assert out[0, END_A] == pytest.approx(1.0e9)
    assert out[1, END_B] == pytest.approx(1.0e9)


def test_v2_thinking_budget_stops_after_natural_end_marker():
    req_states = _make_req_states(
        [1, START, 10, END, 20, 21, 22],
        prompt_len=1,
    )
    state = ThinkingBudgetState(req_states, MockDistinctEndReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[22], local_pos=[0])

    assert torch.all(out == 0)


def test_v2_thinking_budget_ignores_plain_request():
    req_states = _make_req_states([1, START, 10, 11, 12], prompt_len=1)
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams())
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[12], local_pos=[0])

    assert torch.all(out == 0)


def test_v2_greedy_sampling_applies_thinking_budget():
    """Greedy-only requests must not bypass thinking-budget processing."""
    req_states = _make_req_states([1, START, 10, 11, 12], prompt_len=1)
    sampler = Sampler(
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
        req_states=req_states,
        reasoning_config=MockReasoningConfig(),
    )
    sampler.add_request(
        req_idx=3,
        prompt_len=1,
        sampling_params=SamplingParams(
            temperature=0.0,
            thinking_token_budget=3,
        ),
    )
    sampler.apply_staged_writes()

    idx_mapping = torch.tensor([3], dtype=torch.int32, device=DEVICE)
    idx_mapping_np = idx_mapping.cpu().numpy()
    expanded_idx_mapping = idx_mapping.clone()
    input_ids = torch.tensor([12], dtype=torch.int32, device=DEVICE)
    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = sampler.apply_sampling_params(
        logits,
        expanded_idx_mapping,
        idx_mapping,
        idx_mapping_np,
        torch.tensor([4], dtype=torch.int32, device=DEVICE),
        input_ids,
        torch.tensor([0], dtype=torch.int32, device=DEVICE),
    )

    assert out[0, END].item() == pytest.approx(1.0e9)


def test_v2_thinking_budget_latest_prefill_end_disables_forcing():
    req_states = _make_req_states(
        [1, START, 10, 11, 12, END, 13],
        prompt_len=1,
    )
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[13], local_pos=[0])

    assert torch.all(out == 0)


def test_v2_thinking_budget_uses_latest_prefill_start_boundary():
    req_states = _make_req_states(
        [1, START, 10, 11, 12, END, 13, START, 14, 15, 16],
        prompt_len=1,
    )
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[16], local_pos=[0])

    assert out[0, END] == pytest.approx(1.0e9)


def test_v2_thinking_budget_incrementally_scans_long_generation():
    """Guard against rescanning the full token history on every decode step."""
    tokens = [1, START, *([10] * 16382)]
    req_states = _make_req_states(tokens)
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=32768))
    state.apply_staged_writes()

    _apply(state, torch.zeros((1, VOCAB_SIZE), device=DEVICE), [10], [0])
    assert state.cached_scan_pos[3].item() == len(tokens)

    req_states.all_token_ids.stage_write(3, len(tokens), [10])
    req_states.total_len.stage_write_elem(3, len(tokens) + 1)
    req_states.apply_staged_writes()
    _apply(state, torch.zeros((1, VOCAB_SIZE), device=DEVICE), [10], [0])

    assert state.cached_scan_pos[3].item() == len(tokens) + 1


def test_v2_thinking_budget_clamps_oversized_budget():
    """Budgets beyond int32 must not crash and behave as unlimited."""
    req_states = _make_req_states([1, START, 10, 11, 12], prompt_len=1)
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=2**40))
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[12], local_pos=[0])

    assert torch.all(out == 0)


def test_v2_thinking_budget_continues_end_prefix_from_prompt():
    """A resumed prompt ending with a partial forced-end marker must not
    restart the marker sequence and duplicate its first token."""
    req_states = _make_req_states([1, START, 10, 11, END_A], prompt_len=5)
    state = ThinkingBudgetState(req_states, MockMultiTokenEndReasoningConfig())
    state.add_request(3, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, input_ids=[END_A], local_pos=[0])

    assert out[0, END_B] == pytest.approx(1.0e9)
    assert out[0, END_A] == 0


def _reset_rows(case: str, states: int, gen: torch.Generator) -> list[int]:
    if case == "one":
        return [states - 1]
    if case == "repeated":
        return [0, 3, 0]
    if case == "negative":
        return [-1, 2, -states]
    # More requests than a block's threads, negative ones and repeats among them.
    return torch.randint(-states, states, (300,), generator=gen).tolist()


@_SM12X
@pytest.mark.parametrize("index_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("states", [4, 512])
@pytest.mark.parametrize("case", ["one", "repeated", "negative", "many"])
def test_marker_cache_reset_matches_index_fill(index_dtype, states, case):
    """New budgeted requests' marker caches, reset as the three index_fill_ calls
    reset them (start and end -1, scan position 0): repeated requests, a negative
    index wrapping, every other slot kept."""
    gen = torch.Generator().manual_seed(states)
    caches = [
        torch.randint(-5, 1 << 20, (states,), generator=gen, dtype=torch.int32).to(
            DEVICE
        )
        for _ in range(3)
    ]
    idx = torch.tensor(_reset_rows(case, states, gen), dtype=index_dtype)
    idx = idx.to(DEVICE)
    expected = [cache.clone() for cache in caches]
    expected[0].index_fill_(0, idx.long(), -1)
    expected[1].index_fill_(0, idx.long(), -1)
    expected[2].index_fill_(0, idx.long(), 0)
    tilelang_thinking.reset_markers(idx, *caches)
    for name, ours, theirs in zip(("start", "end", "scan"), caches, expected):
        assert torch.equal(ours, theirs), name


@_SM12X
def test_new_budgeted_requests_reset_only_their_marker_cache():
    """``apply_staged_writes`` resets the marker cache of each request added with a
    budget, and no other slot's."""
    req_states = _make_req_states([1, START, 10, 11])
    state = ThinkingBudgetState(req_states, MockReasoningConfig())
    stale = torch.arange(7, 11, dtype=torch.int32, device=DEVICE)
    for cache in (
        state.cached_last_start,
        state.cached_last_end,
        state.cached_scan_pos,
    ):
        cache.copy_(stale)
    state.add_request(1, SamplingParams(thinking_token_budget=4))
    state.add_request(3, SamplingParams(thinking_token_budget=None))
    state.add_request(2, SamplingParams(thinking_token_budget=8))
    state.apply_staged_writes()
    assert state.cached_last_start.tolist() == [7, -1, -1, 10]
    assert state.cached_last_end.tolist() == [7, -1, -1, 10]
    assert state.cached_scan_pos.tolist() == [7, 0, 0, 10]
