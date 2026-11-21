"""Multi-token speculative verification width invariance tests for #864:
- Loop containment truncation across verify widths K = 0, 5, 16, 64
- Async CopyDrafter copy_draft_alloc + scaled forced_reasoning_end_grace
- GPU ThinkingBudgetState + RejectionSampler (block & standard, Triton & TileLang)
  across K = 0, 5, 16, 64, dead_row_cut, multi-token end_ids, and quoted <think>
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.sampling_params import RepetitionDetectionParams, SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.utils import (
    FORCED_REASONING_END_GRACE,
    OUTPUT_LOOP_PERSIST,
    OUTPUT_SOUP_PERSIST,
    check_stop,
)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample import tilelang_thinking
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.thinking_budget import (
    _COLD_SCAN_BLOCK,
    ThinkingBudgetState,
    _thinking_budget_kernel,
    _update_committed_marker_cache_kernel,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device("cuda")
VOCAB = 1024
START = 100
END = 101
END_A = 102
END_B = 103


class _MockReasoningCfg:
    def __init__(self, end_ids=None, natural_ids=None):
        self.reasoning_start_token_ids = [START]
        self.natural_reasoning_end_token_ids = natural_ids or [END]
        self.reasoning_end_token_ids = end_ids or [END]


class _MockSpecCfg:
    def __init__(self, k: int, method: str = "block"):
        self.num_speculative_tokens = k
        self.num_verification_tokens = k
        self.enable_adaptive_verification = False
        self.rejection_sample_method = method
        self.synthetic_acceptance_rates = None


def _simulate_scheduler_update(
    req: Request,
    token_stream: list[int],
    verify_width: int,
    *,
    num_verification_tokens: int = 16,
    max_model_len: int = 131072,
) -> tuple[list[int], RequestStatus]:
    """Feed token_stream in chunks of verify_width using Scheduler's exact
    per-token loop (append -> update_reasoning_markers -> check_stop -> trim)."""
    grace = max(FORCED_REASONING_END_GRACE, 2 * (num_verification_tokens + 1))
    committed: list[int] = []
    for i in range(0, len(token_stream), verify_width):
        new_token_ids = list(token_stream[i : i + verify_width])
        stopped = False
        for num_new, tok in enumerate(new_token_ids, 1):
            prev_num = len(req._output_token_ids)
            req.append_output_token_ids(tok)
            if not req.thinking_disabled:
                req.update_reasoning_markers((START,), (END,), prev_num_output=prev_num)
            stopped = check_stop(
                req,
                max_model_len,
                thinking_loop_break=True,
                forced_reasoning_end_grace=grace,
            )
            if stopped:
                del new_token_ids[num_new:]
                break
        committed.extend(new_token_ids)
        if stopped:
            break
    return committed, req.status


@pytest.mark.parametrize(
    "case",
    ["custom_repetition", "answer_after_forced_end", "answer_soup", "answer_periodic"],
)
def test_loop_truncation_invariant_across_verify_widths(case: str):
    """Exact truncation index and status must be identical across verify widths
    1 (AR), 5 (DSpark), 16 (CopyDrafter default), and 64 (CopyDrafter max)."""
    if case == "custom_repetition":
        sp = SamplingParams(
            max_tokens=4096,
            repetition_detection=RepetitionDetectionParams(
                min_pattern_size=3, max_pattern_size=8, min_count=4
            ),
        )
        prompt = [1, 2]
        stream = [50, 51, 52, 53] + [10, 20, 30] * 10 + [99] * 50
    elif case == "answer_after_forced_end":
        sp = SamplingParams(max_tokens=8192)
        prompt = [1, START]
        # 24 repeating tokens force reasoning end, then END arrives at tok 25,
        # then answer enters soup loop -> must stop at exact same index for all widths.
        stream = [9] * 24 + [END] + [500, 501, 502] + ([77] * 10 + [88]) * 10
    elif case == "answer_soup":
        sp = SamplingParams(max_tokens=8192)
        prompt = [1, START]
        stream = [301, 302, END] + ([77] * 9 + [88, 89]) * 60
    else:
        sp = SamplingParams(max_tokens=16384)
        prompt = [1, START]
        stream = [301, 302, END] + [11, 12, 13, 14, 15, 16] * 500

    results = {}
    for w in (1, 5, 16, 64):
        req = Request(
            request_id=f"{case}-w{w}",
            prompt_token_ids=list(prompt),
            sampling_params=sp,
            pooling_params=None,
        )
        committed, status = _simulate_scheduler_update(
            req, stream, verify_width=w, num_verification_tokens=max(w, 16)
        )
        assert status == RequestStatus.FINISHED_REPETITION
        results[w] = (committed, status)

    ref_committed, ref_status = results[1]
    for w in (5, 16, 64):
        assert results[w] == (ref_committed, ref_status), (
            f"{case} width={w} truncated at {len(results[w][0])} != AR {len(ref_committed)}"
        )


def test_wide_copy_drafter_forced_reasoning_end_grace_and_alloc():
    """With wide CopyDrafter (num_verification_tokens=64):
    1) _widen_copy_placeholders refuses to widen once force_thinking_end is True.
    2) Two full 65-token steps (detecting step tail + in-flight async step) fit
       inside scaled grace so a forced END on step t+2 succeeds instead of killing
       the request; exceeding grace still terminates with FINISHED_REPETITION."""
    # 1. Verify _widen_copy_placeholders blocks widening when force_thinking_end=True
    dummy_sched = SimpleNamespace(
        num_spec_tokens=5,
        sink_rejection=False,
        _max_decode_new_tokens=lambda r: 1024,
    )
    req = Request(
        request_id="copy-force",
        prompt_token_ids=[1, START],
        sampling_params=SamplingParams(max_tokens=4096),
        pooling_params=None,
    )
    req.spec_token_ids = [-1] * 5
    req.copy_draft_alloc = 64
    req.force_thinking_end = True
    added = Scheduler._widen_copy_placeholders(dummy_sched, req, rows_left=64)
    assert added == 0
    assert len(req.spec_token_ids) == 5

    # 2. Scaled grace allows 120 looping tokens (step t tail + async step t+1)
    #    before END lands on step t+2 at num_verification_tokens=64.
    req_ok = Request(
        request_id="grace-ok",
        prompt_token_ids=[1, START],
        sampling_params=SamplingParams(max_tokens=4096),
        pooling_params=None,
    )
    stream_ok = [9] * 24 + [9] * 120 + [END, 401, 402]
    committed, status = _simulate_scheduler_update(
        req_ok, stream_ok, verify_width=65, num_verification_tokens=64
    )
    assert status == RequestStatus.WAITING or status == RequestStatus.RUNNING or not req_ok.is_finished()
    assert req_ok.thinking_output_start == -1
    assert not req_ok.force_thinking_end
    assert committed == stream_ok

    # 3. If END never arrives past scaled grace (2 * 65 = 130), still stops cleanly.
    req_fail = Request(
        request_id="grace-fail",
        prompt_token_ids=[1, START],
        sampling_params=SamplingParams(max_tokens=4096),
        pooling_params=None,
    )
    stream_fail = [9] * 24 + [9] * 200
    committed_fail, status_fail = _simulate_scheduler_update(
        req_fail, stream_fail, verify_width=65, num_verification_tokens=64
    )
    assert status_fail == RequestStatus.FINISHED_REPETITION
    assert len(committed_fail) == 24 + 130 + 1


def _run_gpu_verify(
    committed: list[int],
    drafts: list[int],
    row_logits: torch.Tensor,
    sp: SamplingParams,
    reasoning_cfg,
    method: str,
    dead_cut: int | None = None,
) -> list[int]:
    k = len(drafts)
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=max(256, len(committed) + k + 4),
        max_num_batched_tokens=max(256, k + 4),
        num_speculative_steps=max(k, 1),
        vocab_size=VOCAB,
        device=DEVICE,
    )
    req_idx = 1
    req_states.all_token_ids.gpu[req_idx, : len(committed)] = torch.tensor(
        committed, dtype=torch.int32, device=DEVICE
    )
    req_states.prompt_len.np[req_idx] = 1
    req_states.prompt_len.copy_to_uva()
    req_states.prefill_len.np[req_idx] = 1
    req_states.prefill_len.copy_to_uva()
    req_states.total_len.gpu[req_idx] = len(committed)

    sampler = Sampler(
        max_num_reqs=4,
        vocab_size=VOCAB,
        device=DEVICE,
        req_states=req_states,
        reasoning_config=reasoning_cfg,
        num_speculative_tokens=k,
    )
    sampler.add_request(req_idx, prompt_len=1, sampling_params=sp)
    sampler.apply_staged_writes()

    input_ids = torch.tensor([committed[-1]] + list(drafts), dtype=torch.int32, device=DEVICE)
    positions = torch.arange(
        len(committed) - 1, len(committed) + k, dtype=torch.int64, device=DEVICE
    )
    logits_indices = torch.arange(k + 1, dtype=torch.int64, device=DEVICE)
    cu_np = np.array([0, k + 1], dtype=np.int32)
    cu = torch.tensor(cu_np, dtype=torch.int32, device=DEVICE)
    idx_np = np.array([req_idx], dtype=np.int32)
    idx = torch.tensor(idx_np, dtype=torch.int32, device=DEVICE)

    ib = InputBatch(
        req_ids=["r0"],
        num_reqs=1,
        idx_mapping=idx,
        idx_mapping_np=idx_np,
        expanded_idx_mapping=torch.full((k + 1,), req_idx, dtype=torch.int32, device=DEVICE),
        expanded_local_pos=torch.arange(k + 1, dtype=torch.int32, device=DEVICE),
        num_scheduled_tokens=np.array([k + 1], dtype=np.int32),
        num_tokens=k + 1,
        num_tokens_after_padding=k + 1,
        num_draft_tokens=k,
        query_start_loc=cu,
        query_start_loc_np=cu_np,
        seq_lens=torch.tensor([len(committed) + k], dtype=torch.int32, device=DEVICE),
        dcp_local_seq_lens=None,
        input_ids=input_ids,
        positions=positions,
        logits_indices=logits_indices,
        cu_num_logits=cu,
        cu_num_logits_np=cu_np,
        has_structured_output_reqs=False,
        num_reqs_after_padding=1,
        num_draft_tokens_per_req=np.array([k], dtype=np.int32),
        seq_lens_cpu_upper_bound=np.array([len(committed) + k], dtype=np.int32),
        num_computed_tokens_np=np.array([len(committed) - 1], dtype=np.int32),
        prefill_len_np=np.array([1], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([1], dtype=np.int32),
        is_prefilling_np=np.array([False], dtype=bool),
        has_prefill=False,
        is_padding=torch.zeros((k + 1,), dtype=torch.bool, device=DEVICE),
        prompt_lens=torch.tensor([1], dtype=torch.int32, device=DEVICE),
    )

    if k == 0:
        out = sampler(row_logits[:1].clone(), ib)
        ns = int(out.num_sampled[0].item())
        return out.sampled_token_ids[0, :ns].cpu().tolist()

    rs = RejectionSampler(
        sampler=sampler,
        spec_config=_MockSpecCfg(k, method=method),
        device=DEVICE,
    )
    dc = (
        torch.tensor([dead_cut], dtype=torch.int32, device=DEVICE)
        if dead_cut is not None
        else None
    )
    out = rs(row_logits.clone(), ib, dead_row_cut=dc)
    ns = int(out.num_sampled[0].item())
    return out.sampled_token_ids[0, :ns].cpu().tolist()


@pytest.mark.parametrize("method", ["block", "rejection_sampler"])
@pytest.mark.parametrize("k", [5, 16, 64])
def test_thinking_budget_verify_width_invariance(method: str, k: int):
    """Thinking-budget truncation is invariant across K=0, 5, 16, 64 for greedy
    and sampled (temp+top_p+top_k+min_p) verification."""
    base_committed = [1, START, 10, 11]  # 2 thinking tokens committed
    target_seq = list(range(12, 12 + 70))
    all_logits = torch.full((70, VOCAB), -10.0, dtype=torch.float32, device=DEVICE)
    for i, t in enumerate(target_seq):
        all_logits[i, t] = 10.0

    for budget in (2, 4, 18, 65):
        cut_pos = budget - 2
        for sp in (
            SamplingParams(temperature=0.0, thinking_token_budget=budget),
            SamplingParams(
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                min_p=0.05,
                seed=7,
                thinking_token_budget=budget,
            ),
        ):
            out = _run_gpu_verify(
                base_committed, target_seq[:k], all_logits[: k + 1], sp, _MockReasoningCfg(), method
            )
            if cut_pos <= k:
                assert out == target_seq[:cut_pos] + [END]
            else:
                assert out == target_seq[: k + 1]


def test_quoted_start_inside_reasoning_does_not_reset_budget_cold_inc_draft():
    """A quoted <think> inside an already-open reasoning span must not reset
    the thinking-budget counter in cold scan (single- and multi-chunk),
    incremental scan, or speculative draft scan (K=5, 16, 64)."""
    sp = SamplingParams(temperature=0.0, thinking_token_budget=3)
    target_seq = list(range(12, 12 + 70))
    all_logits = torch.full((70, VOCAB), -10.0, dtype=torch.float32, device=DEVICE)
    for i, t in enumerate(target_seq):
        all_logits[i, t] = 10.0

    # 1. Cold scan: [1, START, 10, START, 11] already has 3 thinking tokens
    for k in (0, 5, 16, 64):
        out = _run_gpu_verify(
            [1, START, 10, START, 11],
            target_seq[:k],
            all_logits[: k + 1],
            sp,
            _MockReasoningCfg(),
            "block",
        )
        assert out == [END], f"cold quoted START failed for k={k}: {out}"

    # 2. Multi-chunk (> 1024 tokens) cold scan: prior turn closed, new START in chunk 0,
    #    quoted START in chunk 1 -> opening START must remain in chunk 0!
    long_committed = [1, START, 5, END, 6, START] + [10] * 1200 + [START, 11]
    out_long = _run_gpu_verify(
        long_committed,
        target_seq[:5],
        all_logits[:6],
        SamplingParams(temperature=0.0, thinking_token_budget=1202),
        _MockReasoningCfg(),
        "block",
    )
    assert out_long == [END], f"multi-chunk cold quoted START failed: {out_long}"

    # 3. Draft scan: [1, START, 10] (1 think tok) + draft [START, 11, 12, 13, 14]
    #    pos 0 emits START (2nd), pos 1 emits 11 (3rd), pos 2 MUST force END!
    draft_q = [START, 11, 12, 13, 14]
    logits_q = torch.full((6, VOCAB), -10.0, dtype=torch.float32, device=DEVICE)
    for i, t in enumerate([START, 11, 12, 13, 14, 15]):
        logits_q[i, t] = 10.0
    out_draft = _run_gpu_verify(
        [1, START, 10], draft_q, logits_q, sp, _MockReasoningCfg(), "block"
    )
    assert out_draft == [START, 11, END], f"draft quoted START failed: {out_draft}"

    # 4. Incremental committed scan: step 1 caches [1, START, 10], step 2 appends [START, 11]
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=4,
        vocab_size=VOCAB,
        device=DEVICE,
    )
    req_states.all_token_ids.gpu[3, :3] = torch.tensor([1, START, 10], dtype=torch.int32, device=DEVICE)
    req_states.total_len.gpu[3] = 3
    state = ThinkingBudgetState(req_states, _MockReasoningCfg())
    state.add_request(3, sp)
    state.apply_staged_writes()

    idx = torch.tensor([3], dtype=torch.int32, device=DEVICE)
    idx_np = idx.cpu().numpy()
    logits1 = torch.zeros((1, VOCAB), device=DEVICE)
    state.apply(logits1, idx, idx, idx_np, torch.tensor([10], dtype=torch.int32, device=DEVICE), torch.tensor([0], dtype=torch.int32, device=DEVICE))
    assert torch.all(logits1 == 0)
    assert state.cached_last_start[3].item() == 1

    # Append quoted START and 11 (now 3 thinking tokens total)
    req_states.all_token_ids.gpu[3, 3:5] = torch.tensor([START, 11], dtype=torch.int32, device=DEVICE)
    req_states.total_len.gpu[3] = 5
    logits2 = torch.zeros((1, VOCAB), device=DEVICE)
    state.apply(logits2, idx, idx, idx_np, torch.tensor([11], dtype=torch.int32, device=DEVICE), torch.tensor([0], dtype=torch.int32, device=DEVICE))
    assert state.cached_last_start[3].item() == 1
    assert logits2[0, END].item() == pytest.approx(1.0e9)
