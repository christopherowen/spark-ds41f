# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``logprob_token_ids`` under speculative decoding, in the V2 runner.

A request's ``logprob_token_ids`` name the tokens whose logprobs it gets for
every generated token (the OpenAI APIs send them with ``logprobs`` unset). The
sampler returns them for the token of a step without drafts; verification must
return them for every verified token too, and a greedy batch that wants them
must not take greedy verification, which returns no logprobs. Otherwise a chat
completion gets fewer logprob entries than tokens and fails with HTTP 500.

The runner's real sampling state and verification code run on the host: its
device buffers are host tensors, and the Triton kernels they launch are torch
references. ``_verify`` is replaced, as in test_gpu_rejection_sampler_chunking.py.
"""

from types import MethodType, SimpleNamespace
from typing import get_args

import numpy as np
import pytest
import torch

from vllm.config.model import PROCESSED_LOGPROBS_MODES, LogprobsMode
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu import tilelang_inputs
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample import (
    logprob,
    states,
    tilelang_auxiliary,
    tilelang_logprob,
    tilelang_topk_ids,
)
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.sink import SinkAlternates
from vllm.v1.worker.gpu.spec_decode import rejection_sampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

pytestmark = pytest.mark.cpu_test

VOCAB = 32
MAX_REQS = 8
IDS = [3, 17, 29, 8, 0]


class HostTensor:
    """UvaBackedTensor on the host: the numpy view is the "device" tensor."""

    def __init__(self, size, dtype, max_concurrency=None):
        self.cpu = torch.zeros(size, dtype=dtype)
        self.np = self.cpu.numpy()
        self.gpu = self.cpu

    def copy_to_uva(self, n=None):
        return self.gpu


class HostStagedTensor:
    """StagedWriteTensor on the host."""

    def __init__(self, size, dtype, device, **kwargs):
        self.gpu = torch.zeros(size, dtype=dtype)
        self.writes: list[tuple[int, int, list]] = []

    def stage_write(self, index, start, x):
        self.writes.append((index, start, list(x)))

    def apply_write(self):
        for index, start, x in self.writes:
            self.gpu[index, start : start + len(x)] = torch.tensor(x)
        self.writes.clear()


def _fill_logprob_token_ids(
    out_ids,
    _ids_stride,
    valid,
    _valid_stride,
    sampled,
    topk,
    _topk_stride,
    expanded_idx_mapping,
    num_per_req,
    per_req_ids,
    _per_req_stride,
    NUM_TOPK,
    PADDED_COLS,
):
    for row in range(out_ids.shape[0]):
        req = int(expanded_idx_mapping[row])
        n = int(num_per_req[req])
        tokens = per_req_ids[req, :n] if n > 0 else topk[row, :NUM_TOPK]
        out_ids[row, 0] = sampled[row]
        out_ids[row, 1 : 1 + len(tokens)] = tokens
        valid[row, : 1 + len(tokens)] = True


def _fill_token_ids(
    out_ids, valid, sampled, topk, expanded_idx_mapping, num_per_req, per_req_ids, k
):
    """``tilelang_auxiliary.fill_token_ids``: every column written, 0 and invalid
    where unused."""
    out_ids.zero_()
    valid.zero_()
    _fill_logprob_token_ids(
        out_ids,
        None,
        valid,
        None,
        sampled,
        topk,
        None,
        expanded_idx_mapping,
        num_per_req,
        per_req_ids,
        None,
        k,
        None,
    )


def _compute_token_logprobs(logits, ids, valid=None):
    scores = torch.log_softmax(logits.float(), -1).gather(1, ids.long())
    return scores if valid is None else scores.masked_fill(~valid, float("-inf"))


def _token_ranks(logits, token_ids, out):
    chosen = logits.gather(1, token_ids.view(-1, 1).long())
    out.copy_((logits >= chosen).sum(1))


def _flatten_sampled(flat, sampled, _, num_sampled, cu_num_logits, num_warps):
    for req in range(num_sampled.shape[0]):
        n = int(num_sampled[req])
        start = int(cu_num_logits[req])
        flat[start : start + n] = sampled[req, :n]


def _flatten_sampled_rows(sampled, num_sampled, cu_num_logits, num_logits):
    flat = torch.zeros(num_logits, dtype=sampled.dtype)
    _flatten_sampled(flat, sampled, None, num_sampled, cu_num_logits, 1)
    return flat


@pytest.fixture(autouse=True)
def host_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(states, "UvaBackedTensor", HostTensor)
    monkeypatch.setattr(logprob, "UvaBackedTensor", HostTensor)
    monkeypatch.setattr(logprob, "StagedWriteTensor", HostStagedTensor)
    monkeypatch.setattr(tilelang_auxiliary, "fill_token_ids", _fill_token_ids)
    monkeypatch.setattr(
        tilelang_topk_ids,
        "topk_token_ids",
        lambda logits, k: torch.topk(logits, k, dim=-1).indices.int(),
    )
    monkeypatch.setattr(
        tilelang_topk_ids,
        "logprob_token_ids",
        lambda logits, k, sampled: torch.cat(
            (sampled.unsqueeze(-1), torch.topk(logits, k, dim=-1).indices), dim=1
        ),
    )
    monkeypatch.setattr(tilelang_logprob, "token_ranks", _token_ranks)
    monkeypatch.setattr(
        tilelang_logprob,
        "gather_logits",
        lambda logits, ids, valid=None: (
            logits.gather(1, ids.long()).float()
            if valid is None
            else logits.gather(1, ids.long()).float().masked_fill(~valid, -torch.inf)
        ),
    )
    monkeypatch.setattr(logprob, "compute_token_logprobs", _compute_token_logprobs)
    # The TileLang launches the samplers reach, as torch on the host.
    monkeypatch.setattr(tilelang_inputs, "flatten_sampled", _flatten_sampled_rows)
    monkeypatch.setattr(
        tilelang_inputs, "logits_rows", lambda ids, pos, rows: (ids[rows], pos[rows])
    )
    monkeypatch.setattr(
        rejection_sampler,
        "get_num_sampled_and_rejected",
        lambda num_sampled, *args, **kwargs: (
            num_sampled,
            torch.zeros_like(num_sampled),
        ),
    )


def _sampler(params: dict[int, SamplingParams], logprobs_mode: str) -> Sampler:
    """The runner's sampler, its sampling state from the requests' params (by
    request-state index)."""
    sampler = object.__new__(Sampler)
    sampler.logprobs_mode = logprobs_mode
    sampler.compute_nans = False
    sampler.req_states = SimpleNamespace(prefill_len=SimpleNamespace(gpu=None))
    sampler.sampling_states = states.SamplingStates(MAX_REQS, VOCAB)
    sampler.logprob_token_ids_state = logprob.LogprobTokenIdsState(
        MAX_REQS, torch.device("cpu")
    )
    sampler.needs_logits_processing = np.zeros(MAX_REQS, dtype=bool)
    for req_idx, sampling_params in params.items():
        sampler.sampling_states.add_request(req_idx, sampling_params)
        sampler.logprob_token_ids_state.add_request(req_idx, sampling_params)
    sampler.logprob_token_ids_state.apply_staged_writes()
    return sampler


# Request-state index -> (params, logits rows: drafts + bonus, verified tokens).
REQUESTS = {
    5: (SamplingParams(temperature=0.0, logprob_token_ids=IDS), 4, 3),
    1: (SamplingParams(temperature=0.0, logprobs=2), 3, 1),
    3: (SamplingParams(temperature=0.0), 2, 2),
}


def _batch(order: list[int]) -> SimpleNamespace:
    num_logits = np.array([REQUESTS[r][1] for r in order], dtype=np.int32)
    cu_num_logits_np = np.concatenate([[0], np.cumsum(num_logits)]).astype(np.int32)
    idx_mapping_np = np.array(order, dtype=np.int32)
    total = int(cu_num_logits_np[-1])
    return SimpleNamespace(
        num_reqs=len(order),
        num_draft_tokens=total - len(order),
        num_draft_tokens_per_req=num_logits - 1,
        cu_num_logits_np=cu_num_logits_np,
        cu_num_logits=torch.from_numpy(cu_num_logits_np),
        idx_mapping_np=idx_mapping_np,
        idx_mapping=torch.from_numpy(idx_mapping_np),
        expanded_idx_mapping=torch.from_numpy(np.repeat(idx_mapping_np, num_logits)),
        expanded_local_pos=torch.from_numpy(
            np.concatenate([np.arange(n, dtype=np.int32) for n in num_logits])
        ),
        input_ids=torch.arange(total),
        positions=torch.arange(total),
        logits_indices=torch.arange(total),
        seq_lens=torch.full((len(order),), 64, dtype=torch.int32),
    )


def _rejection_sampler(
    sampler: Sampler, sink=None, adaptive: bool = False
) -> RejectionSampler:
    """Verification whose target picks each row's argmax of the processed
    logits (twice the raw ones) and accepts ``REQUESTS``' token counts."""
    rs = object.__new__(RejectionSampler)
    rs.sampler = sampler
    rs.sink = sink
    rs.watermark_key = None
    rs.num_speculative_steps = 3
    rs.num_draft_steps = 3
    rs.enable_adaptive_verification = adaptive

    def verify(
        self,
        logits,
        _draft_logits,
        _drafts,
        _pos,
        cu_num_logits,
        idx_mapping,
        _idx_mapping_np,
        _expanded_idx_mapping,
        _expanded_local_pos,
        _dead_row_cut=None,
        chunk=None,
        _kernel_steps=None,
    ):
        processed = logits.float() * 2
        num_reqs = idx_mapping.shape[0]
        sampled = torch.zeros(
            (num_reqs, self.num_speculative_steps + 1), dtype=torch.int64
        )
        num_sampled = torch.empty(num_reqs, dtype=torch.int32)
        for i in range(num_reqs):
            lo, hi = int(cu_num_logits[i]), int(cu_num_logits[i + 1])
            sampled[i, : hi - lo] = processed[lo:hi].argmax(-1)
            num_sampled[i] = REQUESTS[int(idx_mapping[i])][2]
        if chunk is not None:
            chunk.alternates = sampled[:, :2] + 1
        return processed, sampled, num_sampled

    rs._verify = MethodType(verify, rs)
    return rs


def _scores(logits: torch.Tensor, logprobs_mode: str) -> torch.Tensor:
    """What the logprobs report for each logits row, per ``logprobs_mode``."""
    if logprobs_mode in PROCESSED_LOGPROBS_MODES:
        logits = logits.float() * 2
    if logprobs_mode.endswith("logprobs"):
        return torch.log_softmax(logits.float(), -1)
    return logits.float()


@pytest.mark.parametrize("logprobs_mode", get_args(LogprobsMode))
@pytest.mark.parametrize(
    ("max_chunk_logits", "adaptive"),
    [(None, False), (4, False), (None, True)],
    ids=["one-chunk", "chunks", "adaptive-verification"],
)
@pytest.mark.parametrize("order", [[5], [1, 5, 3]])
def test_verified_tokens_carry_their_logprob_token_ids(
    monkeypatch, logprobs_mode, max_chunk_logits, adaptive, order
):
    """Every verified token of the request with ``logprob_token_ids`` carries
    those ids' logprobs in the distribution of its row, beside its own logprob
    and rank, alone or beside a top-logprobs request and one without logprobs,
    in one verification chunk or several, and with adaptive verification (its
    request boundaries on the device). The top-logprobs request keeps its top
    two. (Without the fix: no logprobs alone, the top two beside them.)"""
    if max_chunk_logits is not None:
        monkeypatch.setattr(
            rejection_sampler, "get_max_chunk_logits", lambda _: max_chunk_logits
        )
    sampler = _sampler({r: REQUESTS[r][0] for r in order}, logprobs_mode)
    batch = _batch(order)
    logits = torch.randn(
        int(batch.cu_num_logits_np[-1]),
        VOCAB,
        generator=torch.Generator().manual_seed(1024),
    )
    out = _rejection_sampler(sampler, adaptive=adaptive)(logits, batch)

    assert out.logprobs_tensors is not None
    lists = out.logprobs_tensors.tolists()
    scores = _scores(logits, logprobs_mode)
    for batch_idx, req in enumerate(order):
        params, _, num_tokens = REQUESTS[req]
        if params.num_logprobs is None:
            continue
        lo = int(batch.cu_num_logits_np[batch_idx])
        rows = lists.slice_request(batch_idx, num_tokens)
        assert len(rows.sampled_token_ranks) == num_tokens
        for j in range(num_tokens):
            row = scores[lo + j]
            token = int(out.sampled_token_ids[batch_idx, j])
            wanted = IDS if params.logprob_token_ids else row.topk(2).indices.tolist()
            ids = [token, *wanted]
            width = len(ids)
            assert rows.logprob_token_ids[j, :width].tolist() == ids
            assert rows.logprobs[j, :width].tolist() == pytest.approx(row[ids].tolist())
            assert np.all(rows.logprobs[j, width:] == float("-inf"))
            assert rows.sampled_token_ranks[j] == int((row >= row[token]).sum())


def test_sink_alternates_carry_logprobs_for_logprob_token_ids():
    """With sink rejection, a step whose only logprobs request asks for
    ``logprob_token_ids`` scores its alternates too: the scheduler reports a
    rejected draft zero's replacement with its own logprob and rank, and
    asserts that they are there."""
    calls = []

    def alternate_logprobs_of(logits, cu_num_logits, num_sampled, alternates, **_):
        calls.append(alternates)
        num_reqs, width = alternates.shape
        return SinkAlternates(
            alternates,
            torch.zeros((num_reqs, width)),
            torch.ones((num_reqs, width), dtype=torch.int64),
        )

    sink = SimpleNamespace(alternate_logprobs_of=alternate_logprobs_of)
    sampler = _sampler({5: REQUESTS[5][0], 3: REQUESTS[3][0]}, "raw_logprobs")
    batch = _batch([5, 3])
    logits = torch.randn(int(batch.cu_num_logits_np[-1]), VOCAB)
    out = _rejection_sampler(sampler, sink)(
        logits,
        batch,
        sink_flags=torch.zeros(logits.shape[0], dtype=torch.bool),
    )
    assert len(calls) == 1
    assert out.sink_alternates is not None
    assert out.sink_alternates.logprobs is not None
    assert out.sink_alternates.ranks is not None
    assert out.logprobs_tensors is not None


@pytest.mark.parametrize(
    ("neighbour", "greedy"),
    [
        (SamplingParams(temperature=0.0), True),
        (SamplingParams(temperature=0.0, logprob_token_ids=IDS), False),
        (SamplingParams(temperature=0.0, logprobs=2), False),
    ],
    ids=["no-logprobs", "logprob-token-ids", "top-logprobs"],
)
def test_greedy_verification_only_without_logprobs(neighbour, greedy):
    """An all-greedy step verifies from the vocabulary shards only when no
    request wants logprobs, ``logprob_token_ids`` included: that path returns
    none."""
    runner = object.__new__(GPUModelRunner)
    runner.sampler = _sampler(
        {2: SamplingParams(temperature=0.0), 6: neighbour}, "raw_logprobs"
    )
    runner._greedy_verification = True
    batch = SimpleNamespace(
        num_reqs=2, num_draft_tokens=4, idx_mapping_np=np.array([2, 6])
    )
    assert runner._verifies_greedily(batch, None) is greedy


def test_tilelang_worker_modules_hold_no_live_triton_reference_kernels(monkeypatch):
    """V2 GPU worker modules whose kernels have moved to TileLang do not hold
    live Triton reference kernels, Triton dispatchers, or Triton module imports
    in their runtime globals, while lazy attribute access still resolves the
    parity-test reference kernels from ``vllm.v1.worker.gpu._triton_refs``."""
    import importlib
    import sys

    monkeypatch.delitem(sys.modules, "vllm.v1.worker.gpu._triton_refs", raising=False)
    triton_free_modules = (
        "vllm.v1.worker.gpu.block_table",
        "vllm.v1.worker.gpu.buffer_utils",
        "vllm.v1.worker.gpu.input_batch",
        "vllm.v1.worker.gpu.metrics.logits",
        "vllm.v1.worker.gpu.sample.bad_words",
        "vllm.v1.worker.gpu.sample.logit_bias",
        "vllm.v1.worker.gpu.sample.logprob",
        "vllm.v1.worker.gpu.sample.min_p",
        "vllm.v1.worker.gpu.sample.penalties",
        "vllm.v1.worker.gpu.sample.prompt_logprob",
        "vllm.v1.worker.gpu.sample.sink",
        "vllm.v1.worker.gpu.sample.thinking_budget",
        "vllm.v1.worker.gpu.spec_decode.dflash.speculator",
        "vllm.v1.worker.gpu.spec_decode.dspark.greedy",
        "vllm.v1.worker.gpu.spec_decode.greedy_verification",
        "vllm.v1.worker.gpu.spec_decode.rejection_sampler",
        "vllm.v1.worker.gpu.structured_outputs",
    )
    for mod_name in triton_free_modules:
        mod = importlib.import_module(mod_name)
        assert "triton" not in mod.__dict__, mod_name
        assert "tl" not in mod.__dict__, mod_name
        with pytest.raises(AttributeError):
            getattr(mod, "_nonexistent_triton_ref_attr")

    assert "vllm.v1.worker.gpu._triton_refs" not in sys.modules
    assert "_flatten_sampled_launcher" not in rejection_sampler.__dict__

    lazy_refs = {
        "vllm.v1.worker.gpu.block_table": (
            "_gather_block_tables_kernel",
            "_compute_slot_mappings_kernel",
        ),
        "vllm.v1.worker.gpu.boundary_checkpoint": ("prepare_boundary_capture",),
        "vllm.v1.worker.gpu.buffer_utils": ("_apply_write_kernel", "_load_ptr"),
        "vllm.v1.worker.gpu.cp_utils": ("cp_local_slot",),
        "vllm.v1.worker.gpu.input_batch": (
            "_prepare_prefill_inputs_kernel",
            "_prepare_pos_seq_lens_kernel",
            "_combine_sampled_and_draft_tokens_kernel",
            "_get_num_sampled_and_rejected_kernel",
            "_post_update_kernel",
            "_post_update_num_computed_tokens_kernel",
            "_expand_idx_mapping_kernel",
        ),
        "vllm.v1.worker.gpu.metrics.logits": ("_num_nans_kernel",),
        "vllm.v1.worker.gpu.sample.bad_words": ("_bad_words_kernel",),
        "vllm.v1.worker.gpu.sample.gumbel": (
            "_temperature_kernel",
            "_gumbel_sample_kernel",
        ),
        "vllm.v1.worker.gpu.sample.logit_bias": ("_bias_kernel",),
        "vllm.v1.worker.gpu.sample.logprob": (
            "_topk_log_softmax_kernel",
            "_ranks_kernel",
            "_fill_logprob_token_ids_kernel",
        ),
        "vllm.v1.worker.gpu.sample.min_p": ("_min_p_kernel",),
        "vllm.v1.worker.gpu.sample.penalties": (
            "_penalties_kernel",
            "_bincount_kernel",
        ),
        "vllm.v1.worker.gpu.sample.prompt_logprob": (
            "_prompt_logprobs_token_ids_kernel",
        ),
        "vllm.v1.worker.gpu.sample.sink": (
            "_sink_candidates_kernel",
            "_sink_policy_kernel",
        ),
        "vllm.v1.worker.gpu.sample.thinking_budget": (
            "_load_effective_token",
            "_update_committed_marker_cache_kernel",
            "_thinking_budget_kernel",
        ),
        "vllm.v1.worker.gpu.spec_decode.adaptive_verification": (
            "_assign_draft_token_budget",
        ),
        "vllm.v1.worker.gpu.spec_decode.dflash.speculator": (
            "_prepare_dflash_inputs_kernel",
        ),
        "vllm.v1.worker.gpu.spec_decode.dspark.greedy": (
            "_first_argmax",
            "_partial_argmax",
            "_partial_argmax_logits",
            "_finish_argmax",
            "_finish_argmax_pair",
            "_select_first_argmax",
        ),
        "vllm.v1.worker.gpu.spec_decode.greedy_verification": (
            "_greedy_rejection_kernel",
        ),
        "vllm.v1.worker.gpu.spec_decode.rejection_sampler": (
            "_flatten_sampled_kernel",
        ),
        "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils": (
            "_insert_resampled_kernel",
        ),
        "vllm.v1.worker.gpu.structured_outputs": ("_apply_grammar_bitmask_kernel",),
    }
    from vllm.v1.worker.gpu import _triton_refs

    for mod_name, names in lazy_refs.items():
        mod = importlib.import_module(mod_name)
        for name in names:
            assert name not in mod.__dict__, f"{mod_name}.{name} should be lazy"
            assert getattr(mod, name) is getattr(_triton_refs, name)
        with pytest.raises(AttributeError):
            getattr(mod, "_nonexistent_triton_ref_attr")
