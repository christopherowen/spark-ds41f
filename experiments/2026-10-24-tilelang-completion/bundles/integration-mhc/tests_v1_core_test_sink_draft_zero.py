# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sink rejection in the scheduler: the provisional token as draft zero.

A scripted worker plays the V2 runner's side of the contract
(vllm.v1.worker.gpu.sample.sink) without a model: output position ``q`` of a
request draws from a list of candidates, best first; a token in ``SUNK`` sinks
its own row, so the worker rejects it when it comes back as draft zero (and
replaces it by the next alternate of its draw) and cuts the verification at it
when the drafter proposes it. The scheduler under test is the real one, sync
and async, with its KV cache manager, prefix cache and stop checks.

The collapse case is ``Say hi.`` at temperature 0: the greedy reply reaches
``How can`` with `` can`` as a sink (DeepSeek V4.1 Flash, the 1017 checkpoint);
the next token after it would be document-start noise. Here `` can`` is
``CAN`` and its draw's next candidate ``I_``.
"""

import functools
import tempfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm.entrypoints.generate.base.serving import build_sink_usage
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import (
    RequestOutputKind,
    SamplingParams,
    StructuredOutputsParams,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.utils import stops_on_next_token
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.metrics.stats import RequestSinkMetrics
from vllm.v1.outputs import (
    DraftTokenIds,
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.sample.sink import SinkConfig, SinkEvent, SinkOutcome, SinkRow

from .utils import EOS_TOKEN_ID, create_scheduler

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16
HOW, CAN, I_, HELP, YOU, TODAY, QM = 101, 7, 102, 103, 104, 105, 106
# Tokens whose own row the detector flags.
SUNK = frozenset({CAN, 9})


@pytest.fixture(autouse=True)
def sink_rejection_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # The test model has no entry in SINK_CHANNELS: name its column and
    # threshold, and switch rejection on.
    monkeypatch.setenv("VLLM_SINK_DETECTOR_COLUMN", "2455")
    monkeypatch.setenv("VLLM_SINK_DETECTOR_THRESHOLD", "85000")
    monkeypatch.setenv("VLLM_SINK_REJECTION", "1")
    monkeypatch.setenv("VLLM_SINK_REJECTION_MAX_RETRIES", "3")


def logprob_of(rank: int) -> float:
    """The scripted distribution: candidate ``rank`` (1-based) has
    probability 2**-rank."""
    return -0.6931471805599453 * rank


@dataclass
class WorkerRequest:
    committed: list[int] = field(default_factory=list)
    provisional: int | None = None
    # The draw of the provisional token: its candidates after it (and their
    # ranks), and how many of them replaced it already.
    alternates: list[int] = field(default_factory=list)
    alternate_ranks: list[int] = field(default_factory=list)
    tries: int = 0
    drafts: list[int] = field(default_factory=list)


# One logprobs row per sampled token: (token, its rank, the draw's best).
Row = tuple[int, int, int]


def scripted_prompt_logprobs(
    prompt_token_ids: list[int], positions: list[int]
) -> LogprobsTensors:
    """The scripted prompt logprobs: row ``p`` scores prompt token ``p + 1``,
    at logprob ``-(p + 1) / 64`` and rank 1 (it is its own best)."""
    tokens = torch.tensor([prompt_token_ids[p + 1] for p in positions])
    values = torch.tensor([-(p + 1) / 64 for p in positions])
    return LogprobsTensors(
        logprob_token_ids=torch.stack([tokens, tokens], dim=1),
        logprobs=torch.stack([values, values], dim=1),
        selected_token_ranks=torch.ones(len(positions), dtype=torch.int64),
    )


class ScriptedWorker:
    """The V2 runner's side of draft zero, with a perfect drafter unless
    ``draft_errors`` says otherwise (``(req_id, position) -> token``)."""

    def __init__(
        self,
        scheduler,
        scripts: dict[str, list[list[int]]],
        num_drafts: int,
        *,
        max_retries: int = 3,
        draft_errors: dict[tuple[str, int], int] | None = None,
        no_drafts: frozenset[str] = frozenset(),
        logprobs: bool = False,
    ):
        self.scheduler = scheduler
        self.scripts = scripts
        self.num_drafts = num_drafts
        self.max_retries = max_retries
        self.draft_errors = draft_errors or {}
        self.no_drafts = no_drafts
        self.logprobs = logprobs
        self.requests: dict[str, WorkerRequest] = {}
        # (req_id, output position, token) of every draw, in order.
        self.draws: list[tuple[str, int, int]] = []
        # Per scheduler output (by id) and request: the drafter tokens the
        # step's inputs held.
        self.consumed_drafts: dict[int, dict[str, list[int]]] = {}
        # Prompt logprobs, as the V2 runner's PromptLogprobsWorker: the prompt
        # positions scored so far, until the step that finishes the prompt
        # returns them all.
        self.prompt_positions: dict[str, list[int]] = {}

    def candidates(self, req_id: str, position: int) -> list[int]:
        return self.scripts[req_id][position]

    def _draw(self, st: WorkerRequest, req_id: str, position: int, exclude=()) -> Row:
        everything = self.candidates(req_id, position)
        candidates = [c for c in everything if c not in exclude]
        ranks = [everything.index(c) + 1 for c in candidates]
        st.provisional = candidates[0]
        st.alternates = candidates[1 : 1 + self.max_retries]
        st.alternate_ranks = ranks[1 : 1 + self.max_retries]
        st.tries = 0
        self.draws.append((req_id, position, candidates[0]))
        return candidates[0], ranks[0], everything[0]

    def _propose(self, req_id: str, st: WorkerRequest) -> None:
        if req_id in self.no_drafts:
            st.drafts = []
            return
        start = len(st.committed) + 1
        st.drafts = [
            self.draft_errors.get(
                (req_id, start + j), self.candidates(req_id, start + j)[0]
            )
            for j in range(self.num_drafts)
        ]

    def execute(self, so: SchedulerOutput) -> ModelRunnerOutput:
        for req_id in (so.preempted_req_ids or set()) | so.finished_req_ids:
            self.requests.pop(req_id, None)
            self.prompt_positions.pop(req_id, None)
        for new_req in so.scheduled_new_reqs:
            params = new_req.sampling_params
            assert params is not None and new_req.prompt_token_ids is not None
            assert new_req.prefill_token_ids is not None
            # A request resumed after its prompt had its prompt logprobs
            # computed before the preemption.
            if params.prompt_logprobs is not None and len(
                new_req.prefill_token_ids
            ) == len(new_req.prompt_token_ids):
                self.prompt_positions[new_req.req_id] = []
        prompt_logprobs: dict[str, LogprobsTensors] = {}
        draft_zero = so.sink_draft_zero_req_ids or set()
        req_ids = list(so.num_scheduled_tokens)
        sampled: list[list[int]] = []
        events: dict[str, SinkEvent] = {}
        rows: list[list[Row]] = []
        consumed: dict[str, list[int]] = {}
        for req_id in req_ids:
            request = self.scheduler.requests[req_id]
            spec = so.scheduled_spec_decode_tokens.get(req_id, [])
            event = None
            positions = self.prompt_positions.get(req_id)
            if positions is not None:
                end = request.num_computed_tokens
                start = end - so.num_scheduled_tokens[req_id]
                positions.extend(range(start, min(end, request.num_prompt_tokens - 1)))
                if not request.is_prefill_chunk:
                    assert request.prompt_token_ids is not None
                    prompt_logprobs[req_id] = scripted_prompt_logprobs(
                        request.prompt_token_ids, positions
                    )
                    del self.prompt_positions[req_id]
            if req_id in draft_zero:
                st = self.requests[req_id]
                consumed[req_id] = st.drafts[: len(spec) - 1]
                out, step_rows, event = self._verify(req_id, st, spec)
            elif request.is_prefill_chunk:
                out, step_rows = [], []
            else:
                # The step finishes the (re)prefill: its last row draws the
                # request's first provisional token.
                st = self.requests[req_id] = WorkerRequest(
                    committed=list(request.output_token_ids)
                )
                row = self._draw(st, req_id, len(st.committed))
                out, step_rows = [row[0]], [row]
            sampled.append(out)
            rows.append(step_rows)
            if event is not None:
                events[req_id] = event
            if out:
                self._propose(req_id, self.requests[req_id])
        self.consumed_drafts[id(so)] = consumed
        logprobs = None
        if self.logprobs:
            flat = [row for step_rows in rows for row in step_rows]
            logprobs = LogprobsLists(
                np.array([[t, best] for t, _, best in flat], dtype=np.int64).reshape(
                    -1, 2
                ),
                np.array(
                    [[logprob_of(rank), logprob_of(1)] for _, rank, _ in flat],
                    dtype=np.float32,
                ).reshape(-1, 2),
                np.array([rank for _, rank, _ in flat], dtype=np.int64),
                np.cumsum([0] + [len(r) for r in rows]).tolist(),
            )
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=sampled,
            logprobs=logprobs,
            prompt_logprobs_dict=prompt_logprobs,
            pooler_output=[],
            sink_events=events or None,
        )

    def _verify(self, req_id: str, st: WorkerRequest, spec: list[int]):
        draft_zero = st.provisional
        assert draft_zero is not None
        if spec[0] != -1:
            # Synchronous scheduling: the scheduler holds the same token.
            assert spec[0] == draft_zero, (req_id, spec, draft_zero)
        position = len(st.committed)
        if (
            draft_zero in SUNK
            and st.tries < self.max_retries
            and st.tries < len(st.alternates)
        ):
            alternate = st.alternates[st.tries]
            rank = st.alternate_ranks[st.tries]
            st.tries += 1
            st.provisional = alternate
            event = SinkEvent(
                SinkOutcome.REJECTED,
                SinkRow.DRAFT_ZERO,
                0,
                -1,
                recovered_logprob=logprob_of(rank),
                recovered_rank=rank,
            )
            # The step's own row for it is the sunk row's: the scheduler
            # must not use it.
            return [alternate], [(alternate, 99, -1)], event
        outcome = SinkOutcome.FALLTHROUGH if draft_zero in SUNK else SinkOutcome.NONE
        st.committed.append(draft_zero)
        drafts = st.drafts[: len(spec) - 1]
        out = [draft_zero]
        step_rows: list[Row] = []
        cut_row = -1
        for j, draft in enumerate(drafts, 1):
            q = position + j
            best = self.candidates(req_id, q)[0]
            if draft == best and draft not in SUNK:
                out.append(draft)
                step_rows.append((draft, 1, best))
                st.committed.append(draft)
                continue
            if draft == best:
                # The draft's own row is sunk: cut, recover without it.
                cut_row = j
                row = self._draw(st, req_id, q, exclude=(draft,))
            else:
                row = self._draw(st, req_id, q)
            break
        else:
            row = self._draw(st, req_id, position + len(drafts) + 1)
        out.append(row[0])
        step_rows.append(row)
        event = None
        if outcome != SinkOutcome.NONE or cut_row >= 0:
            event = SinkEvent(outcome, SinkRow.DRAFT_ZERO, int(cut_row >= 0), cut_row)
        return out, step_rows, event


class ScriptedGrammar:
    """A grammar over token ids: ``allowed[i]`` is the set of tokens allowed
    at accepted-token index ``i``. Records every accepted token and the state
    of every mask it fills."""

    def __init__(self, allowed: list[set[int]], vocab_size: int):
        self.allowed = allowed
        self.vocab_size = vocab_size
        self.state = 0
        self.accepted: list[int] = []
        self.filled: list[int] = []

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        for token in tokens:
            if token not in self.allowed[self.state]:
                return False
            self.state += 1
            self.accepted.append(token)
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        valid = []
        for i, token in enumerate(tokens):
            if token not in self.allowed[self.state + i]:
                break
            valid.append(token)
        return valid

    def rollback(self, num_tokens: int) -> None:
        self.state -= num_tokens
        del self.accepted[len(self.accepted) - num_tokens :]

    def fill_bitmask(self, bitmask: torch.Tensor, batch_index: int) -> None:
        self.filled.append(self.state)
        row = torch.zeros(self.vocab_size, dtype=torch.bool)
        row[list(self.allowed[self.state])] = True
        words = row.view(-1, 32).to(torch.int64)
        packed = (words << torch.arange(32)).sum(dim=1)
        bitmask[batch_index] = packed.to(torch.int64).view(-1).to(torch.int32)

    def is_terminated(self) -> bool:
        return self.state >= len(self.allowed)

    def reset(self) -> None:
        self.state = 0


class ScriptedGrammarBackend:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor:
        return torch.full((max_num_seqs, self.vocab_size // 32), -1, dtype=torch.int32)


def _make_request(
    req_id: str,
    *,
    max_tokens: int = 32,
    logprobs: int | None = None,
    prompt_logprobs: int | None = None,
    stop_token_ids: list[int] | None = None,
    prompt_len: int = 20,
    structured: bool = False,
) -> Request:
    init_none_hash(sha256)
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        logprobs=logprobs,
        prompt_logprobs=prompt_logprobs,
        stop_token_ids=stop_token_ids,
        temperature=0.0,
        structured_outputs=StructuredOutputsParams(regex=".*") if structured else None,
    )
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    return Request(
        request_id=req_id,
        prompt_token_ids=[ord(req_id[0]) % 50 + i for i in range(prompt_len)],
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def _script(*greedy: int, length: int = 64) -> list[list[int]]:
    """Candidates per position: the given greedy tokens (then filler), each
    followed by two runner-ups; `` can`` is followed by `` I``."""
    tokens = list(greedy) + [200 + i for i in range(length - len(greedy))]
    return [
        [CAN, I_, 300 + i] if token == CAN else [token, 400 + i, 500 + i]
        for i, token in enumerate(tokens)
    ]


def expected_reply(script: list[list[int]], length: int) -> list[int]:
    """Position by position, the first candidate that is not a sink."""
    return [next(c for c in script[q] if c not in SUNK) for q in range(length)]


@dataclass
class Run:
    emitted: dict[str, list[int]] = field(default_factory=dict)
    outputs: dict[str, list[EngineCoreOutput]] = field(default_factory=dict)
    finished: dict[str, EngineCoreOutput] = field(default_factory=dict)
    steps: int = 0


def drive(
    scheduler,
    worker: ScriptedWorker,
    *,
    async_scheduling: bool,
    max_steps: int = 400,
    on_output=None,
    on_schedule=None,
) -> Run:
    """The engine core loop: synchronous (schedule, execute, update, take the
    drafts), or asynchronous with one step in flight (the worker runs each
    step as it is scheduled, the scheduler sees its output one step later)."""
    run = Run()
    queue: deque[tuple[SchedulerOutput, ModelRunnerOutput]] = deque()

    def process(so: SchedulerOutput, mro: ModelRunnerOutput) -> None:
        engine_outputs = scheduler.update_from_output(so, mro)
        for eco in engine_outputs.values():
            for out in eco.outputs:
                run.emitted.setdefault(out.request_id, []).extend(out.new_token_ids)
                run.outputs.setdefault(out.request_id, []).append(out)
                if out.finish_reason is not None:
                    run.finished[out.request_id] = out
        if not async_scheduling:
            req_ids = [r for r in mro.req_ids if r in worker.requests]
            scheduler.update_draft_token_ids(
                DraftTokenIds(
                    req_ids, [list(worker.requests[r].drafts) for r in req_ids]
                )
            )
        for request in scheduler.running:
            assert request.num_output_placeholders >= 0
            if not async_scheduling and not request.is_prefill_chunk:
                # Every committed token is computed; the held one is not.
                assert request.num_computed_tokens == request.num_tokens, (
                    request.request_id
                )
        if on_output is not None:
            on_output(so, mro)

    while run.steps < max_steps:
        run.steps += 1
        so = scheduler.schedule()
        if on_schedule is not None:
            on_schedule(so)
        if so.total_num_scheduled_tokens > 0:
            queue.append((so, worker.execute(so)))
            if async_scheduling and len(queue) < 2:
                continue
        if not queue:
            if not scheduler.has_requests():
                break
            continue
        process(*queue.popleft())
    assert not queue and not scheduler.has_requests()
    return run


@functools.cache
def _model_dir() -> str:
    """A local OPT config: the scheduler needs no download or tokenizer."""
    path = Path(tempfile.mkdtemp(prefix="sink-draft-zero-"))
    (path / "config.json").write_text(
        '{"architectures":["OPTForCausalLM"],"model_type":"opt",'
        '"max_position_embeddings":2048}'
    )
    return str(path)


def _scheduler(async_scheduling: bool, num_drafts: int | None, **kwargs):
    scheduler = create_scheduler(
        model=_model_dir(),
        skip_tokenizer_init=True,
        device="cpu",
        async_scheduling=async_scheduling,
        num_speculative_tokens=num_drafts,
        # The scheduler does not look at the drafter; async scheduling only
        # takes some kinds of it.
        speculative_method="ngram_gpu" if num_drafts is not None else None,
        use_v2_model_runner=True,
        block_size=BLOCK_SIZE,
        enable_prefix_caching=True,
        **kwargs,
    )
    assert scheduler.sink_rejection
    return scheduler


SAY_HI = _script(HOW, CAN, I_, HELP, YOU, TODAY, QM)


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("num_drafts", [None, 1, 5])
def test_collapse_sequence_draft_zero_rejected(async_scheduling, num_drafts):
    """`` can`` is sampled as the last token of a step (the drafter missed
    it), held back, flagged as draft zero and replaced by its draw's next
    candidate; `` can`` is never emitted and nothing was committed after it."""
    scheduler = _scheduler(async_scheduling, num_drafts)
    request = _make_request("a", max_tokens=12)
    scheduler.add_request(request)
    worker = ScriptedWorker(
        scheduler, {"a": SAY_HI}, num_drafts or 0, draft_errors={("a", 1): 999}
    )
    run = drive(scheduler, worker, async_scheduling=async_scheduling)
    assert run.emitted["a"] == expected_reply(SAY_HI, 12)
    metrics = run.finished["a"].sink_metrics
    assert metrics == RequestSinkMetrics(
        num_flagged_draft_zero=1, num_draft_zero_rejections=1
    )
    # The prefix cache hashed the committed tokens, and only them.
    committed = Request(
        request_id="committed",
        prompt_token_ids=list(request.all_token_ids),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )
    assert request.block_hashes == committed.block_hashes


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_collapse_sequence_drafter_cut(async_scheduling):
    """The drafter proposes `` can`` behind a clean draft zero: the
    verification is cut at it and its position recovered without it."""
    scheduler = _scheduler(async_scheduling, 3)
    scheduler.add_request(_make_request("a", max_tokens=12))
    worker = ScriptedWorker(scheduler, {"a": SAY_HI}, 3)
    run = drive(scheduler, worker, async_scheduling=async_scheduling)
    assert run.emitted["a"] == expected_reply(SAY_HI, 12)
    assert run.finished["a"].sink_metrics == RequestSinkMetrics(
        num_flagged_drafter=1, num_draft_cuts=1
    )


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_alternates_used_up_falls_through(async_scheduling):
    """A position whose draw has only sinks left commits the sunk token as it
    is and counts the fallthrough."""
    script = _script(HOW, CAN)
    script[1] = [CAN, 9, 400, 401]
    scheduler = _scheduler(async_scheduling, None)
    scheduler.add_request(_make_request("a", max_tokens=6))
    worker = ScriptedWorker(scheduler, {"a": script}, 0, max_retries=1)
    run = drive(scheduler, worker, async_scheduling=async_scheduling)
    # CAN is replaced by 9, also a sink: past one retry, 9 stands.
    assert run.emitted["a"][:2] == [HOW, 9]
    assert run.finished["a"].sink_metrics == RequestSinkMetrics(
        num_flagged_draft_zero=2, num_draft_zero_rejections=1, num_fallthroughs=1
    )


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_neighbours_unchanged_by_a_rejection(async_scheduling):
    """Three requests in one batch; the middle one has its draft zero
    rejected. Its neighbours emit what they emit when nothing sinks."""

    def run_batch(with_sink: bool) -> Run:
        scheduler = _scheduler(async_scheduling, 3)
        scripts = {
            "a": _script(11, 12, 13, 14),
            "b": _script(HOW, CAN if with_sink else I_, I_, HELP),
            "c": _script(21, 22, 23, 24),
        }
        for req_id in scripts:
            scheduler.add_request(_make_request(req_id, max_tokens=10))
        worker = ScriptedWorker(scheduler, scripts, 3, draft_errors={("b", 1): 999})
        return drive(scheduler, worker, async_scheduling=async_scheduling)

    clean = run_batch(False)
    sunk = run_batch(True)
    for req_id in "abc":
        assert sunk.emitted[req_id] == clean.emitted[req_id]
    assert sunk.finished["b"].sink_metrics.num_draft_zero_rejections == 1
    assert sunk.finished["a"].sink_metrics.num_flagged_rows == 0
    assert sunk.finished["c"].sink_metrics.num_flagged_rows == 0


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_request_with_zero_drafter_tokens(async_scheduling):
    """A request the drafter gives nothing still verifies its draft zero, in
    a batch whose other request carries drafts."""
    scheduler = _scheduler(async_scheduling, 3)
    scripts = {"a": SAY_HI, "b": _script(31, 32, 33, 34)}
    for req_id in scripts:
        scheduler.add_request(_make_request(req_id, max_tokens=8))
    seen_spec: list[list[int]] = []

    def record(so: SchedulerOutput) -> None:
        if "a" in (so.sink_draft_zero_req_ids or ()):
            seen_spec.append(so.scheduled_spec_decode_tokens["a"])

    worker = ScriptedWorker(scheduler, scripts, 3, no_drafts=frozenset({"a"}))
    run = drive(
        scheduler, worker, async_scheduling=async_scheduling, on_schedule=record
    )
    assert run.emitted["a"] == expected_reply(SAY_HI, 8)
    assert run.emitted["b"] == expected_reply(scripts["b"], 8)
    assert run.finished["a"].sink_metrics.num_draft_zero_rejections == 1
    assert seen_spec
    if not async_scheduling:
        # Draft zero alone.
        assert all(len(spec) == 1 for spec in seen_spec)


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_accounting_through_a_rejection(async_scheduling):
    """A rejected draft zero rolls its whole step back (its own slot
    included): what is computed past the committed tokens is exactly what
    is in flight, and once nothing is, every committed token is computed and
    no placeholder is left."""
    scheduler = _scheduler(async_scheduling, 2)
    request = _make_request("a", max_tokens=10)
    scheduler.add_request(request)
    worker = ScriptedWorker(scheduler, {"a": SAY_HI}, 2, draft_errors={("a", 1): 999})
    snapshots = []

    def record(so: SchedulerOutput, mro: ModelRunnerOutput) -> None:
        snapshots.append(
            (
                "a" in (mro.sink_events or {}),
                request.num_computed_tokens - request.num_tokens,
                request.num_in_flight_tokens,
                request.num_output_placeholders,
            )
        )

    run = drive(scheduler, worker, async_scheduling=async_scheduling, on_output=record)
    assert run.emitted["a"] == expected_reply(SAY_HI, 10)
    assert any(rejected for rejected, *_ in snapshots)
    for _, past_committed, in_flight, placeholders in snapshots[:-1]:
        assert past_committed == in_flight
        assert placeholders == (in_flight if async_scheduling else 0)


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("stop", ["eos", "stop_token"])
def test_provisional_stop_token_commits_unverified(async_scheduling, stop):
    """A sampled token that ends the request (EOS, a stop token) is
    committed when it is sampled: no step follows to verify it."""
    stop_token = EOS_TOKEN_ID if stop == "eos" else 77
    scheduler = _scheduler(async_scheduling, 2)
    request = _make_request(
        "a", max_tokens=20, stop_token_ids=[77] if stop == "stop_token" else None
    )
    scheduler.add_request(request)
    script = _script(HOW, I_, stop_token)
    worker = ScriptedWorker(scheduler, {"a": script}, 2, draft_errors={("a", 2): 1})
    run = drive(scheduler, worker, async_scheduling=async_scheduling)
    assert run.emitted["a"] == [HOW, I_, stop_token]
    assert request.status == RequestStatus.FINISHED_STOPPED
    draw = worker.draws.index(("a", 2, stop_token))
    # Synchronous scheduling schedules nothing after it; asynchronous already
    # had the next step in flight, whose output is discarded.
    assert len(worker.draws) - 1 - draw == (1 if async_scheduling else 0)


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("max_tokens", [1, 2, 4])
def test_max_tokens_token_commits_unverified(async_scheduling, max_tokens):
    scheduler = _scheduler(async_scheduling, 2)
    request = _make_request("a", max_tokens=max_tokens)
    scheduler.add_request(request)
    scheduled_steps = []
    worker = ScriptedWorker(scheduler, {"a": SAY_HI}, 2)
    run = drive(
        scheduler,
        worker,
        async_scheduling=async_scheduling,
        on_schedule=lambda so: scheduled_steps.append(so.total_num_scheduled_tokens),
    )
    assert run.emitted["a"] == expected_reply(SAY_HI, max_tokens)
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    if max_tokens == 1:
        # Only the prefill: its token is the max_tokens token.
        assert [n for n in scheduled_steps if n] == [request.num_prompt_tokens]


def test_stops_on_next_token_mirrors_check_stop():
    request = _make_request("a", max_tokens=3, stop_token_ids=[77])
    assert stops_on_next_token(request, EOS_TOKEN_ID, 1000)
    assert stops_on_next_token(request, 77, 1000)
    assert not stops_on_next_token(request, 5, 1000)
    request.append_output_token_ids([5, 6])
    assert stops_on_next_token(request, 5, 1000)
    # max_model_len 21: the prompt (20) plus one token.
    assert stops_on_next_token(_make_request("b"), 5, 21)


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_abort_while_provisional(async_scheduling):
    """An abort drops the held token: it was never emitted."""
    scheduler = _scheduler(async_scheduling, 2)
    request = _make_request("a", max_tokens=20)
    scheduler.add_request(request)
    worker = ScriptedWorker(scheduler, {"a": SAY_HI}, 2)
    held: list[int] = []

    def abort_when_holding(_so, _mro) -> None:
        if not held and len(request.output_token_ids) >= 3:
            assert request.sink_provisional is not None
            held.append(request.sink_provisional)
            scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)

    run = drive(
        scheduler,
        worker,
        async_scheduling=async_scheduling,
        on_output=abort_when_holding,
    )
    assert held
    assert run.emitted["a"] == list(request.output_token_ids)
    reply = expected_reply(SAY_HI, 20)
    # The held token is the next one of the reply, never emitted.
    assert held[0] == reply[len(run.emitted["a"])]
    assert request.status == RequestStatus.FINISHED_ABORTED


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_logprobs_of_committed_provisional_tokens(async_scheduling):
    """A held token carries the logprobs of the step that drew it until it is
    committed; a rejected one's replacement carries its own logprob and rank
    in that same draw (not the sunk row's)."""
    scheduler = _scheduler(async_scheduling, 2)
    request = _make_request("a", max_tokens=8, logprobs=1)
    scheduler.add_request(request)
    worker = ScriptedWorker(
        scheduler, {"a": SAY_HI}, 2, draft_errors={("a", 1): 999}, logprobs=True
    )
    run = drive(scheduler, worker, async_scheduling=async_scheduling)
    tokens = run.emitted["a"]
    assert tokens == expected_reply(SAY_HI, 8)
    rows = []
    for out in run.outputs["a"]:
        if out.new_token_ids:
            lp = out.new_logprobs
            assert lp is not None and len(lp.sampled_token_ranks) == len(
                out.new_token_ids
            )
            rows.extend(
                zip(
                    lp.logprob_token_ids[:, 0].tolist(),
                    lp.logprobs[:, 0].tolist(),
                    lp.sampled_token_ranks.tolist(),
                    lp.logprob_token_ids[:, 1].tolist(),
                )
            )
    assert [token for token, *_ in rows] == tokens
    # `` I`` replaced `` can`` at position 1: rank 2 of `` can``'s draw, whose
    # best (top-1 column) is `` can``.
    assert rows[1] == (I_, pytest.approx(logprob_of(2)), 2, CAN)
    assert all(rank == 1 for i, (_, _, rank, _) in enumerate(rows) if i != 1)


def _assert_prompt_logprobs_once(run: Run, request: Request) -> None:
    """The request's first output carries its prompt logprobs, the worker's
    for every prompt position, and no other output carries any."""
    first, *rest = run.outputs[request.request_id]
    assert first.new_token_ids
    assert all(out.new_prompt_logprobs_tensors is None for out in rest)
    assert request.prompt_token_ids is not None
    expected = scripted_prompt_logprobs(
        request.prompt_token_ids, list(range(request.num_prompt_tokens - 1))
    )
    got = first.new_prompt_logprobs_tensors
    assert got is not None
    assert torch.equal(got.logprob_token_ids, expected.logprob_token_ids)
    assert torch.equal(got.logprobs, expected.logprobs)
    assert torch.equal(got.selected_token_ranks, expected.selected_token_ranks)


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("num_drafts", [None, 2])
@pytest.mark.parametrize("max_num_batched_tokens", [8192, 16])
def test_prompt_logprobs_go_out_with_the_held_first_token(
    async_scheduling, num_drafts, max_num_batched_tokens
):
    """The step that finishes a prompt samples the request's first token, and
    sink rejection holds it back: that step has no output for the request.
    The prompt logprobs it returns go out with the token, in the request's
    first output (the engine core died on them). Its tokens, and those of the
    request beside it, are as without prompt logprobs."""
    scripts = {"a": SAY_HI, "b": _script(HOW, I_, HELP, YOU)}

    def serve(prompt_logprobs: int | None) -> tuple[Request, Run]:
        scheduler = _scheduler(
            async_scheduling,
            num_drafts,
            max_num_batched_tokens=max_num_batched_tokens,
        )
        request = _make_request(
            "a", max_tokens=10, prompt_len=40, prompt_logprobs=prompt_logprobs
        )
        scheduler.add_request(request)
        scheduler.add_request(_make_request("b", max_tokens=10))
        worker = ScriptedWorker(
            scheduler, scripts, num_drafts or 0, draft_errors={("a", 1): 999}
        )
        return request, drive(scheduler, worker, async_scheduling=async_scheduling)

    request, run = serve(prompt_logprobs=1)
    _, plain = serve(prompt_logprobs=None)
    assert run.emitted == plain.emitted
    assert run.emitted["a"] == expected_reply(SAY_HI, 10)
    _assert_prompt_logprobs_once(run, request)
    assert all(out.new_prompt_logprobs_tensors is None for out in run.outputs["b"])


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize(
    "output_kind", [RequestOutputKind.FINAL_ONLY, RequestOutputKind.DELTA]
)
def test_prompt_logprobs_reach_the_response(async_scheduling, output_kind):
    """Through the frontend's output processor: one response carries the
    prompt logprobs, a logprob for every prompt token after the first."""
    scheduler = _scheduler(async_scheduling, 2)
    request = _make_request("a", max_tokens=8, prompt_logprobs=1)
    scheduler.add_request(request)
    worker = ScriptedWorker(scheduler, {"a": SAY_HI}, 2, draft_errors={("a", 1): 999})
    run = drive(scheduler, worker, async_scheduling=async_scheduling)

    processor = OutputProcessor(None, log_stats=False)
    processor.add_request(
        EngineCoreRequest(
            request_id="a",
            external_req_id="a",
            prompt_token_ids=list(request.prompt_token_ids or ()),
            mm_features=None,
            sampling_params=SamplingParams(
                max_tokens=8,
                prompt_logprobs=1,
                output_kind=output_kind,
                detokenize=False,
            ),
            pooling_params=None,
            arrival_time=0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        ),
        None,
    )
    tokens: list[int] = []
    prompt_logprobs = []
    for out in run.outputs["a"]:
        for request_output in processor.process_outputs(
            [out], engine_core_timestamp=0
        ).request_outputs:
            tokens.extend(request_output.outputs[0].token_ids)
            if request_output.prompt_logprobs:
                prompt_logprobs.append(request_output.prompt_logprobs)
    assert tokens == expected_reply(SAY_HI, 8)
    assert len(prompt_logprobs) == 1
    prompt = request.prompt_token_ids
    assert prompt is not None and len(prompt_logprobs[0]) == len(prompt)
    assert prompt_logprobs[0][0] is None
    for p, position in enumerate(prompt_logprobs[0][1:], 1):
        assert position is not None and list(position) == [prompt[p]]
        assert position[prompt[p]].logprob == pytest.approx(-p / 64)
        assert position[prompt[p]].rank == 1


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_prompt_logprobs_held_across_a_preemption(async_scheduling):
    """A request preempted while its first token is held has committed
    nothing, so its resumed prompt returns its logprobs again: those replace
    the ones held, and go out once, with the first token."""
    scheduler = _scheduler(async_scheduling, 2)
    request = _make_request("a", max_tokens=8, prompt_logprobs=1)
    scheduler.add_request(request)
    worker = ScriptedWorker(scheduler, {"a": SAY_HI}, 2, draft_errors={("a", 1): 999})
    returned: list[LogprobsTensors] = []

    def on_output(so: SchedulerOutput, mro: ModelRunnerOutput) -> None:
        if "a" not in mro.prompt_logprobs_dict:
            return
        returned.append(mro.prompt_logprobs_dict["a"])
        if len(returned) == 1:
            assert request.sink_prompt_logprobs is returned[0]
            assert not request.output_token_ids
            scheduler.reset_prefix_cache(reset_running_requests=True)

    run = drive(
        scheduler, worker, async_scheduling=async_scheduling, on_output=on_output
    )
    assert request.num_preemptions == 1 and len(returned) == 2
    assert run.emitted["a"] == expected_reply(SAY_HI, 8)
    _assert_prompt_logprobs_once(run, request)
    assert run.outputs["a"][0].new_prompt_logprobs_tensors is returned[1]


def test_prompt_logprobs_once_across_preemptions():
    """Async scheduling under KV pressure, both requests with prompt
    logprobs: each request's go out once, with its first token, and a request
    resumed after it committed tokens does not compute them again."""
    scheduler = _scheduler(True, 2, num_blocks=7)
    scripts = {"a": SAY_HI, "b": _script(HOW, I_, HELP, YOU)}
    requests = {r: _make_request(r, max_tokens=40, prompt_logprobs=1) for r in scripts}
    for request in requests.values():
        scheduler.add_request(request)
    worker = ScriptedWorker(scheduler, scripts, 2, draft_errors={("a", 1): 999})
    run = drive(scheduler, worker, async_scheduling=True)
    assert any(request.num_preemptions for request in requests.values())
    for req_id, script in scripts.items():
        assert run.emitted[req_id] == expected_reply(script, 40)
        _assert_prompt_logprobs_once(run, requests[req_id])


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_two_sequences_of_one_request(async_scheduling):
    """n > 1: two sequences of one request run as two engine requests, each
    with its own held token; the response usage sums their events."""
    scheduler = _scheduler(async_scheduling, 2)
    scripts = {"0_parent": SAY_HI, "1_parent": _script(HOW, I_, HELP)}
    for req_id in scripts:
        scheduler.add_request(_make_request(req_id, max_tokens=6))
    worker = ScriptedWorker(scheduler, scripts, 2, draft_errors={("0_parent", 1): 999})
    run = drive(scheduler, worker, async_scheduling=async_scheduling)
    for req_id, script in scripts.items():
        assert run.emitted[req_id] == expected_reply(script, 6)
    usage = build_sink_usage(
        RequestOutput(
            request_id="parent",
            prompt=None,
            prompt_token_ids=[1],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=i,
                    text="",
                    token_ids=run.emitted[req_id],
                    cumulative_logprob=None,
                    logprobs=None,
                    sink_metrics=run.finished[req_id].sink_metrics,
                )
                for i, req_id in enumerate(scripts)
            ],
            finished=True,
        )
    )
    assert usage is not None
    assert usage.draft_zero_rejections == 1
    assert usage.flagged_draft_zero == usage.flagged_rows == 1


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("stream_interval", [1, 3])
def test_streamed_tokens_with_stream_interval(async_scheduling, stream_interval):
    """The frontend streams the committed tokens as they come, held tokens
    never; with a stream interval above one, every delta but the first and
    the last holds at least that many tokens."""
    scheduler = _scheduler(async_scheduling, 2)
    request = _make_request("a", max_tokens=12)
    scheduler.add_request(request)
    worker = ScriptedWorker(scheduler, {"a": SAY_HI}, 2, draft_errors={("a", 1): 999})
    run = drive(scheduler, worker, async_scheduling=async_scheduling)

    processor = OutputProcessor(None, log_stats=False, stream_interval=stream_interval)
    processor.add_request(
        EngineCoreRequest(
            request_id="a",
            external_req_id="a",
            prompt_token_ids=list(request.prompt_token_ids or ()),
            mm_features=None,
            sampling_params=SamplingParams(
                max_tokens=12, output_kind=RequestOutputKind.DELTA, detokenize=False
            ),
            pooling_params=None,
            arrival_time=0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        ),
        None,
    )
    deltas = []
    for out in run.outputs["a"]:
        for request_output in processor.process_outputs(
            [out], engine_core_timestamp=0
        ).request_outputs:
            deltas.append(list(request_output.outputs[0].token_ids))
            if request_output.finished:
                assert request_output.outputs[0].sink_metrics is not None
    assert [t for delta in deltas for t in delta] == expected_reply(SAY_HI, 12)
    if stream_interval > 1:
        assert all(len(delta) >= stream_interval for delta in deltas[1:-1])


def test_grammar_sees_committed_tokens_only():
    """Structured output across a rejection at draft zero: the masks of a
    draft-zero step start after draft zero (one per spec token), the grammar
    accepts committed tokens only, and nothing is ever rolled back from it."""
    vocab_size = 1024
    scheduler = _scheduler(False, 2)
    script = SAY_HI
    allowed = [set(candidates) for candidates in script]
    grammar = ScriptedGrammar(allowed, vocab_size)
    scheduler.structured_output_manager.backend = ScriptedGrammarBackend(vocab_size)
    request = _make_request("a", max_tokens=10, structured=True)
    assert request.structured_output_request is not None
    request.structured_output_request.grammar = grammar
    scheduler.add_request(request)
    worker = ScriptedWorker(scheduler, {"a": script}, 2, draft_errors={("a", 1): 999})
    masks: list[tuple[int, int, int]] = []

    def build_masks(so: SchedulerOutput) -> None:
        start = len(grammar.filled)
        committed = grammar.state
        output = scheduler.get_grammar_bitmask(so)
        if output is None:
            return
        spec = so.scheduled_spec_decode_tokens.get("a", [])
        rows = output.grammar_bitmask.shape[0]
        if "a" in (so.sink_draft_zero_req_ids or ()):
            # One mask per spec token, the first after draft zero.
            assert rows == len(spec)
            assert grammar.filled[start:] == [
                committed + 1 + i for i in range(len(spec))
            ]
        else:
            assert rows == len(spec) + 1
        masks.append((committed, rows, len(spec)))

    run = drive(scheduler, worker, async_scheduling=False, on_schedule=build_masks)
    assert run.emitted["a"] == expected_reply(script, 10)
    assert grammar.accepted == run.emitted["a"]
    assert CAN not in grammar.accepted
    assert masks


def test_async_backfill_puts_draft_zero_first():
    """Deferred structured-output sampling back-fills the scheduler's
    placeholders: draft zero, the held token, leads the worker's drafts."""
    scheduler = _scheduler(True, 3)
    request = _make_request("a", max_tokens=30)
    scheduler.add_request(request)
    worker = ScriptedWorker(scheduler, {"a": SAY_HI}, 3)
    checked: list[list[int]] = []

    def backfill(so: SchedulerOutput, _mro) -> None:
        # The step after this output is the one whose sampling would wait:
        # back-fill it now, as the engine core's deferred path does.
        pending = scheduler_outputs[-1] if scheduler_outputs else None
        if (
            pending is None
            or "a" not in (pending.sink_draft_zero_req_ids or ())
            or request.is_finished()
        ):
            return
        placeholders = list(pending.scheduled_spec_decode_tokens["a"])
        drafts = worker.consumed_drafts[id(pending)].get("a", [])
        scheduler.update_draft_token_ids_in_output(
            DraftTokenIds(["a"], [list(drafts)]), pending
        )
        filled = pending.scheduled_spec_decode_tokens["a"]
        assert len(filled) == len(placeholders)
        assert filled[0] == request.sink_provisional
        assert filled[1 : 1 + len(drafts)] == drafts
        pending.scheduled_spec_decode_tokens["a"] = placeholders
        checked.append(filled)

    scheduler_outputs: list[SchedulerOutput] = []
    run = drive(
        scheduler,
        worker,
        async_scheduling=True,
        on_schedule=lambda so: scheduler_outputs.append(so),
        on_output=backfill,
    )
    assert run.emitted["a"] == expected_reply(SAY_HI, 30)
    assert checked


def test_preempted_request_drops_its_held_token():
    """Async scheduling under KV pressure: a preempted request's output in
    flight (stale) commits its verified tokens, drops the held one, and the
    resumed request draws that position again."""
    scheduler = _scheduler(True, 2, num_blocks=7)
    scripts = {"a": SAY_HI, "b": _script(HOW, I_, HELP, YOU)}
    requests = {r: _make_request(r, max_tokens=40) for r in scripts}
    for request in requests.values():
        scheduler.add_request(request)
    worker = ScriptedWorker(scheduler, scripts, 2, draft_errors={("a", 1): 999})
    run = drive(scheduler, worker, async_scheduling=True)
    assert any(request.num_preemptions for request in requests.values())
    for req_id, script in scripts.items():
        assert run.emitted[req_id] == expected_reply(script, 40)


def test_engine_core_outputs_carry_sink_metrics():
    """The final output's sink metrics survive the engine-core-to-frontend
    encoding."""
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    metrics = RequestSinkMetrics(num_flagged_draft_zero=2, num_draft_zero_rejections=1)
    outputs = EngineCoreOutputs(
        outputs=[
            EngineCoreOutput(
                request_id="a",
                new_token_ids=[1],
                finish_reason=None,
                sink_metrics=metrics,
            )
        ]
    )
    decoded = MsgpackDecoder(EngineCoreOutputs).decode(MsgpackEncoder().encode(outputs))
    assert decoded.outputs[0].sink_metrics.to_dict() == metrics.to_dict()


def test_usage_reports_zero_flags_with_the_detector_on():
    scheduler = _scheduler(False, 2)
    scheduler.add_request(_make_request("a", max_tokens=4))
    worker = ScriptedWorker(scheduler, {"a": _script(1, 2, 3, 4)}, 2)
    run = drive(scheduler, worker, async_scheduling=False)
    assert run.finished["a"].sink_metrics == RequestSinkMetrics()


def _served(architecture: str):
    from types import SimpleNamespace

    return SimpleNamespace(model_config=SimpleNamespace(architecture=architecture))


@pytest.fixture
def unset_sink_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in (
        "VLLM_SINK_DETECTOR_COLUMN",
        "VLLM_SINK_DETECTOR_THRESHOLD",
        "VLLM_SINK_REJECTION",
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_sink_config_on_by_default_for_deepseek_v41(unset_sink_env):
    config = SinkConfig.for_model(_served("DeepseekV41ForCausalLM"))
    assert config == SinkConfig(
        column=2455, threshold=85000.0, rejection=True, max_retries=3
    )
    assert config.num_alternates == 3


def test_sink_config_switched_off(unset_sink_env):
    unset_sink_env.setenv("VLLM_SINK_REJECTION", "0")
    assert SinkConfig.for_model(_served("DeepseekV41ForCausalLM")) is None


def test_sink_config_off_for_other_models(unset_sink_env):
    assert SinkConfig.for_model(_served("OPTForCausalLM")) is None
    unset_sink_env.setenv("VLLM_SINK_REJECTION", "1")
    with pytest.raises(ValueError, match="no sink channel"):
        SinkConfig.for_model(_served("OPTForCausalLM"))


def test_sink_config_explicit_column_runs_the_detector_alone(unset_sink_env):
    unset_sink_env.setenv("VLLM_SINK_REJECTION", "0")
    unset_sink_env.setenv("VLLM_SINK_DETECTOR_COLUMN", "17")
    with pytest.raises(ValueError, match="THRESHOLD"):
        SinkConfig.for_model(_served("OPTForCausalLM"))
    unset_sink_env.setenv("VLLM_SINK_DETECTOR_THRESHOLD", "100")
    assert SinkConfig.for_model(_served("OPTForCausalLM")) == SinkConfig(
        column=17, threshold=100.0, rejection=False, max_retries=3
    )
    # On the model with an entry, the explicit column replaces its own.
    config = SinkConfig.for_model(_served("DeepseekV41ForCausalLM"))
    assert config is not None and (config.column, config.rejection) == (17, False)


@pytest.mark.parametrize("rejection", [False, True])
def test_sink_log_names_the_flagged_rows_position(
    monkeypatch: pytest.MonkeyPatch, rejection: bool
):
    """The Sink: line gives the largest flagged value with the output position
    of its own row: row r reads output token first + r, where first is draft
    zero (the next output token) with rejection on, else the last committed
    token. Here row 0 is clean and drafter row 1 holds the value."""
    from types import SimpleNamespace

    from vllm.v1.core.sched import scheduler as scheduler_module
    from vllm.v1.metrics.stats import SinkStats

    lines: list[str] = []
    monkeypatch.setattr(
        scheduler_module.logger,
        "info",
        lambda message, *args: lines.append(message % args),
    )
    scheduler = _scheduler(False, 5)
    scheduler.sink_rejection = rejection
    request = SimpleNamespace(request_id="r", num_output_tokens=688, sink_metrics=None)
    event = SinkEvent(
        SinkOutcome.NONE, SinkRow.DRAFT_ZERO, 1, -1, value=62720, value_row=1
    )
    scheduler._observe_sink_event(request, event, SinkStats())
    position = 689 if rejection else 688
    assert lines == [
        "Sink: request r, next output position 688: draft_zero row none, "
        f"1 drafter row(s) flagged, value 62720 at output position {position} (row 1)"
    ]
