# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Copy drafts in the scheduler: the verification lookahead apart from the
drafter's block, and a copying request's rows laid out from its reported copy
state, within the decode-row envelope.

The worker is scripted: every draft is accepted, and each output reports the
copy allocation the test gives a request. The scheduler is the real async one,
driven as the engine core drives it (a batch queue two deep: step t + 1 is
scheduled before step t's output is processed).
"""

import json
from collections import deque

import pytest

from vllm.config.speculative import CopyDraftsConfig, SpeculativeConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput

pytestmark = pytest.mark.cpu_test

BLOCK = 5  # DSpark's block (num_speculative_tokens)
LOOKAHEAD = 16  # copy_drafts.max_drafts

# A minimal OPT-125m configuration, so the scheduler builds without the hub
# (test_aligned_prefill_chunks.py's).
_OPT_CONFIG = {
    "architectures": ["OPTForCausalLM"],
    "model_type": "opt",
    "hidden_size": 768,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "ffn_dim": 3072,
    "max_position_embeddings": 2048,
    "vocab_size": 50272,
    "word_embed_proj_dim": 768,
    "torch_dtype": "float16",
    "do_layer_norm_before": True,
    "pad_token_id": 1,
    "bos_token_id": 2,
    "eos_token_id": 2,
}


def test_levels_double_from_the_minimum_match():
    assert CopyDraftsConfig().levels() == (16, 32, 64, 128, 256)
    assert CopyDraftsConfig(min_match=24).levels() == (24, 48, 96, 192)
    assert CopyDraftsConfig(min_match=256).levels() == (256,)


def test_the_lookahead_is_the_block_without_copy_drafts():
    spec = SpeculativeConfig(model="ngram", num_speculative_tokens=BLOCK)
    assert spec.copy_drafts is None
    assert spec.num_verification_tokens == BLOCK
    hashed = spec.compute_hash()
    spec.copy_drafts = CopyDraftsConfig(max_drafts=LOOKAHEAD)
    assert spec.num_verification_tokens == LOOKAHEAD
    # Copy drafts change the decode graphs' width, so the compile hash.
    assert spec.compute_hash() != hashed
    spec.copy_drafts = CopyDraftsConfig(max_drafts=64)
    assert spec.num_verification_tokens == 64


@pytest.mark.parametrize(
    "copy_drafts",
    [
        CopyDraftsConfig(),  # n-gram is not DSpark with adaptive verification
    ],
)
def test_copy_drafts_need_dspark_with_adaptive_verification(copy_drafts):
    with pytest.raises(ValueError, match="copy_drafts requires method 'dspark'"):
        SpeculativeConfig(
            model="ngram", num_speculative_tokens=BLOCK, copy_drafts=copy_drafts
        )


def _scheduler(monkeypatch, tmp_path, *, copy_drafts: bool, max_num_seqs: int):
    utils = pytest.importorskip("tests.v1.core.utils")
    # The configuration validates the n-gram fixture for the V1 runner; the
    # scheduler then runs as the V2 runner's (use_v2_model_runner below).
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    async_scheduler = utils.AsyncScheduler

    def scheduler_cls(*args, vllm_config, **kwargs):
        # The scheduler tests' GPU n-gram fixture (async scheduling takes it), with
        # copy drafts set after the configuration's validation, which asks for a
        # DSpark drafter on the V2 runner this test does not load.
        if copy_drafts:
            vllm_config.speculative_config.copy_drafts = CopyDraftsConfig(
                max_drafts=LOOKAHEAD
            )
        return async_scheduler(*args, vllm_config=vllm_config, **kwargs)

    monkeypatch.setattr(utils, "AsyncScheduler", scheduler_cls)
    (tmp_path / "config.json").write_text(json.dumps(_OPT_CONFIG))
    scheduler = utils.create_scheduler(
        model=str(tmp_path),
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=1024,
        num_speculative_tokens=BLOCK,
        speculative_method="ngram_gpu",
        async_scheduling=True,
        skip_tokenizer_init=True,
        device="cpu",
        use_v2_model_runner=True,
    )
    return utils, scheduler


def _output(scheduler_output: SchedulerOutput, alloc: dict[str, int] | None):
    """Every scheduled draft accepted, and with ``alloc`` each request's copy
    allocation reported."""
    req_ids = list(scheduler_output.num_scheduled_tokens)
    sampled = []
    for req_id in req_ids:
        drafts = len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, ()))
        sampled.append([7] * (drafts + 1))
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=sampled,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        copy_draft_alloc=(
            None if alloc is None else [alloc.get(req_id, BLOCK) for req_id in req_ids]
        ),
    )


def _run(scheduler, requests, steps: int, alloc: dict[str, int] | None):
    """The engine core's batch queue: schedule step t + 1, then process step t's
    output. Returns each step's scheduled drafts per request."""
    for request in requests:
        scheduler.add_request(request)
    queue: deque[SchedulerOutput] = deque()
    drafts: list[dict[str, int]] = []
    for _ in range(steps):
        scheduled = scheduler.schedule()
        drafts.append(
            {
                req_id: len(tokens)
                for req_id, tokens in scheduled.scheduled_spec_decode_tokens.items()
            }
        )
        rows = sum(
            scheduled.num_scheduled_tokens[req_id]
            for req_id in scheduled.scheduled_spec_decode_tokens
        )
        assert rows <= scheduler.max_decode_rows, (rows, scheduler.max_decode_rows)
        queue.append(scheduled)
        if len(queue) == 2:
            done = queue.popleft()
            scheduler.update_from_output(done, _output(done, alloc))
    return drafts


def test_without_copy_drafts_every_request_keeps_the_block(monkeypatch, tmp_path):
    utils, scheduler = _scheduler(
        monkeypatch, tmp_path, copy_drafts=False, max_num_seqs=4
    )
    assert scheduler.max_decode_rows == 4 * (1 + BLOCK)
    requests = utils.create_requests(num_requests=2, num_tokens=20, max_tokens=500)
    drafts = _run(scheduler, requests, 12, alloc=None)
    assert {n for step in drafts for n in step.values()} == {BLOCK}


def test_a_copying_request_widens_a_step_after_its_report(monkeypatch, tmp_path):
    utils, scheduler = _scheduler(
        monkeypatch, tmp_path, copy_drafts=True, max_num_seqs=4
    )
    assert scheduler.max_decode_rows == 4 * (1 + BLOCK) + LOOKAHEAD - BLOCK
    requests = utils.create_requests(num_requests=2, num_tokens=20, max_tokens=500)
    copying = requests[0].request_id
    drafts = _run(scheduler, requests, 12, alloc={copying: LOOKAHEAD})
    # The prefill step, the first decode step (laid out before any output came
    # back), then the output of the prefill reports the allocation: the step
    # after the next one is the first wide one.
    widths = [step.get(copying) for step in drafts]
    first = widths.index(LOOKAHEAD)
    assert all(w in (None, BLOCK) for w in widths[:first])
    assert all(w == LOOKAHEAD for w in widths[first:])
    other = requests[1].request_id
    assert {step[other] for step in drafts if other in step} == {BLOCK}


def test_wide_requests_share_the_envelope(monkeypatch, tmp_path):
    utils, scheduler = _scheduler(
        monkeypatch, tmp_path, copy_drafts=True, max_num_seqs=4
    )
    requests = utils.create_requests(num_requests=4, num_tokens=20, max_tokens=500)
    alloc = {request.request_id: LOOKAHEAD for request in requests}
    drafts = _run(scheduler, requests, 12, alloc=alloc)
    # Four running requests leave one wide block: 3 x 6 + 17 = 35 rows.
    last = drafts[-1]
    assert sorted(last.values()) == [BLOCK, BLOCK, BLOCK, LOOKAHEAD]


def test_two_running_requests_both_widen(monkeypatch, tmp_path):
    utils, scheduler = _scheduler(
        monkeypatch, tmp_path, copy_drafts=True, max_num_seqs=4
    )
    requests = utils.create_requests(num_requests=2, num_tokens=20, max_tokens=500)
    alloc = {request.request_id: LOOKAHEAD for request in requests}
    drafts = _run(scheduler, requests, 12, alloc=alloc)
    # Two empty slots' rows (12) and the one wide block's (11) cover both.
    assert sorted(drafts[-1].values()) == [LOOKAHEAD, LOOKAHEAD]


def test_preemption_forgets_the_allocation(monkeypatch, tmp_path):
    utils, scheduler = _scheduler(
        monkeypatch, tmp_path, copy_drafts=True, max_num_seqs=4
    )
    requests = utils.create_requests(num_requests=1, num_tokens=20, max_tokens=500)
    request = requests[0]
    _run(scheduler, requests, 6, alloc={request.request_id: LOOKAHEAD})
    assert request.copy_draft_alloc == LOOKAHEAD
    scheduler.running.remove(request)
    scheduler._preempt_request(request, 0.0)
    assert request.copy_draft_alloc == 0


def test_widen_only_uniform_placeholders(monkeypatch, tmp_path):
    _, scheduler = _scheduler(monkeypatch, tmp_path, copy_drafts=True, max_num_seqs=4)

    class Request:
        is_prefill_chunk = False
        copy_draft_alloc = LOOKAHEAD
        spec_token_ids = [-1] * BLOCK

    request = Request()
    assert scheduler._widen_copy_placeholders(request, 4) == 4
    assert request.spec_token_ids == [-1] * (BLOCK + 4)
    # Widened by an attempt that was not scheduled: laid out again for the room
    # of the step at hand, down to the block.
    assert scheduler._widen_copy_placeholders(request, 2) == 2
    assert request.spec_token_ids == [-1] * (BLOCK + 2)
    assert scheduler._widen_copy_placeholders(request, 0) == 0
    assert request.spec_token_ids == [-1] * BLOCK
    # Real drafts (synchronous scheduling) and prefill chunks: left alone.
    request.spec_token_ids = [3] * BLOCK
    assert scheduler._widen_copy_placeholders(request, 4) == 0
    request.spec_token_ids = [-1] * BLOCK
    request.is_prefill_chunk = True
    assert scheduler._widen_copy_placeholders(request, 4) == 0


def test_waiting_requests_keep_their_slots_rows(monkeypatch, tmp_path):
    """A free slot a waiting request may fill this step is not lent to a copy:
    two copying requests that both widen alone keep one wide block between them
    once two more requests wait."""
    utils, scheduler = _scheduler(
        monkeypatch, tmp_path, copy_drafts=True, max_num_seqs=4
    )
    requests = utils.create_requests(num_requests=4, num_tokens=20, max_tokens=500)
    alloc = {request.request_id: LOOKAHEAD for request in requests}
    drafts = _run(scheduler, requests[:2], 12, alloc=alloc)
    assert sorted(drafts[-1].values()) == [LOOKAHEAD, LOOKAHEAD]
    for request in requests[2:]:
        scheduler.add_request(request)
    scheduled = scheduler.schedule()
    widths = sorted(
        len(scheduled.scheduled_spec_decode_tokens[r.request_id]) for r in requests[:2]
    )
    assert widths == [BLOCK, LOOKAHEAD]
