# Custom logprob token ids return HTTP 500 under speculative decoding

A chat completion with `logprob_token_ids` fails with HTTP 500; the engine keeps serving. The
cause is in the V2 runner's verification: tokens that speculative decoding verifies come back
without logprobs, so the response has fewer logprob entries than tokens. Sink rejection is not
the cause. The fix is one vLLM patch,
[vllm/0267-Spec-decode-verified-tokens-carry-their-logprob_toke.patch](vllm/0267-Spec-decode-verified-tokens-carry-their-logprob_toke.patch),
on candidate 4's series with 0266. [vllm/series](vllm/series) is that series.
Production (spark-ds41f-1022) has the same code and is affected; nothing is deployed.

## What was seen (2026-10-24, run `tilelang-ports-serving-3`)

The `logprob-ids` scenario of `experiments/2026-10-24-tilelang-completion/controls_digest.py`
sent a chat completion with these settings:
- `logprobs: true` and `logprob_token_ids: [100, 1000, 3000, 4000, 5000]`, temperature 0;
- `chat_template_kwargs.thinking=false`;
- its first prompt, `max_tokens` 96, `ignore_eos`.

It returned HTTP 500 in candidate 4 (`spark-ds41f:19f2c20ed4d6-next-4`) and the TileLang ports
build. The engine log on dgx1 is
`results/private/determinism/tilelang-ports-serving-3/controls-candidate4.jsonl.engine-dgx1.log`:

```text
File ".../vllm/entrypoints/openai/chat_completion/serving.py", line 960, in chat_completion_full_generator
    logprobs = self._create_chat_logprobs(
File ".../vllm/entrypoints/openai/chat_completion/serving.py", line 1262, in _create_chat_logprobs
    step_top_logprobs = top_logprobs[i]
IndexError: list index out of range
```

The scenario's one-token probe (the same request with `max_tokens` 1) returned 200.

## Root cause

Line numbers are candidate 4's tree (`97fc3579`, vLLM commit `174a746f4`).

1. The chat and completion APIs send `logprob_token_ids` with `logprobs` unset
   (`chat_completion/protocol.py:786-792`, `completion/protocol.py:385`). Only
   `SamplingParams.num_logprobs` counts the ids. The V2 runner's `SamplingStates` takes
   `sampling_params.logprobs` alone (`sample/states.py:56`), so such a request has no logprobs
   there. Its ids live in the sampler's `LogprobTokenIdsState`.
2. The sampler (`sample/sampler.py`) asks `get_logprobs_dims` (line 148), which counts both,
   and passes the ids to `compute_topk_scores` (line 210). It handles a step without drafts:
   the step that finishes a prompt samples the first token there.
3. With DSpark, every later step has drafts and verifies them. With sink rejection on, that
   includes the step that verifies the first token, as draft zero. Verification takes one of
   two paths (`model_runner.py:1956-2001`), and neither returns the ids:
   - **Greedy verification** (`VLLM_GREEDY_VERIFICATION`, on by default) handles a
     temperature-0 step when `_verifies_greedily` finds no request that wants logprobs.
     It checks `sampling_states.max_num_logprobs(idx) == NO_LOGPROBS`
     (`model_runner.py:2285`), which is true for an ids-only request. That path returns
     `logprobs_tensors=None` (line 2336).
   - **The rejection sampler** (`spec_decode/rejection_sampler.py`) sizes its logprobs from
     `sampling_states.max_num_logprobs` too (line 440). With no other request wanting
     logprobs it returns none. Beside a request with top logprobs, it calls
     `compute_topk_scores` without the ids state (line 218), so every row gets the top-k
     tokens instead of the ids. It also skips the sink alternates' logprobs (line 377).
4. The scheduler extracts a step's logprobs only when the step has them
   (`scheduler.py:3369-3375`). A verified step brings none, so its tokens go out without
   logprobs. Under sink rejection the first token's row is lost as well: the step that
   verifies it has no logprobs, so `_sink_logprobs` never runs and the row kept with the
   held token is dropped (`scheduler.py:3192-3216`).
5. The output processor appends logprobs only for outputs that carry them. The response
   therefore holds fewer logprob entries than tokens. `_create_chat_logprobs` indexes them
   by token (`serving.py:1262`), so it raises IndexError and the request returns HTTP 500.
   The engine finished the request first, so it keeps serving.
6. The one-token probe passed because its only token is sampled by the plain sampler at the
   prompt's last step, then committed at once (`max_tokens`).

The scheduler's sink-rejection handling of logprob rows (`_sink_logprobs`,
`_hold_sink_provisional`, `_concat_logprobs`) is not at fault. Given rows with the ids, it
keeps one row per committed token, including a rejected draft zero's replacement. The new
scheduler test below passes on candidate 4's scheduler.

**Who triggers it:** every chat or completion request with `logprob_token_ids` that
generates more than one token under speculative decoding, at any temperature. A request
that sends `logprobs` (top-k) together with the ids is not possible through the APIs, which
clear `logprobs` when the ids are set. Through the offline API, such a request gets the
top-k tokens instead of its ids for verified tokens.

**Builds:**
- **Production, spark-ds41f-1022** (image `vllm-ds41f-kkref:19f2c20ed4d6-1022-4`, vLLM tree
  `e68aa19c`, commit `8fc15c18f`): affected. Its `rejection_sampler.py`, `sampler.py`,
  `logprob.py`, `states.py`, `scheduler.py` and `request.py` are byte-identical to candidate
  4's. Its `model_runner.py` differs by one comment line. `config/cluster.json` runs DSpark
  and leaves `VLLM_GREEDY_VERIFICATION` at its default. No request was sent to production
  to check this.
- **Candidate 4 and the TileLang ports branch:** affected (seen).
- **Builds without sink rejection:** affected too, by the same code (not run). The first
  token keeps its row, and the response fails at its second token. The V2 rejection sampler
  in upstream vLLM (`c687594b8a`, 2026-09-13) has the same code. Earlier releases were not
  checked one by one.

## The fix (0267)

vLLM commit `2c6f86e0e` (tree `1f642c0f`) on branch `logprob-ids-verified`, on 0266's
`5acb05bc7`. The branch is checked out in `~/projects/vllm-ds41-logprob-ids`, a worktree of
`~/projects/vllm-ds41-prompt-logprobs`.

- `spec_decode/rejection_sampler.py`: verification takes the sampler's
  `get_logprobs_dims` (top logprobs and per-request ids). It passes the ids state and each
  logits row's request-state index (`expanded_idx_mapping`, per chunk) to
  `compute_topk_scores`, as the sampler does. It also scores the sink alternates and
  clamps dead rows whenever any request wants logprobs.
- `model_runner.py`: `_verifies_greedily` takes greedy verification only when
  `get_logprobs_dims` is None, which rules out ids too. The batch-sharded gather sizes its
  columns with the ids for verification steps as well.
- `sample/sampler.py`: `get_logprobs_dims` loses its `include_token_ids` switch, whose
  only use was that gather.
- `boundary_checkpoint.py`: its four slot constants follow `watermark.py`'s `HAS_TRITON`
  guard, so the runner, sampler and rejection sampler import without Triton (the CPU tests
  below). With Triton they are unchanged.

Requests without logprobs take the same paths as before. A batch holding an ids request now
verifies through the rejection sampler instead of greedy verification, as a batch with
top-logprob requests already does. Greedy verification is defined to commit the rejection
sampler's tokens bit for bit; the lab check compares the tokens.

Not exercised here: batch-sharded sampling. Sink rejection refuses it, so DS4.1 never runs it.

## Tests

On CPU on a Mac (macOS arm64, torch 2.14.1, no Triton), the tree from `PYTHONPATH`, as in
[runs/run-suites.sh](runs/run-suites.sh).

`tests/v1/worker/test_gpu_verified_logprob_token_ids.py` (new) runs the runner's real
`SamplingStates`, `LogprobTokenIdsState`, `Sampler.get_logprobs_dims`, `RejectionSampler` and
`compute_topk_scores` on the host. The device buffers are host tensors, and the Triton
launches they reach are torch references. `_verify` is replaced, as in
`test_gpu_rejection_sampler_chunking.py`. There are 28 cases:
- `test_verified_tokens_carry_their_logprob_token_ids` (24): every verified token of the ids
  request carries the five ids' logprobs in its row's distribution, with its own logprob and
  rank. The request runs alone, and beside a top-2 request and one without logprobs. The
  cases cover one verification chunk, several, adaptive verification, and every
  `logprobs_mode`. The top-2 request keeps its top two.
- `test_sink_alternates_carry_logprobs_for_logprob_token_ids`: an ids-only batch scores the
  sink alternates. Without that, the scheduler's assertion on a rejected draft zero's
  logprob would stop the engine once verified steps carry logprobs.
- `test_greedy_verification_only_without_logprobs` (3): greedy verification for a batch
  without logprobs; not for one with ids, or with top logprobs.

`tests/v1/core/test_sink_draft_zero.py` gains `test_logprob_token_ids_of_every_token`
(sync and async). Its scripted worker returns rows with the ids, scored in each draw. The
test runs the scheduler, the output processor and `_create_chat_logprobs`. It checks one
entry per token, each listing the ids, scored in the draw that chose the token. That
includes a rejected draft zero's replacement.

| Run | Result |
| --- | --- |
| The new worker tests on 0266's tree with only 0267's `boundary_checkpoint.py` ([runs/tests-before-fix.txt](runs/tests-before-fix.txt)) | 26 failed (every rejection-sampler and sink case, and greedy verification with ids); 2 passed (greedy verification without logprobs, and with top logprobs) |
| The new scheduler test on 0266's scheduler ([runs/tests-fix.txt](runs/tests-fix.txt)) | 2 passed: the scheduler half was right |
| 0267: the new tests, `test_gpu_rejection_sampler_chunking.py` and `test_sink_draft_zero.py` ([runs/tests-fix.txt](runs/tests-fix.txt)) | 94 passed, 4 skipped (CUDA) |
| The scheduler suites of 0266's record on both trees, with the worker tests on 0267 ([runs/suites.txt](runs/suites.txt)) | 0266: 272 passed. 0267: 303 passed (the 31 new ones). Both: the same 72 failing ids as 0266's record (gated `meta-llama/Llama-3.2-1B` downloads, model configs) |

`test_gpu_rejection_sampler_chunking.py`'s CUDA test calls the changed methods. It now
passes `logprobs_dims=(2, 0)` and the expanded mapping; it is skipped here (no CUDA).

`ruff check` and `ruff format --check` (0.14.0, the tree's pre-commit pin) are clean.
The series applies to `19f2c20ed` with `git am` (224 patches): 0265 gives candidate 4's tree
`97fc3579`, 0266 gives `baa1aea6` and 0267 gives the branch's `1f642c0f`
([runs/series-check.txt](runs/series-check.txt)).

## Lab check

Run `logprob-token-ids`, spec [serving-check.json](serving-check.json). One `measure` job on
the ring, with no bench and one extra, [logprob_ids_check.py](logprob_ids_check.py), on four
arms:

| Arm | Config | Requests with ids |
| --- | --- | --- |
| `candidate4-0266` | `experiments/2026-10-24-prompt-logprobs-sink/fix-tp4.json`: candidate 4 with 0266 | one, to show the 500 and that the engine keeps serving |
| `candidate4-0267` | [fix-tp4.json](fix-tp4.json): candidate 4 with 0266 and 0267 ([overlay/](overlay/), six files) | yes |
| `production-0266` | `experiments/2026-10-24-prompt-logprobs-sink/production-fix-tp4.json`: the promoted recipe with 0266, the pending hotfix | none |
| `production-0267` | [production-fix-tp4.json](production-fix-tp4.json): the promoted recipe with 0266 and 0267, a hotfix candidate for both | yes |

`overlay-production/` holds production's `model_runner.py` with 0267: it differs from
candidate 4's by one comment line. The other five files are the same for both images.

The check runs these steps:
1. The seven other prompts, one at a time, without logprobs. Alone, they take greedy
   verification.
2. The probe with `logprobs` (top 0: each chosen token's logprob).
3. In `candidate4-0266`: the probe with the ids once, then the probe again.
4. In the fix arms:
   - the probe with the ids;
   - the same through `/v1/completions`, on the probe's prompt tokens;
   - the seven and the probe with the ids, all at once (the seven then verify with the
     rejection sampler);
   - teacher forcing: completions of the probe's prompt plus its first P reply tokens
     (P = 0, 1, 2, 3, 15, 31, 63, 95), one token each, with the ids. Each scores the ids at
     reply position P with the plain sampler. The probe scored them in its verified steps.
5. In the reference arms (no 0267): the seven and the probe with `logprobs`, all at once.

An arm passes on these:
- every request returns HTTP 200 (except the one that shows the bug);
- the probe with the ids returns one logprob entry per token, each listing the five ids
  with logprobs of at most 0. The completion does the same;
- the probe's tokens, and each chosen token's logprob, are the same with and without the
  ids, and alone or beside the seven;
- the seven's replies are the same alone and beside the probe.

[tables_arms.py](tables_arms.py) then requires each `-0267` arm's temperature-0 digests to be
its `-0266` arm's. These cover the seven alone and together, and the probe's tokens and
chosen logprobs alone and together. Three results are recorded, not gated: the teacher's
largest gap, the completion's tokens against the chat's, and the ids' logprobs alone against
beside the seven.

Results: pending. A window held by run `request-inventory` was open when this was written.
[runs/lab/run.sh](runs/lab/run.sh) waits for it to close.

## Production

Production is affected now. A client that sends `logprob_token_ids` gets HTTP 500 for any
reply longer than one token. Nothing else breaks: the engine keeps serving every other
request. This session made no production change. The options, for the owner:

1. **Hotfix both:** if the owner takes the pending 0266 hotfix, boot
   `production-fix-tp4.json` from this experiment instead, once the lab check passes. One
   restart fixes both.
2. **Wait for the next release** with 0266 and 0267 (`vllm/series`). Until then, the ids
   fail with HTTP 500 and nothing else is affected.
3. **Reject at validation:** answer `logprob_token_ids` with HTTP 400 under speculative
   decoding. This is a code change, a restart and a lost feature, with no gain over 1.

The recommendation is 0267 in the next release. If the owner takes the 0266 hotfix,
take this one with it (option 1). Both wait for the owner's go.

## Follow-ups

- `controls_digest.py`'s `logprob-ids` scenario passes once 0267 is in the build it
  measures. Until then its HTTP 500 stops the script, as in run `tilelang-ports-serving-3`.
- Upstream: upstream vLLM's V2 rejection sampler has the same gap. 0267's rejection-sampler
  and sampler hunks could go upstream on their own; greedy verification is ours.
