# Sink rejection v2: each step's last sampled token is verified as draft zero

Owner, 2026-10-19: one release carrying both the sink detector and a correct correction of a sunk
token, with the forward pass untouched. v1 of the candidate (record on the local branch
`sink-rejection`, never merged; patch 0221, vLLM 82eff9023) detected sunk rows and replaced a sunk *committed* token with a custom
rollback. A read-only review the same day found the rollback unsafe. This record rebuilds the
correction on the speculative-decoding machinery.

**Status: the full GPU gate passed** (2026-10-20 13:31-18:50 UTC, [GPU gate](#gpu-gate-owner-2026-10-20-run-the-full-gate-the-grid-settles-the-step-cost)).
vLLM branch `sink-rejection-v2`, one commit on `r6x-candidate` f3b58ed9d (5ac8364ec), in
~/projects/vllm-ds41-sink-rejection; the series patch
[vllm/0221-Sink-detector-and-draft-zero-rejection.patch](vllm/0221-Sink-detector-and-draft-zero-rejection.patch)
(provisional number 0221). Threshold 85,000. On the TP3 triangle against r6x: 256/256 greedy release
responses identical (KL 0) with zero flags on the release suites; engine grid, throughput, sampled,
single-stream, logprobs and prefill level; time to first token one decode step longer; `Say hi.`,
`Thanks!` and `ok` cleaned with every rank agreeing, alone, in a staggered eight-stream batch and
without speculation; 144/144 greedy replies identical with and without speculation. Not exercised: no
long context (64K, 512K) collapses on r6x, so no rejection deep in the caches was seen. Not promoted;
no recipe change.

## What v1 got wrong, and what v2 does instead

| v1 (0221) | v2 |
|---|---|
| **Never ran in production.** The recipe sets `VLLM_USE_V2_MODEL_RUNNER=1`, and DSpark, block verification and adaptive verification exist only in the V2 runner (`vllm/v1/worker/gpu/`). 0221 changed the V1 runner (`gpu_model_runner.py`, `vllm/v1/sample/rejection_sampler.py`) and listed the V2 runner under "not covered". | Built on the V2 runner: its sampler, rejection sampler, greedy verification from vocabulary shards, DSpark/DFlash inputs and request state. The V1 runner is untouched; the detector needs the V2 runner, which the recipe sets (not enforced). |
| A zero-draft async step left the sink token's KV in place (`update_num_computed_tokens_for_batch_change` applied the correction only with drafts). | No rollback exists. The sunk token was never committed: it is draft zero of the next step, and a rejected draft zero rolls back with the rejected-draft accounting (the scheduler takes `len(spec)` off its computed count; the worker's computed count does not advance). |
| A rolled-back request's drafts scattered into the neighbouring request's slot and wrote its MTP KV at wrong positions (`clamp(min=0)` covered request 0 only). | A request with no valid row anchors its drafter query at draft zero's own position and writes no context KV (`prepare_dflash_inputs`); tested with the rejected request in the middle of a batch. |
| No cross-rank agreement on the flags. | Settings come from the same environment on every rank. Flags: an open assumption (below) with a lab assertion, `VLLM_SINK_DETECTOR_CHECK_RANKS=1`, and per-rank flag counts in each worker's log. |
| The frontend hold-back was decided by the API-server process's environment (duplicate output when unset, assertion when set and nothing held). | No frontend change: the scheduler emits committed tokens only. |
| A stale (preempted, in-flight) output's rollback appended without popping. | A stale output commits its verified tokens and drops the held one; the resumed request draws that position again. Tested under KV pressure with async scheduling. |
| The replacement's logprob came from the sunk row's distribution. | The replacement carries its own logprob and rank in the draw that produced the replaced token, with that draw's top-k. |
| `swap_row` used pageable host-to-device copies, so "no extra synchronization" was false. | The V2 runner keys per-request state by its persistent slot: nothing moves rows, `swap_row`/`move_row` are gone. The step adds one pinned asynchronous copy (the decoding flags) and no synchronization. |
| Random rows lost the FlashInfer sampler. | FlashInfer stays on with rejection off (detector only or no detector); with rejection on it is off, because a draw's alternates are the next of its Gumbel scores, which FlashInfer does not expose. |
| Threshold 1e4 with no margin. | Default 50,000 (analysis below). |
| Found here: v1's `usage.sink_rejection` was never filled (the output processor did not pass `sink_metrics` on). Its residual buffer made sequence-parallel prefill all-gather all four residual streams (4x the bytes) instead of the normalized hidden state. | `sink_metrics` reaches `CompletionOutput`; with the detector on every response carries the field (zeros when nothing was flagged). The model keeps only the watched column, `[tokens, 4]` float32 (a gather of 16 bytes a token under sequence parallelism). |

## The design

**The contract.** With `VLLM_SINK_REJECTION=1`, the last token a step samples (bonus, recovered,
or the last token before dead verification rows) is the request's *provisional* token. The scheduler
does not commit it. It holds it (`Request.sink_provisional`, with its logprobs entry) and schedules it
as **draft zero** of the request's next step, ahead of the drafter's tokens;
`SchedulerOutput.sink_draft_zero_req_ids` names those requests. The V2 runner's inputs already are
`[last sampled token, drafts]`, so the worker only translates the counts (drafter tokens = spec tokens
minus draft zero) and asserts that the scheduler's draft-zero requests are exactly the requests that
are decoding. Everything else is the existing machinery:

- **KV.** A step's spec tokens are draft zero and the drafter's. `num_accepted = len(output) - 1`,
  `num_rejected = len(spec) - num_accepted`, both unchanged. A rejected draft zero rejects the whole
  step, so its slot is rewritten by the next step like any rejected draft's. Under async scheduling the
  placeholders count the spec tokens only (`num_output_placeholders += len(spec)`): the held token
  becomes the next step's draft zero and is counted there.
- **Prefix cache.** Only committed tokens are hashed and cached (`num_computed - placeholders`
  equals the committed count; sync scheduling caps at `request.num_tokens`). Nothing is ever forgotten.
- **Structured output.** The grammar accepts committed tokens only; nothing is rolled back. A
  draft-zero request's masks are one per spec token, starting after draft zero (the mask before draft
  zero was its own step's). Deferred async sampling back-fills draft zero from the held token.
- **Output.** The frontend receives committed tokens only. A step that rejects draft zero emits nothing
  for that request.
- **Terminal tokens.** A held token that is EOS, a stop token, or the max-tokens or max-model-len
  token is committed as soon as it is sampled, unverified: no step follows to verify it. A sunk final
  token is therefore never checked.
- **Preemption, a new streaming input.** The held token was never emitted and is dropped; the resumed
  request draws that position again.

**The verifier's rule.** Row 0's input is draft zero.

- **Draft zero:** accepted unless row 0 is flagged. A flagged draft zero is rejected, and the step's one
  token is the next alternate of the draw that produced it (sampling without replacement), with that
  alternate's stored logprob and rank. At most `VLLM_SINK_REJECTION_MAX_RETRIES` alternates are tried
  per position (default 3). Past them, or when the next alternate is impossible, the sunk token is
  committed as it is and counted as a **fallthrough**.
- **Drafter rows j >= 1:** 0221's cut. A flagged row j masks its draft in row j-1 before the
  verification kernels run, so row j-1 rejects it and recovers its best other token (greedy) or
  resamples the residual without it (random, one-hot drafts). The logits are restored before logprobs
  are computed.
- **Alternates** of each request's last draw are the top of the same scores after the drawn token: the
  logits for greedy rows, and for random rows the processed logits plus the same Gumbel noise. The noise
  is keyed by the request's seed, the row's position and the token, shared with the sampler through
  `gumbel_noised_scores`. A random row that rejected its draft drew from the residual, so its alternates
  exclude that draft. They stay on the device per request slot, with logprobs and ranks when the batch
  returns logprobs. On the all-greedy fast path each rank takes its shard's top candidates of the last
  rows, and one all-gather of `[requests, 8]` floats merges them identically on every rank.
- **DSpark after a rejected draft zero.** No row is valid, so the drafter anchors the replacement at
  draft zero's position. Its context up to the position before is in the draft KV cache already, so it
  drafts normally. (A drafter that needed the previous row's hidden state could not; the runner refuses
  rejection with drafters other than DSpark and DFlash.)
- **Dead verification rows** (adaptive verification) skipped routed MoE, so their flags are ignored.
  A token committed through the dead-row clamp is the step's last and is verified next step as draft
  zero.

**Where it lives.** `vllm/v1/sample/sink.py`: configuration and the host form of the events.
`vllm/v1/worker/gpu/sample/sink.py`: two Triton kernels, registered for the startup warmup (the
candidates of each request's last draw, and the policy); the flags, the cut and the merge of the
candidates are a few sync-free torch operations; and the per-slot state. The policy is one kernel per
step. It
applies the draft-zero rule, takes the new alternates, writes the host tokens with draft zero in front,
takes a rejected draft zero out of the request state (`total_len - 1`, its bin count), sets
`num_rejected` to the request's row count, and writes one event record per request. The runner calls
it after verification in all three sampling paths (plain sampler, rejection sampler, greedy
verification from shards). The scheduler side is in `scheduler.py` / `async_scheduler.py`, with
`stops_on_next_token` beside `check_stop` in `sched/utils.py`.

**Refused at startup** (rejection on): pipeline parallelism, batch-sharded sampling, prefill context
parallelism, dual-batch overlap (detector too), model-specific samplers, drafters other than DSpark and
DFlash, probabilistic drafts, watermarking, sampling masks, trace replay, returned routed experts, KV
transfer, request boundary checkpoints, diffusion models. None of these is in the recipe.

**Surfaces.** `vllm:sink_flagged_rows{row="draft_zero"|"drafter"|"prompt"}` (where each flag landed),
`vllm:sink_draft_zero_rejections`, `vllm:sink_draft_cuts`, `vllm:sink_fallthroughs`. The periodic stats
line carries the same counts. The scheduler logs a line per event with the request id and output
position, and each worker rank logs its own cumulative counts (`Sink detector, rank R: N flagged rows
(draft zero a, drafter b, prompt c)`), read back without a synchronization. `usage.sink_rejection` on
chat and completion responses has the same fields, summed over a request's sequences (n > 1). With
`VLLM_SINK_REJECTION=0` the detector counts and decoding is r6x's.

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_SINK_DETECTOR_COLUMN` | -1 (off) | the watched residual column; 2455 for V4.1 Flash |
| `VLLM_SINK_DETECTOR_THRESHOLD` | 50000 | `max over streams of \|residual[column]\|` above which a sampled row is sunk |
| `VLLM_SINK_REJECTION` | 0 | the correction (needs the column) |
| `VLLM_SINK_REJECTION_MAX_RETRIES` | 3 | alternates per provisional token, so rejections per position |
| `VLLM_SINK_DETECTOR_CHECK_RANKS` | 0 | lab check: all-gather each step's flags and raise if the ranks differ (one collective a step) |

The compiled model depends on the column alone: the other four are excluded from the compile cache key,
so the detector-only and rejection arms share one compiled artifact.

## The threshold

**Superseded by the screen and the survey (below; threshold now 85,000):** the analysis here saw only short chat answers, whose largest
clean row was 10,688. On longer reasoning text, number tokens (a lone space before digits, `202`)
reach 47,360-51,968 in the reference with confident, correct predictions, and the engine's own
numerics put one of them over 50,000. Rows at 115,712-124,416 (prompt words) predict like BOS.

[analysis/sink_column.txt](analysis/sink_column.txt) (v1's reference dumps, extended) and
[analysis/effort_rows.txt](analysis/effort_rows.txt) ([analysis/effort_rows.py](analysis/effort_rows.py)
over the 40 runs of the effort-line variants experiment, on a local branch not merged).
The detector reads `|residual[2455]|` after the last layer, at the sampled rows only: the last prompt
row and every generated token's row.

| Sampled rows | `\|residual[2455]\|` after the last layer |
|---|---|
| sunk (the collapse) | 108,032 to 125,440 (` can` at efforts 64 and 75 and in v1's dumps, ` there` at 75, ` I` at 90) |
| clean, largest | 10,688 (` effectively`, effort 50, a run stopped mid-decode); 9,984 (` friendly`) among finished runs. Reasoning words that only the last layer writes into the column |
| clean, normal | 150 to 300 |
| last prompt row | at most 89 |

Threshold 50,000 is 2.2x below the smallest sunk row and 4.7x above the largest clean one. Prompt rows
that are not sampled can carry large values: the effort line's ` (` at 7,008, the first user token at
132,096-136,192 without the effort line, five effort-50 prompts at 126,976-130,560 at a late user
token, BOS at 315,392. The detector never reads them; a prefill chunk's last row is computed but not
sampled, and its flag is ignored.

## Cost

- **Latency.** Every token is emitted one step later than r6x, when its step's successor verifies it.
  Time to first token grows by one decode step. Requests with `max_tokens=1`, and replies whose first
  token ends them, are unchanged.
- **Throughput.** Unchanged: the same steps commit the same tokens, shifted by one step. A request
  takes one step more per rejection. A final EOS, stop or max-tokens token is committed without the
  extra step, and async scheduling skips the step after a prefill that samples the max-tokens token.
- **Per step, rejection on.**
  - The flags (one gather of 4 floats per sampled row) and the policy kernel.
  - With drafts: the cut and restore (two gathers and a scatter over the target rows).
  - The candidates kernel over the vocabulary for one row per request, and the merge.
  - On the greedy fast path: a top-k over the last rows' local logits, and one all-gather of 8 floats
    per request, the only added collective.
- **Per step, detector only:** the flags and the policy kernel.
- **Per forward:** the model copies one column of the four residual streams into `[tokens, 4]` float32.
- **Sampling.** With rejection on, an unseeded random request with top-k or top-p, in a step without
  drafts, is drawn by the Gumbel kernels instead of FlashInfer's sampler: the same distribution, other
  tokens than r6x for the same request. Greedy and seeded requests draw exactly as on r6x.

The GPU gate measures the step cost.

## Change (vLLM, 29 files; tests 2 new files)

| Area | Files |
|---|---|
| configuration, events | `vllm/v1/sample/sink.py`, `envs.py` (threshold 50,000, rank check, compile-key exclusions) |
| model | `models/deepseek_v4_1/nvidia/model.py`, `vl_model.py` (the SM120 model GB10 loads): `get_sink_residual_columns()` |
| V2 runner | `worker/gpu/sample/sink.py` (new), `model_runner.py`, `sample/sampler.py`, `sample/gumbel.py` (`gumbel_noised_scores`, the argmax unchanged), `sample/output.py`, `spec_decode/rejection_sampler.py`, `spec_decode/dflash/speculator.py` (zero valid rows), `async_utils.py`, `warmup.py` (warmup steps carry draft zero; counts reset after) |
| scheduler | `core/sched/scheduler.py`, `async_scheduler.py`, `output.py`, `utils.py`, `request.py`, `structured_output/__init__.py` |
| surfaces | `v1/metrics/stats.py`, `loggers.py`, `v1/engine/__init__.py`, `output_processor.py`, `outputs.py`, `v1/outputs.py`, `entrypoints/...` |
| removed from 0221 | the V1 runner and V1 sampler changes, the frontend hold-back, `rollback_output_tokens` and the prefix-cache forget, the grammar rewind, the sync-mode event copies, 0221's three test files |

## Size

| | patch lines | source (+/-) | tests (+) |
|---|---|---|---|
| v1, 82eff9023 | 3,436 | 1,487 / 34 (26 files) | 853 |
| v2 first build, ad9059483 (on v1's branch) | 5,502 | 2,465 / 74 (30 files) | 1,752 |
| v2, 7f4c7dc8e (one commit on `r6x-candidate`) | 5,063 | 2,069 / 59 (29 files) | 1,752 |
| v2, 2334b92c0 (the column capture in the SM120 model) | 5,069 | 2,073 / 59 (29 files) | 1,752 |
| v2, e1c474dca (warmup rows neither logged nor counted) | 5,126 | 2,083 / 59 (29 files) | 1,791 |
| v2, 0d63e52a3 (each event logs its largest flagged value) | 5,164 | 2,107 / 59 (29 files) | 1,805 |
| v2, 616d1a28d (threshold 85,000 from the survey) | 5,171 | 2,108 / 59 (29 files) | 1,809 |

Cut from the first build, with the same tests passing: the flags, the cut and the merge of the
candidates became torch operations (three Triton kernels and their warmup declarations gone; the
worker's `sink.py` 1,050 to 771 lines); the V1 runner is untouched (its startup refusal went); the
startup comparison of the settings across tensor-parallel ranks went (every rank reads the same
environment; the lab flag check stays); `check_stop` is no longer refactored (`stops_on_next_token`
is a 12-line helper beside it); one scheduler log line per sink event; the held token's logprobs are
one row table per request. v2 stays larger than v1 because draft zero is scheduled and accounted in
both schedulers, the policy and the Gumbel-replay alternates run on the device, and its tests drive
the real kernels.

## Tests run

[runs/run-tests.sh](runs/run-tests.sh) on dgx4 (idle, not in the serving set): throwaway containers
from the promoted r6x image, with the candidate's changed files mounted over the image's sources. The
image's vLLM tree is the candidate's base (f3b58ed9d, tree 4d6851c8). No network; dgx4's own GPU.
[runs/tests.txt](runs/tests.txt) is the record.

| Suite | Where | Result |
|---|---|---|
| `tests/v1/core/test_sink_draft_zero.py` (new). The real scheduler with a scripted worker playing the V2 runner's side, sync and async: the collapse by draft-zero rejection (0, 1, 5 drafts) and by drafter cut; fallthrough; a rejection mid-batch with its neighbours unchanged; zero drafter tokens; the accounting; held EOS, stop and max-tokens tokens; abort while holding; logprobs; n > 1; stream intervals 1 and 3; the grammar; the async back-fill; preemption with a stale output; the encoding of `sink_metrics` | CPU | 43 passed |
| `tests/v1/worker/test_gpu_sink.py` (new). The detector on the reference numbers; the alternates as the draw's next samples (greedy and random); cut and restore (full and shard); greedy shards against full logits; the policy; the request-state update after a rejection; DSpark inputs with a rejected request mid-batch; the collapse sequence across steps through the real sampler and rejection sampler (greedy and random, block verification) with its batch neighbours' tokens and request state unchanged | GPU | 17 passed |
| V4.1 and V4 tokenizers and parsers | CPU | 235 passed |
| existing: scheduler output, repetition stop, output processor | CPU | 27 passed, 2 failed, 33 errors: the untouched image's |
| existing: Gumbel sampling, rejection sampler (chunking, int64), sampler flags, dead rows, draft-token handler, verified draft counts, input batch v2, rejection-sampler utilities, synthetic rejection, DFlash inputs, adaptive verification, V1 rejection sampler and sampler (14 files) | GPU | 278 passed, 5 failed, 1 skipped: the untouched image's |
| existing: `test_scheduler.py`, `test_async_scheduler.py` | GPU container | 180 passed, 40 failed: the untouched image's |

The 80 failures and errors are the same tests on the untouched image ([runs/offline-failures.txt](runs/offline-failures.txt)):
73 offline (no tokenizer or uncached model configs), 2 async tests without pytest-asyncio, and 5 that
fail in the image as built.

## Screen (owner, 2026-10-19: a short screen first, then the gate)

[screen.json](screen.json) then [probes.json](probes.json), queued into one lab window on the TP3
triangle. The arms mount the candidate's 29
changed vLLM files ([overlay/](overlay), now vLLM 616d1a28d; the screen ran 2334b92c0) over the r6x
image, whose tree is the
candidate's base: every change is Python, so this is the candidate's image.

| Arm | What it is |
|---|---|
| [r6x-dev.json](r6x-dev.json) | production TP3 (r6x at 1M) with the dev-mode RPC for the engine grid |
| [correction-dev.json](correction-dev.json) | r6x-dev with the overlay, detector and rejection on |
| [detector.json](detector.json) | production with the overlay and the detector alone |
| [correction-ranks.json](correction-ranks.json) | production with the overlay, rejection on and `VLLM_SINK_DETECTOR_CHECK_RANKS=1` |

1. **Speed and identity** (gate items 1, 5, 8 in short): the `screen-tp3` suite with the companion
   suites `single-stream` and [logprobs.json](logprobs.json) (closed loops at 1 and 8 streams with
   five top logprobs) on correction, r6x, correction again. The screen's model step scores each
   arm's text against the first arm's; the correction arm's must equal r6x's (no prompt there
   collapses). Single-stream gives time to first token.
2. **The collapse and the ranks** (gate items 2 and 6): [probes.py](probes.py) on correction-ranks,
   detector and r6x: the 16 answers, `usage.sink_rejection`, the `vllm:sink_*` counters and each
   rank's detector log line.

**First attempt, 2026-10-19 21:36-21:43 UTC (#437, vLLM 7f4c7dc8e):** both overlay arms failed to
boot on every rank: `ValueError: The sink detector needs the model to keep its residual column`. On
SM120 (GB10) vLLM loads DeepSeek V4.1 from `vllm/models/deepseek_v4_1/`; the candidate had put the
column capture in `vllm/models/deepseek_v41/nvidia/model.py`, the generic NVIDIA model that never runs
on the Sparks (v1 had the same gap). No unit test loads the model. The window closed on a request and
restored production (`doctor --live` OK). vLLM 2334b92c0 moves the capture into the SM120 model, after
the last layer's `mhc.post` and before the collapse and norm, gathered like the hidden states (static
sequence parallelism, per-forward prefill SP, CED row compaction), and leaves the generic model
untouched. The suites on dgx4 give the same results ([runs/tests.txt](runs/tests.txt)); runs `-screen2`
and `-probes2` repeat the screen.

### Results (runs `-screen2` and `-probes2`, 22:01-23:35 UTC, vLLM 2334b92c0)

Tables: [runs/screen2/](runs/screen2) (screen-tp3, single-stream and logprobs tables, the three
probe outputs, and the scheduler's flagged events during the repeat arm). Verdicts are against the
run's own repeat (correction, then correction-again).

| | correction vs r6x | repeat | verdict |
|---|---|---|---|
| engine grid, geometric mean of 15 shapes | +0.17% | -0.38% | same |
| 8 streams, output tok/s | +0.12% | +0.31% | same |
| one stream, output tok/s | -1.20% (±7.0%) | +0.05% | same |
| logprobs (5), 1 / 8 streams | -1.49% / -0.10% | -0.04% / -0.03% | same |
| prefill 32K / 256K | +0.16% / -0.11% | -0.69% / -0.43% | same |
| time to first token, p50, 1 / 8 streams | 0.124 -> 0.164 s / 0.358 -> 0.462 s | level | one decode step (26.6 and 9.2 steps/s) |

- **Model behaviour (128 prompts, greedy, 8 streams):** 126/128 responses identical to r6x, and the
  teacher-forced distributions identical everywhere (KL 0): the forward pass is r6x's. The other two
  diverge at reply tokens 230 and 236, where the correction emitted the second-ranked token (-3.0
  against -0.23, -2.9 against -0.06). Both repeats of the correction arm give the same text.
- **They are false positives.** r6x's continuations read normally ("between 1.0 and 2.5 m above
  chart datum", "2023: 96% contained OSS"). DeepSeek's CPU reference, teacher-forced through r6x's
  replies ([analysis/false_positive_rows.txt](analysis/false_positive_rows.txt), cases beside it):
  the flagged rows are a lone space before digits (47,360) and `202` (51,968), each predicting its
  next digit at -0.52 and -0.04; other number rows in the same replies reach 30,848-39,168. In the
  prompt, ` the` (effort line) and ` you` reach 124,416 and 115,712 and predict `\n` / `#` / `<`, the
  BOS-like distribution of the collapse rows (108,032-125,440). The column marks real sinks; 50,000
  sits inside ordinary text.
- **During the repeat arm's workloads** ([runs/screen2/flagged-events.txt](runs/screen2/flagged-events.txt)):
  17 flagged events in about 25 minutes of normal traffic, mostly drafter cuts; the closed loops
  replay the same prompts, so the same positions recur.
- **The collapse ([probes](runs/screen2), 16 prompts, thinking high, temperature 0):**

  | prompt | r6x and the detector arm | correction arm (rank check on) |
  |---|---|---|
  | `Say hi.` | `Hi! 👋 How can\n` and stop | `Hi! 👋 How's your day going?` (1 draft-zero rejection, 1 draft cut) |
  | `Thanks!` | `Let me know if there\n there's anything else` | `Let me know if you need anything else — happy to help.` (1 rejection) |
  | `ok` | `e\n first message` in the reasoning | clean reasoning, `Got it — what would you like help with?` (1 rejection, 3 cuts) |

  The other 13 answers are byte-identical to r6x on both candidate arms, unflagged; the detector arm
  is byte-identical to r6x on all 16 and flags the same three prompts.
- **Ranks agree:** `VLLM_SINK_DETECTOR_CHECK_RANKS=1` never raised, and every rank's log line read the
  same counts (correction 9 rows, detector 5). The scheduler counted 2 draft-zero rows fewer on both
  arms: most likely in-flight async steps of requests that had already ended (their outputs are
  dropped), not yet verified.
- **Boots** 90.9-92.8 s, as r6x's 89.8-92.0 s.
- **Startup log lines:** the warmup's synthetic requests hold token 0, DeepSeek V4.1's BOS, as draft
  zero, so each warmup pass logged flagged rows before the server was ready; the counts were reset,
  but the last line before the reset stayed. vLLM e1c474dca keeps the warmup silent (`warming`, then
  `end_warmup()`), with a GPU test; overlays and patch carry it.

## Survey (owner, 2026-10-20: survey high rows, then set the threshold from the data)

vLLM 0d63e52a3 adds to each sink event the largest |residual[2455]| among the request's flagged rows
that step, printed in the scheduler's `Sink:` line. [survey.json](survey.json) boots
[survey-detector.json](survey-detector.json), the detector alone at threshold 20,000 (decoding is
r6x's), and runs [survey.py](survey.py) three times: servebench's 256 default prompts at the server's
default sampling, 128 of the other template-topic pairs greedily, and the 16 collapse probes at
thinking high, 1,024 tokens each at 8 streams. It joins every `Sink:` line to its response with the
tokens around the flagged position. The highest normal rows then go through DeepSeek's CPU reference
(as [analysis/false_positive_rows.txt](analysis/false_positive_rows.txt)) to read their predictions.

### Survey results (run `-survey`, 2026-10-20 06:39-07:21 UTC, vLLM 0d63e52a3)

[runs/survey/](runs/survey) holds the three outputs; [analysis/survey_tail.py](analysis/survey_tail.py)
gives [analysis/survey_tail.txt](analysis/survey_tail.txt). The first boot at 06:36 was stopped by
dgx1's startup memory guard (MemAvailable 34 MB under its 5 GiB floor); the immediate retry booted in
91.8 s.

| ordinary text, 358,785 tokens (384 responses) | 20,000 | 30,000 | 40,000 | 50,000 | 60,000 | 70,000 |
|---|---|---|---|---|---|---|
| rows at or above | 526 | 285 | 116 | 24 | 3 | 0 |

Each count is one step's largest flagged row (the Sink: line reports one value per request and
step), so it counts steps; a per-row count could only be higher where two rows of one step pass a
bin. Every value is a real served row's: the attribution run below names the rows of the top ones.

- **Ordinary text peaks at 62,720** ("H2‖ 1.3%"), then 61,184 (`max‖_n=2`) and 60,928 ("structure
  ‖3-4 paragraphs"). Every row above 50,000 is a number context: times, amounts, percentages,
  `NUMERIC(8,2)`, ranges.
- **The collapse rows read 116,736-118,784 served** (`Say hi.`, `Thanks!`, `ok`), and nothing else
  in the probes reaches 20,000.
- **The tail falls about 8x per 10,000** from 50,000 to 60,000. If it keeps falling so, a threshold
  of 80,000 flags about one ordinary row per 8 M generated tokens, 85,000 one per 22 M, 90,000 one
  per 61 M.
- **DeepSeek's reference on the top rows:** the reply's high rows (spaces before digits) predict
  their digits confidently, as the survey's ordinary rows should. An earlier version of this record
  said served values "ran 1.1-1.6x the reference's" on number rows; that rested on three rows of one
  reply whose attribution was inferred, because the Sink: line then gave the step's next output
  position with the step's largest value, not that value's row. vLLM 5ac8364ec names the row; the
  attribution run below confirms those three rows and finds the ratio running both ways (0.59-1.61).

**Threshold: 85,000** (vLLM 616d1a28d), the geometric midpoint between the largest ordinary row
(62,720) and the smallest sunk row seen anywhere (108,032, DeepSeek's reference): about 1.35x of
margin on each side.

### Re-screen at 85,000 (runs `-screen3` and `-probes3`, 2026-10-20 07:42-09:15 UTC, vLLM 616d1a28d)

Tables and probes: [runs/screen3/](runs/screen3). Verdicts against the run's own repeat.

| | correction vs r6x | correction-again vs r6x | repeat | verdict |
|---|---|---|---|---|
| greedy responses identical (128) | 128/128 | 128/128 | 128/128 | same text as r6x |
| engine grid, geometric mean of 15 shapes | +1.13% | +0.47% | -0.65% | worse (first screen +0.17% / -0.21%) |
| 8 streams | +0.23% | +0.18% | -0.06% | same |
| one stream | -1.22% (±7.0%) | | -0.25% | same |
| logprobs (5), 1 / 8 streams | -1.54% / -0.06% | | -0.37% / -0.23% | same |
| prefill 32K / 256K | -0.31% / -0.24% | | -0.13% / -0.11% | same |
| time to first token, p50, 1 / 8 streams | 0.124 -> 0.162 s / 0.357 -> 0.472 s | | level | one decode step |

- **Model behaviour:** 128/128 greedy responses identical to r6x on both correction arms, rank-1
  fraction 1.0: the number-token false positives are gone.
- **Collapse probes:** `Say hi.` (`Hi! 👋 How's your day going?`), `Thanks!` (`You're welcome! 😊
  Let me know if you need anything else — happy to help.`) and `ok` (`Got it! What would you like help
  with?`) flagged and cleaned; the other 13 byte-identical to r6x; the detector arm byte-identical to
  r6x on all 16. Every rank logged the same counts (8 rows on correction, 5 on detector), and
  `VLLM_SINK_DETECTOR_CHECK_RANKS=1` never raised.
- **Engine grid:** the per-shape changes are largest at the smallest shapes (+1.0% to +2.3% at 1-3
  rows), as a fixed cost of about 0.2 ms a step would be; r6x itself moved -0.45% between the two
  screens, and the code added between them (each row's value for the log, a sixth event field) is two
  small tensor operations a step. Not settled; see Open.
- **Startup:** the correction arm's log shows only the detector's configuration line: the warmup's
  rows are no longer logged (e1c474dca).

### x203 against the reference, position by position

Owner's question: do the served engine and DeepSeek's reference part ways past the 128-token window,
where the compressed KV and the sparse indexer run? Production r6x scored x203's prompt and greedy
reply (738 ids, reproduced on production) with `prompt_logprobs` 5, a teacher-forced prefill, against
the reference's teacher-forced run ([analysis/x203_gap.py](analysis/x203_gap.py),
[analysis/x203_gap.txt](analysis/x203_gap.txt)).

| | median gap (nats) | p90 | max | top-1 agreement | gaps over 3 nats |
|---|---|---|---|---|---|
| positions below 128 (127) | 0.047 | 0.311 | 5.43 | 89.8% | 1 (position 16, inside the prompt's effort line) |
| positions from 128 (610) | 0.026 | 0.215 | 1.41 | 93.1% | 0 |

No jump at 128: past the window production's prefill follows the reference as closely as before it,
within the FP8 noise floor (gaps up to about 3 nats). The predictions made from x203's three high
sink-column rows match to 0.002-0.039 nats.

### Row attribution (run `-attribution`, 2026-10-20 10:22-10:25 UTC, vLLM 5ac8364ec)

The detector-only arm at 20,000 reran d027, d045 and x203 greedily; each Sink: line now names the row
holding its value. Every served prompt and reply matched the reference's ids up to each compared row
([analysis/attribution_compare.py](analysis/attribution_compare.py),
[analysis/attribution_compare.txt](analysis/attribution_compare.txt), [runs/attribution/](runs/attribution)).

| reply, row (flagged input in brackets) | served | reference | served / reference |
|---|---|---|---|
| x203 688, `H2[ ]1.3%` | 62,720 | 38,912 | 1.61 |
| x203 681, `H1[ ]1.0%` | 48,384 | 33,280 | 1.45 |
| x203 609, `Hive[ ]3` | 51,200 | 46,848 | 1.09 |
| d027 155, `amplitude[ ]1.5 m` | 41,728 | 38,400 | 1.09 |
| d027 173, `at least[ ]9.2` | 23,808 | 30,848 | 0.77 |
| d027 175, `at least 9[.]2` | 22,016 | 14,144 | 1.56 |
| d027 230, `between[ ]1.0` | 51,456 | 47,360 | 1.09 |
| d045 158, `OpenLogic [202]4` | 23,040 | 39,168 | 0.59 |
| d045 236, `high-risk? [202]3` | 53,504 | 51,968 | 1.03 |

- The survey's 62,720 is x203's row 688 (the space before `1.3%`): the earlier comparison read the
  right row, but nothing logged could show it then.
- Served against reference runs 0.59-1.61, median 1.09, both ways: on number rows past the window the
  sink column of the served decode rows wanders by up to ±60% from the reference's prefill rows, while
  the predictions agree (above). The collapse rows (positions 39 and 74, inside the window) agreed within
  about 8%.
- **The threshold stands.** It was set on served values on both sides: the ordinary tail's largest row
  (62,720, now attributed) and the served collapse rows (116,736-118,784); the re-screen at 85,000 gives
  128/128 identical.

## GPU gate (owner, 2026-10-20: run the full gate; the grid settles the step cost)

Threshold 85,000, vLLM 5ac8364ec (each `Sink:` line names its row). Three specs, queued into one window:

- [gate-release.json](gate-release.json) (items 1, 5, 8 and the step cost): `release-tp3` with the
  companions `single-stream`, `sampled` and [logprobs.json](logprobs.json) on
  [correction-dev.json](correction-dev.json), [r6x-dev.json](r6x-dev.json),
  [detector-dev.json](detector-dev.json) and correction again. The release suite's model step scores
  every arm's text against the first; a read-only watcher on dgx1 records every `Sink:` line and the
  `vllm:sink_*` counters per arm, which must stay at zero on the release prompts. The engine grid gives
  each arm's step cost; it decides the two held optimizations (below, Open).
- [gate-probes.json](gate-probes.json) (items 2, 3, 4, 6, and the speculation side of 7):
  [probes.py](probes.py), [batch.py](batch.py) (the 16 probes alone, then at eight streams with
  staggered arrivals: TP3 serves eight sequences, so eight streams stand in for the plan's sixteen),
  [longctx.py](longctx.py) (`Say hi.` and `Thanks!` after a 64K- and a 512K-token conversation, the
  same contexts on every arm, so r6x shows which still collapse) and [identity.py](identity.py), on
  [correction-ranks.json](correction-ranks.json) (rank check on), [detector.json](detector.json) and r6x.
- [gate-nospec.json](gate-nospec.json) (item 7): probes and identity on
  [correction-nospec.json](correction-nospec.json), the correction arm without `--speculative-config`,
  on port 8001 so returning agent traffic cannot reach it; [analysis/gate_identity.py](analysis/gate_identity.py)
  compares its tokens with the speculative correction arm's.

### Gate results (2026-10-20 13:31-18:50 UTC, vLLM 5ac8364ec)

Tables, probe outputs and the watcher's sink lines: [runs/gate/](runs/gate). Release verdicts are
against the run's own repeat (correction, then correction again).

| | correction vs r6x | detector vs r6x | correction again vs r6x | repeat | verdict |
|---|---|---|---|---|---|
| greedy release responses (256) | identical, KL 0 | identical | identical | | same text |
| engine grid, geometric mean of 15 shapes | -0.28% | +0.00% | -0.65% | -0.37% | same |
| 8 / 4 streams | +0.14% / -0.12% | -0.08% / -0.04% | +0.18% / -0.12% | +0.04% / +0.00% | same |
| Poisson arrivals | +0.06% | +0.01% | +0.12% | +0.06% | same |
| prefill 32K / 128K / 256K / 500K | -0.13% / -0.28% / -0.18% / -0.24% | | | -0.06% to +0.11% | same |
| one stream | -1.26% (±6.9%) | | | -0.24% | same |
| sampled, 16 / 8 / 1 streams | +0.37% / +0.21% / -0.77% | -0.49% / -0.11% / -0.35% | | | same |
| logprobs (5), 1 / 8 streams | -1.56% / -0.18% | | | -0.39% / +0.38% | same |
| time to first token p50, 1 / 4 / 8 streams | 0.124 -> 0.162 / 0.256 -> 0.331 / 0.358 -> 0.470 s | unchanged | as correction | | one decode step |

1. **Release prompts, zero flags: pass.** All arms' release text equals r6x's; the watcher read
   `vllm:sink_*` 1,686 times on the correction arms and 840 on the detector arm during the release
   suites: every counter stayed at 0, and no `Sink:` line came from them.
2. **The collapse: pass.** Correction arm (rank check on): `Say hi.`, `Thanks!`, `ok` flagged and clean,
   the other 13 byte-identical to r6x; the detector arm flags the same three and decodes as r6x
   (`Thanks!` collapsed, `Say hi.` stops at `How can\n`).
3. **Concurrent batch: pass** (eight streams, TP3's limit): on every arm the 16 answers and their flags at
   eight streams with staggered arrivals equal the one-stream answers; the correction arm stays clean.
4. **Long context: not exercised.** After a 64K- (60,756 tokens) and a 512K- (494,952 tokens)
   conversation, `Say hi.` and `Thanks!` answer normally on r6x (`Hi!`, `You're welcome! ...`), with
   no flag on any arm: the collapse needs its short context, so no rejection deep in the compressed and
   sliding-window caches was seen.
5. **Time to first token: pass.** +38 ms at one stream (26.6 steps/s), +75 ms at four, +112 ms at
   eight: one decode step each.
6. **Ranks agree: pass.** `VLLM_SINK_DETECTOR_CHECK_RANKS=1` ran through the probes, the batch, the
   long contexts and the identity run without raising; every rank's log read the same counts
   (correction 8 rows, detector 5, no speculation 6).
7. **Speculation does not change greedy tokens: pass.** [analysis/gate_identity.py](analysis/gate_identity.py):
   144/144 replies (128 screen prompts, 16 probes) identical with and without `--speculative-config`.
   Without speculation the collapse is cleaned by draft-zero rejections alone (`Say hi.`: 2 rejections;
   with speculation 1 rejection and 1 draft cut): the first serving run of that path.
8. **Logprobs: within the noise.** One stream -1.56% (±7.7%), eight -0.18% (±2.5%).

**The held optimizations are not applied** (owner's rule: only if the grid shows their cost outside the
release noise): the correction arm's step cost is level with r6x (-0.28%, -0.65% again), and the logprobs
companion is level too. The second screen's +1.13% / +0.47% grid reading was session noise.

The items as first planned:


Build the image from the series with this 0221, then one lab window on the TP3 triangle with three
arms: **r6x**, **detector** (`VLLM_SINK_DETECTOR_COLUMN=2455`) and **correction** (also
`VLLM_SINK_REJECTION=1`).

1. **Release prompts, zero flags.** `scripts/lab.py` measure with `"suite": ["release-tp3",
   "single-stream", "sampled"]` on all three arms. Tokens must be bit-identical across the arms
   (greedy and sampled: the sampled suite seeds every request, so r6x draws with the same Gumbel
   kernels the correction arm keeps), and `vllm:sink_flagged_rows` read from `/metrics` must be 0
   after each candidate arm. The engine grid gives the step cost of each arm; the budget is the
   release noise.
2. **The collapse.** [probes.py](probes.py) (the thinking-collapse experiment's 16 prompts) at
   temperature 0, thinking high, against the correction arm. `Say hi.` and `Thanks!` must show
   `usage.sink_rejection.flagged_draft_zero >= 1` (or a draft cut) and a clean answer (no
   `How can\n can`, no `there\n there's`, no `<ds_safety>`). The other 14 prompts must be byte-identical
   to the r6x arm's answers with zero flags. On the detector arm the same two prompts
   must flag and collapse as r6x does.
3. **Concurrent batch.** `Say hi.` and `Thanks!` interleaved with the other 14 prompts at c16, staggered
   arrivals, on the correction arm. The collapsing pair is clean, the other 14 equal their c1 answers on
   the correction arm, and the flag counts equal the c1 run's.
4. **Long context.** A sink after a long prompt, so a rejection rewrites a slot deep in the compressed
   and sliding-window caches: `Say hi.` as the last turn after a 64K- and a 512K-token
   conversation. Find a context that still collapses on r6x first; the correction arm must answer
   cleanly with the flag counted.
5. **Time to first token.** The single-stream suite's TTFT on the correction arm within one decode step
   of r6x.
6. **Ranks agree.** One correction-arm run of items 2-3 with `VLLM_SINK_DETECTOR_CHECK_RANKS=1` (it
   raises on any disagreement), and the per-rank log lines' counts equal on all three ranks in every
   run.
7. **Speculation does not change greedy tokens.** The correction arm with and without
   `--speculative-config`, temperature 0, on the release prompts and the 16 probes: identical tokens,
   with `Say hi.` and `Thanks!` flagged and clean in both.
8. **Logprobs.** The release-tp3 c1 and c16 cells with `logprobs=5` on r6x and the correction arm.
   With logprobs, every step computes the logprob and rank of the three alternates of every request
   (several passes over the vocabulary per request), used only on a rejection; this is where a step
   cost should show. Within the release noise, it stays as it is; otherwise those passes are fused
   into one.

The other per-step costs of the correction arm (the candidates kernel, one all-gather of 8 floats per
request on the all-greedy path, about 25 small launches, the scheduler's per-request event
conversion) are read from item 1's engine grid. The first boot with the detector recompiles: its
column is part of the compile cache key.

## Open

- **The threshold** is 85,000 from a 358,785-token survey: about one false positive per 22 M tokens
  if the tail keeps falling as measured, and 0.73x of the smallest served collapse row. Three served
  collapse rows is a small sample of the sink side, and served sink values on number rows wander ±60%
  from the reference's (row attribution); a sink row past the window reading 0.6x would sit near the
  threshold.
- **Long context.** No 64K or 512K context collapsed on r6x, so a rejection deep in the compressed and
  sliding-window caches has unit tests behind it but no serving run.
- **Served decode against reference prefill.** The sink column of served decode rows differs from the
  reference's prefill rows by up to ±60% on number rows while predictions agree; a served prefill of
  the same rows (a debug read of the column for every prompt row) would split decode from prefill.
- The per-rank counts read 2 draft-zero rows more than the scheduler's on both probe arms; check that
  they are the discarded in-flight steps of finished requests.

- **Cross-rank agreement is an assumption.** Every rank computes the flags from its own copy of the
  residual and decides the same rows only if the copies agree bit for bit after the all-reduces (or no
  value sits at the threshold; sunk and clean rows are a decade apart). A disagreement would diverge
  the ranks' request states. Item 6 of the gate checks it; the serving path adds no collective for it.
- A sunk final token (EOS, a stop token, the max-tokens or max-model-len token) is committed unchecked.
- A draft cut under random sampling counts the sunk draft whether or not the acceptance test would also
  have rejected it (as in v1).
- The threshold rests on the CPU reference: 4 sunk sampled rows and 2,174 clean ones over the effort
  variants' 40 runs (2,138 generated rows and 40 last prompt rows), plus v1's dumps. The gate's
  detector arm is the first count over serving traffic.
- The no-speculation path and DFlash are covered by the CPU and kernel tests and gate item 7, not by a
  serving run.
- Two cheap optimizations are held until the gate has measured their costs (owner, 2026-10-19):
  fusing the alternates' logprob and rank passes (item 8), and skipping the host event conversion,
  which walks every request each step, when no row is flagged.
