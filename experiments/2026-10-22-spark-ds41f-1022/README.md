# spark-ds41f-1022: r6x with the broadcast fix, late admission, sink rejection and sparknet 0.5.0

The next release candidate on top of r6x, which serves the TP3 triangle (dgx1-3) at 1M context.
It assembles four improvements that were each proven alone on 2026-10-20/21 and are not in
production, gates the combination with one focused window, then compares it with r6x as the
promotion gate. Named per the release naming adopted for the next release: `spark-ds41f-MMDD`
(the candidate's own name until the owner promotes it).

**Status:** series built and unit-tested (2026-10-22); sparknet 0.5.0 cut locally, waiting
for the owner to push; image, gate and release comparison pending.

## Contents

| Piece | Where it comes from | What it was shown to do alone |
| --- | --- | --- |
| vLLM 0185, the TP group's broadcast on the RoCE policy | [2026-10-20-broadcast-oneshot](../2026-10-20-broadcast-oneshot/README.md), merged #443 | TP3 gate: outputs identical to r6x; release comparison level on every metric |
| vLLM 0221, sink detection and draft-zero rejection | [2026-10-19-sink-rejection-v2](../2026-10-19-sink-rejection-v2/README.md), gate passed #452 | `Say hi.` / `Thanks!` corrected with the ranks agreeing; 256/256 release text identical; zero flags on the release suites; throughput level; time to first token one decode step longer |
| vLLM 0222, late admission at decoder layer 30 | experiments/2026-10-21-late-admission (PR #454; analysis in PR #453) | TP3 screen: TTFT p50 −26% at four streams, −30% at eight; one stream and prefill level; outputs identical |
| vLLM 0223, sink rejection on by default for DeepSeek V4.1 | this record | the owner's decision of 2026-10-22, below |
| vLLM 0224, late admission with sink rejection through the engine-core loop | this record | tests only |
| sparknet 0.5.0, the wider one-shot data phase | dgx-spark-networking main 8618973 and the 0.5.0 release commit | TP3 engine grid decode step −1.6%, outputs identical |

The gated 0221 is vLLM 5ac8364ec (616d1a28d with each sink event naming its row); its patch
file is unchanged. 0222 is re-exported on 0221 (same change, new context).

### Sink rejection: optional, on by default (vLLM 0223)

Owner, 2026-10-22: sink rejection stays switchable, so the code that exists only for this
checkpoint is marked for removal when a later DeepSeek no longer needs it, and is on by default.

- `VLLM_SINK_REJECTION` stays the switch. Unset, it is on for the architectures listed in
  `vllm.v1.sample.sink.SINK_CHANNELS` and off for every other model; `0` turns detector and
  rejection off.
- `SINK_CHANNELS` is the one place the numbers live: `DeepseekV41ForCausalLM` → column 2455,
  threshold 85,000 (the survey's geometric midpoint). Nothing else names them.
- `VLLM_SINK_DETECTOR_COLUMN` and `_THRESHOLD` become overrides with no defaults of their own;
  with rejection off an explicit column runs the detector alone (lab surveys).
- The code stays where 0221 put it: `vllm/v1/sample/sink.py`, `vllm/v1/worker/gpu/sample/sink.py`
  and the scheduler's draft-zero hook. The V1 runner, which has no detector, now refuses to start
  while the detector is on, so the scheduler never holds tokens that nothing verifies.
- `VLLM_SINK_REJECTION` left the compile-key exclusions: it now decides whether the forward
  keeps the column.

The recipe sets nothing for it.

### Late admission with draft zero (vLLM 0222 + 0221)

0221 and 0222 applied in order without conflicts. They meet in the engine-core loop: late
admission moves the decision of step N+1 to the moment step N reaches layer 30, and sink
rejection schedules each decoding request's held token as draft zero of N+1. N's output is
unprocessed when N+1 is decided either way, so the held tokens and placeholders are the same;
what the later decision changes is which arrivals and aborts it sees. 0224 tests exactly that
through the real `step_with_batch_queue`, the async scheduler and the sink tests' scripted
V2-runner worker:
- an arrival beside a held draft zero joins one step earlier with late admission; both replies
  are identical with and without it (0 and 3 drafter tokens, arrivals at three points);
- an abort that arrives while a token is held ends the request without emitting a token it
  never verified, its neighbour unchanged;
- a structured-output step sampled through the engine core's deferred path back-fills draft
  zero ahead of the drafts, and the grammar accepts committed tokens only, with late admission
  waiting on that deferred step.

## Build inputs

- vLLM: r6x's series ([vllm/series](vllm/series)) plus 0185, 0221, 0222, 0223, 0224 on base
  19f2c20e; patch head c6a567ada, tree 32ae33c9 (`git am --committer-date-is-author-date` with
  the build's committer reproduces it). Working branch `next-release` in
  ~/projects/vllm-ds41-next (local, as vLLM working branches are).
- sparknet: 0.5.0, release commit ca660575 on main 8618973 (tree 9d486705); to be pushed and
  tagged `v0.5.0` on the owner's word.
- Everything else as r6x: [upstreams.lock.json](upstreams.lock.json),
  [source.json](source.json).
- Image `vllm-ds41f-kkref:19f2c20ed4d6-1022` (the live-name rename stays a separate step).

## Unit tests (2026-10-22, dgx4)

[runs/run-tests.sh](runs/run-tests.sh): throwaway containers from the r6x image on dgx4 (idle,
not in the serving set), the series' changed files mounted over the image's sources (whose tree
is the series' base). No network.

| Suite | Result |
| --- | --- |
| 0185: `tests/distributed/test_roce_policy_no_fallback.py` (CPU) | 13 passed |
| 0221/0223: `tests/v1/core/test_sink_draft_zero.py` (CPU), the model-keyed default included | 48 passed |
| 0222: `tests/v1/engine/test_late_admission.py` (CPU) | 5 passed |
| 0224: `tests/v1/engine/test_late_admission_sink.py` (CPU) | 18 passed |
| 0221: `tests/v1/worker/test_gpu_sink.py` (GPU) | 19 passed |
| 0222 + 0224: both late-admission files and the combined one, captured graph included (GPU) | 29 passed |
| existing CPU suites: scheduler output, repetition stop, output processor | 27 passed, 2 failed, 33 errors |
| existing batch-queue, sampling, verification and drafter-input suites (GPU) | 111 passed, 5 failed, 1 skipped |
| `test_scheduler.py`, `test_async_scheduler.py` (GPU container) | 180 passed, 40 failed |

Every failure and error in the existing suites is one of the 80 the untouched r6x image shows
offline ([../2026-10-19-sink-rejection-v2/runs/offline-failures.txt](../2026-10-19-sink-rejection-v2/runs/offline-failures.txt)):
the same 80 test ids, none added. The sink and combined suites build their scheduler from a local
OPT config and need no Hugging Face cache or tokenizer, so the release tests (offline) run them.

`scripts/release_tests.py check`: the five new or changed test files are classified as gates
([config/release-tests.json](../../config/release-tests.json)); 65 gate files, 0 problems.

## The focused gate (one window, TP3 triangle)

[gate-tp3.json](gate-tp3.json): the image job (build on dgx2, copy to dgx1 and dgx3 over the
fabric and to dgx4 over the LAN for the release tests), then one boot of
[spark-ds41f-1022-tp3.json](spark-ds41f-1022-tp3.json), the recipe's TP3 sizing unchanged
(1,048,576 tokens, 4.90625 GiB KV pool) under the unchanged 5/3 GiB memory guard:

| Step | Pass |
| --- | --- |
| `token_determinism_text.py --wide`, with log probabilities and without | identical, 0 token and 0 log-probability differences, hashes r6x's TP3: 62313cb5 / 9742ac2c and 4d143977 / 12fd8257 |
| `ttft_trace.py` (PR #453): twelve isolated prompts, c4 for 120 s, c8 for 60 s | TTFT p50 at the late-admission screen's level (190.9 / 249.0 ms; r6x 258.8 / 353.9), no request waiting a step that did not carry it |
| [gate-suite.json](gate-suite.json): one stream 120 s, cold 32K and 131K prefill x3 | level with the screen's r6x arm (56.7 tok/s; 4461 / 4386 tok/s) |
| `mem_snapshot.py` | MemAvailable once serving, against r6x's 6.70 / 8.04 / 8.11 GiB |
| [sink_counters.py](sink_counters.py), after the ordinary text | zero flagged rows |
| `probes.py --nodes dgx1,dgx2,dgx3` (16 prompts, thinking at effort high, temperature 0) | `Say hi.` and `Thanks!` corrected, no collapse marker, every rank's count equal |

The release tests ([bundles/release-tests](bundles/release-tests), 65 gate files) run on dgx4
with `scripts/lab.py kernel-local` once the window has copied the image there.

## Release comparison (promotion gate)

[release-tp3.json](release-tp3.json): `release-tp3` with `single-stream` and `sampled` as
companions; arms candidate, r6x, candidate again. The r6x reference arm
[r6x-tp3-dev.json](r6x-tp3-dev.json) extends r6x's recipe frozen under [r6x/](r6x/), which is
also the rollback.

## Not covered

- The four-node ring: [spark-ds41f-1022-tp4.json](spark-ds41f-1022-tp4.json) is ready, but the
  ring is not cabled. TP4 stays unqualified on this release until a ring release run.
- dgx1's startup peak: a production-sized pool boot can fail the 5 GiB startup guard from
  boot-to-boot spread alone (2026-10-21, 12.7 MiB short on one boot). The guard stays as it is;
  a separate experiment works on lowering the peak.
