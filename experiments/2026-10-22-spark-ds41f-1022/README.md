# spark-ds41f-1022: r6x with the broadcast fix, late admission, sink rejection and sparknet 0.5.0

The next release candidate on top of r6x, which serves the TP3 triangle (dgx1-3) at 1M context.
It assembles four improvements that were each proven alone on 2026-10-20/21 and are not in
production, gates the combination with one focused window, then compares it with r6x as the
promotion gate. Named per the release naming adopted for the next release: `spark-ds41f-MMDD`
(the candidate's own name until the owner promotes it).

**Status (2026-10-22):** the focused gate, the release tests (1,781 passed on image `-1022-2`)
and the owner's trimmed promotion comparison against r6x have passed: engine grid −1.1%, eight
streams, one stream and prefill same against the run's own repeat, outputs identical. Time to
first token is level with r6x at four and eight streams and one decode step longer alone (late
admission pays for sink rejection's held step; [Results](#results)). **Promotion is on hold:**
prefill reads about 1% lower in every pair and in the gate, and the owner asked for that to be
fixed before the release. The fix is traced and made in a separate experiment; the candidate takes
it, then promotion follows.

## Contents

| Piece | Where it comes from | What it was shown to do alone |
| --- | --- | --- |
| vLLM 0185, the TP group's broadcast on the RoCE policy | [2026-10-20-broadcast-oneshot](../2026-10-20-broadcast-oneshot/README.md), merged #443 | TP3 gate: outputs identical to r6x; release comparison level on every metric |
| vLLM 0221, sink detection and draft-zero rejection | [2026-10-19-sink-rejection-v2](../2026-10-19-sink-rejection-v2/README.md), gate passed #452 | `Say hi.` / `Thanks!` corrected with the ranks agreeing; 256/256 release text identical; zero flags on the release suites; throughput level; time to first token one decode step longer |
| vLLM 0222, late admission at decoder layer 30 | experiments/2026-10-21-late-admission (PR #454; analysis in PR #453) | TP3 screen: TTFT p50 −26% at four streams, −30% at eight; one stream and prefill level; outputs identical |
| vLLM 0223, sink rejection on by default for DeepSeek V4.1 | this record | the owner's decision of 2026-10-22, below |
| vLLM 0224, late admission with sink rejection through the engine-core loop | this record | tests only |
| vLLM 0225, the DS4.1 model tests' stand-ins carry the sink column buffer and the served architecture | this record (the release tests) | tests only |
| sparknet 0.5.0, the wider one-shot data phase | dgx-spark-networking `v0.5.0` (ed6af327): main 8618973, the striping-test fix, the release commit | TP3 engine grid decode step −1.6%, outputs identical |

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

- vLLM: r6x's series ([vllm/series](vllm/series)) plus 0185, 0221, 0222, 0223, 0224, 0225 on
  base 19f2c20e; patch head 672231a1b, tree c282fa51 (`git am --committer-date-is-author-date`
  with the build's committer reproduces it). Working branch `next-release` in
  ~/projects/vllm-ds41-next (local, as vLLM working branches are). The gate ran the series
  without 0225 (patch head c6a567ada, tree 32ae33c9); 0225 changes three test files and nothing
  else.
- sparknet: 0.5.0, tag `v0.5.0` at ed6af327 (tree f997a594), released 2026-10-22 on the owner's
  word: main 8618973 (the wide read phase), the GPU striping test for per-peer routes (b00a08e,
  qualified on the triangle 2026-10-21, applied onto main as 2031687) and the release commit
  (version, changelog, README serving status). `make check` and the branch's CI passed first.
- Everything else as r6x: [upstreams.lock.json](upstreams.lock.json),
  [source.json](source.json).
- Image `vllm-ds41f-kkref:19f2c20ed4d6-1022-2` (the live-name rename stays a separate step). The
  gate's image was `vllm-ds41f-kkref:19f2c20ed4d6-1022` (sha256:b1bbdea2521c, tree 32ae33c9),
  built 2026-10-22 on dgx2 in 10 min; the new tag keeps the two apart.

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

End-to-end smoke ([runs/smoke.sh](runs/smoke.sh), [runs/smoke-results.txt](runs/smoke-results.txt)):
a dummy-weight Llama-shaped model served with the series on dgx4, V2 runner, async scheduling,
multiprocess executor. Llama has no sink entry, so the detector stays off: the series serves
another model unchanged. Greedy outputs are identical with late admission off and at layer 18
(d73cddf1, late admission's own smoke hash); at four streams no request waits a step that does
not carry it (108 of 108 with it off), TTFT p50 54.5 to 35.7 ms.

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

### Results

[runs/gate-results.txt](runs/gate-results.txt) holds every step's summary line. Window
09:09-09:42 CET, no foreign requests in the traced phases; the candidate was ready 79 s after
launch at the production pool, under the 5 GiB startup guard and then the 3 GiB steady guard.

| Step | Result | Verdict |
| --- | --- | --- |
| determinism, with and without log probabilities | identical in all six scenarios, 0 token and 0 log-probability differences; hashes 62313cb5 / 9742ac2c and 4d143977 / 12fd8257, r6x's | pass |
| collapse probes | no collapse marker; `Say hi.` answers "Hi! 👋 How's your day going?"; flagged and corrected: `Say hi.`, `Thanks!`, `ok` (4 draft-zero rejections, 2 drafter cuts, 0 fallthroughs); the other 13 prompts unflagged; every rank 8 flagged rows | pass |
| sink counters after the ordinary text (determinism, trace, suite) | 0 flagged rows | pass |
| one stream, 120 s | 56.6 ± 3.2 tok/s (screen's r6x 56.7 ± 3.1) | level |
| cold prefill, three repeats | 32K 4413 ± 1.0%, 131K 4335 ± 0.6% tok/s (screen's r6x 4461 / 4386, a day earlier) | about −1%, across sessions: the release comparison judges |
| MemAvailable once serving | dgx1 6.70, dgx2 7.98, dgx3 8.19 GiB (r6x 6.70 / 8.04 / 8.11) | level |
| traced TTFT p50, four / eight streams | 269.1 / 351.7 ms (screen: r6x 258.8 / 353.9, late admission alone 190.9 / 249.0) | level with r6x; the planned criterion did not hold |
| isolated TTFT (twelve prompts, one at a time) | 146.9 ms mean (screen: r6x 105.9, late admission alone 107.0) | +41 ms, one decode step: sink rejection's held step |

**Time to first token.** The gate's TTFT criterion was set from late admission's screen alone and
missed sink rejection's cost: every first token is held one decode step and emitted when the next
step verifies it as draft zero. The two add as expected: late admission's 190.9 / 249.0 ms plus one
plain step at four and eight streams (71.3 / 99.6 ms in this trace) predicts 262 / 349 ms;
measured 269.1 / 351.7. Without late admission the held step would sit on top of r6x's
258.8 / 353.9 (about 330 / 454 ms). So late admission is working, and its whole gain at two or more
streams pays for the sink correction; a request alone, which late admission cannot shorten, waits
about 41 ms (one decode step) longer than on r6x. The trace's decomposition counts the held step as the "lookahead" step
(every request "waited a step"), because the first token appears one step after the step that
carried the request.

### Release tests

[bundles/release-tests](bundles/release-tests) (65 gate files) on dgx4 with `scripts/lab.py
kernel-local`, the gate's image: 1,766 passed, 158 skipped, **15 failed**, all in hand-built
DS4.1 model stand-ins:
- `test_deepseek_v4_1_ced_model.py` (2) and `test_deepseek_v4_1_sp_prefill.py` (12) call
  `DeepseekV4Model.forward` on a namespace without `_sink_columns`, which 0221's forward reads.
  0221's own record never ran the release bundle, so this is the first run to see it.
- `test_deepseek_v4_1_engram.py::test_disk_engram_model_allocates_hash_buffer_from_declared_caps`
  builds the model from a config namespace without `model_config.architecture`, which 0223's
  `SinkConfig.for_model` reads.

vLLM 0225 gives the stand-ins what the code reads (`_sink_columns=None`; the served architecture).
The three files then pass, 114 tests through the release runner in the gate's image with the fixed
files mounted. The bundle names image `-1022-2`; its full run on dgx4 (2026-10-22, 15:29-15:37
CET, sha256:07143428dc39): **1,781 passed, 158 skipped, 0 failed**.

## Promotion comparison (2026-10-22 10:03-11:05 CET)

Owner, 2026-10-22: sink rejection on as built, and a trimmed comparison of about an hour as the
promotion gate instead of the release suites: the engine grid, eight streams, one stream and 32K /
131K prefill, candidate first, r6x, candidate again. [promotion-suite.json](promotion-suite.json)
(model behaviour on 64 prompts, the grid, c8 and c1 closed loops of 180 s, prefill three repeats)
and [promotion-tp3.json](promotion-tp3.json): the image job for `-1022-2` (built on dgx2 in 10
min, sha256:07143428dc39, one ID on dgx1-3 and dgx4), then the three arms at the production pool
under the unchanged guard. The run read the suite from a copy staged under
`results/private/lab/spark-ds41f-1022/` on dgx1-3 (sha256 c6277891, this file). Table:
[results/promotion-tp3-table.txt](results/promotion-tp3-table.txt).

Verdicts, r6x to the candidate, against the run's own repeat (candidate to candidate again):

| Metric | Change | ±95% | Repeat | Verdict |
| --- | ---: | ---: | ---: | --- |
| engine grid, decode geometric mean (15 shapes) | −1.11% | 0.16% | −0.18% | better |
| eight streams, output tok/s | +1.53% | 2.06% | −0.31% | same |
| one stream, output tok/s | −0.09% | 9.99% | −0.38% | same |
| prefill 32K tok/s | −0.82% | 1.22% | −0.69% | same |
| prefill 131K tok/s | −0.74% | 0.53% | −0.75% | same |

- **Outputs:** greedy responses identical 64/64 in both pairs, teacher-forced distributions
  identical (KL 0, top-1 agreement 100%).
- **Engine grid:** better at 12 of 15 shapes in the first pair and 13 in the second, about
  −1.1 to −1.3%: sparknet 0.5.0's wider read phase (its A/B: −1.6%).
- **Prefill:** about 1% lower, in both pairs and in the gate. The first pair's 131K change is
  beyond its own interval, but the repeat moved as much (the candidate's second arm ran last and
  hottest), so the verdict is same. It is a direction worth tracing: the series adds to every
  forward only the sink column copy (16 bytes a token) and late admission's one event record.
- **Time to first token** (latency rows carry no interval): eight streams +2.6% / +3.8%, one
  stream +25% (0.127 to 0.158 s at p50). Late admission pays for sink rejection's held step at
  several streams; a request alone waits one decode step longer.
- **Boots:** ready in 80 / 93 / 93 s (r6x's restore 92 s). dgx1's MemAvailable minimum while
  starting (read-only 0.1 s sampler, the startup-memory experiment): candidate 6.77 and 6.01 GiB,
  r6x 5.83 and 5.68 GiB, against the 5 GiB guard; the candidate's two boots differ by 0.76 GiB,
  more than the candidate against r6x.

## Not covered

- The four-node ring: [spark-ds41f-1022-tp4.json](spark-ds41f-1022-tp4.json) is ready, but the
  ring is not cabled. TP4 stays unqualified on this release until a ring release run.
- dgx1's startup peak: a production-sized pool boot can fail the 5 GiB startup guard from
  boot-to-boot spread alone (2026-10-21, 12.7 MiB short on one boot). The guard stays as it is;
  a separate experiment works on lowering the peak.
