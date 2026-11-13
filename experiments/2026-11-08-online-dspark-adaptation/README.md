# Online verification-driven DSpark drafter adaptation

PR #1002. Base commit: `858229403` (main). Image: `spark-ds41f:candidate-54` (frozen candidate-54, vLLM tree
`ad2dc11f`), at `tp4-switched` (sp1..sp4). The change is mounted over the image; nothing is built.

The 2026-11-08 version of this experiment measured the code on random weights and synthetic tokens in a standalone
script, and the code was not called by serving. That record is kept unchanged in
[screen-2026-11-08/](screen-2026-11-08/README.md); its numbers say nothing about DS4.1. This page is the version that
serves, and its one measurement on DS4.1.

## What was wrong with the 2026-11-08 code

1. **Not wired into serving.** `step_on_candidates`, `step_from_verified_tokens`, `OnlineConfidenceCalibrator.observe`
   and `CopyDrafter.record_session_tokens` were called only from the tests and the screen script. The
   `VLLM_DSPARK_*` flags built the objects and nothing fed them.
2. **Global token ids on a vocabulary shard.** Since `0012` (`dspark-vocab-parallel-greedy`) DSpark's Markov W2 is a
   vocabulary-sharded `ParallelLMHead` (and on the TileLang path 12-bit packed). The adapter read and wrote
   `markov_w2.weight[candidate_ids]` with global ids, and `update_w2` defaulted to `True`. On rank 1 of TP4, id 5
   reads token 32,325's row; ids past 32,320 index past the shard. Ranks would read and write different rows: mixed
   rank state. The TP test ran four "ranks" in one process on one unsharded table.
3. **Drift.** Updates were in place with no bound, no way back to the loaded weights (weight decay pulled toward
   zero and was off) and one state for all traffic.
4. **Session memory shared by every request.** `SessionTokenMemory` was one ring for all requests: one user's
   tokens drafted for another (and timing would show it).

## What changed

[The patch](vllm/0001-feat-dspark-online-verification-driven-drafter-adapt.patch) is on candidate-54's vLLM head
`15833ed1` (0799); applied to a fresh candidate-54 tree it gives tree `6e7f28c4`. [overlay/](overlay) holds the six
files it changes outside `tests/`, from that tree:

| File | sha256 (12) |
| --- | --- |
| `vllm/v1/worker/gpu/model_runner.py` | `b8e4a86f08cf` |
| `vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py` | `0f972f2a4daa` |
| `vllm/v1/worker/gpu/spec_decode/copy_drafts.py` | `e1b62c888aa5` |
| `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` | `9bfc115c5a30` |
| `vllm/v1/worker/gpu/spec_decode/online_dspark_adaptation.py` | `a2549e65878a` |
| `vllm/v1/worker/gpu/spec_decode/speculator.py` | `c13b39431462` |

Everything is off unless its variable is set; unset, nothing is allocated or launched.

- **Part 1, `VLLM_DSPARK_ONLINE_CALIBRATION=1`** (or part of `VLLM_DSPARK_ONLINE_ADAPT`): the per-position calibrator
  (`c_hat = sigmoid(gamma_i * logit(c) + beta_i)`, FP64, an EMA-Newton step at every verification over the accepted
  positions and the first rejected one) replaces `OnlineAcceptanceEstimator`'s shared-slope refit as the calibration
  the captured draft graph applies to the confidence head (the Triton kernel reads a slope per position,
  `PER_POSITION_SLOPE`). It learns from the raw head log-odds the draft step stores, with the estimator's own
  positions and labels (dead-row cut included, copy-drafted rows left out). The confidences count as calibrated after
  100 steps, as with the estimator.
- **Part 2, `VLLM_DSPARK_ONLINE_ADAPT=1`**: `DSparkOnlineAdaptation` grades each draft with the next step's
  verification and takes the 2026-11-08 E2E-TV + BCE step on the replicated Markov W1 and the confidence head.
  - Inside the captured draft step it keeps what the update needs and the next step overwrites: each request's head
    hidden rows, its anchor token and its request slot (`capture`). Only a real draft step is graded (`arm`); a
    warmup or dummy draft is not, and a draft is graded once.
  - `observe_verification` now receives the step's sampled tokens. At each verified position (`i <= A`, `i < B`) the
    support is the target's token, the draft (or the next id when they agree) and one other id, with target
    probabilities 0.92 / 0.025 / 0.025 and 0.03 excluded: greedy verification exposes only the target's argmax, so
    these are the 2026-11-08 code's synthetic teacher, unchanged.
  - The drafter's logits on the support are its own: base logits `norm(h) . lm_head[v]` and Markov logits
    `W1[prev] . W2[v]`. W2 and the LM head are read through `VocabRows`: the rank whose shard owns an id
    contributes its row (decoded from the 12-bit packed table by the TileLang kernel, declared at load), the others
    zeros, and one all-reduce sums them. Each id has one owner, so the sum is exact and every rank gets the same
    bits. Nothing indexes a shard with a global id.
  - W2 is never written: `update_w2` defaults to `False`, and `True` (`VLLM_DSPARK_ONLINE_UPDATE_W2`) is refused for a
    sharded or packed W2 and under TP.
  - Every reduction has a fixed order and nothing uses atomics (duplicate W1 rows are summed after a stable sort and
    written once), so ranks stay identical. `VLLM_DSPARK_ONLINE_DIGEST_STEPS=N` logs, per rank, a bit digest of W1
    and the confidence head at load and every N updates.
  - Drift: the loaded W1 is kept (63 MiB a rank), each touched row's displacement is bounded by
    `VLLM_DSPARK_ONLINE_MAX_DRIFT` (default 1.0) times the loaded row's norm, and `reset()` restores the loaded heads.
    The state is still one for all traffic: W1 and the confidence head are shared by every request.
  - It refuses a sharded W1, top-k or probabilistic drafts, and the display carve-out integrity check (which would
    report the W1 writes as corruption).
- **Part 3, `VLLM_DSPARK_SESSION_MEMORY_TOKENS=N`** (no longer implied by Part 2): one ring per request slot, cleared
  when a request takes the slot, filled with the request's own committed tokens and read only for that request.
  Scoped this way it holds a subset of the request's own history, which copy drafting already matches, so it can add
  little; it is not measured here. A session key from the API (none reaches the worker today) would be needed for
  anything wider.

## Tests

`pytest --noconftest tests/v1/spec_decode/test_online_dspark_adaptation.py`, in the candidate-54 image on sp8 with the
patched files mounted and the DS4.1 checkpoint (`DS41F_SNAPSHOT`): 13 passed. On DS4.1's shapes (vocabulary 129,280,
Markov rank 256, head hidden 5,120, 5 drafts, 16 requests):

- global ids map to the owning shard (vLLM's own `_get_indices` sharding), and the shard bug is shown;
- `update_w2=True` is refused for a sharded and for a packed W2;
- **four real ranks (gloo processes) on sharded W2 and LM heads end bit-identical, equal to one rank with the full
  tables, with W2 untouched**;
- only armed drafts, observed positions and DSpark-drafted (not copy) rows move W1, and a draft is graded once;
- the drift bound holds and `reset` restores the loaded bits;
- the closed-form W1 step equals autograd's E2E-TV gradient;
- duplicate ids are summed once; the calibrator drives the estimator (and its Triton kernel's per-position slope
  matches);
- packed rows of a real TP4 shard decode to the BF16 rows; GPU updates are run-to-run identical;
- the checkpoint's own heads (`mtp.2.markov_head.*`, `mtp.2.confidence_head`, `mtp.2.norm`, `head`) adapt within the
  bound and reset to the checkpoint's bits;
- session rings are per request.

With the spec-decode suites the change touches (acceptance estimator, adaptive verification, copy drafts, DSpark top-k,
gathered Markov, vocabulary parallel, greedy Markov, utils, autoregressive speculator): 247 passed, 1 skipped (two
GPUs), 2 failed: `test_dspark_topk.py::test_dspark_revisited_position_uses_step_seed[False|True]`, which fail the same
way on the unmodified candidate-54 image ([receipt](receipts/tests/pytest-spec-decode-suites.txt)).

## The experiment

**One variable:** adaptation on (`VLLM_DSPARK_ONLINE_ADAPT=1`: Parts 1 and 2, W2 frozen, no session memory) against
off (the same overlay, variables unset). Both arms boot candidate-54 with the overlay, so the control also shows the
overlay is inert when off. Two boots each, in the order on, off, on, off; each boot starts from the checkpoint's heads.

- [tp4-switched-off.json](tp4-switched-off.json): candidate-54 at `tp4-switched` with the overlay mounted
  (`container.mounts`).
- [tp4-switched-on.json](tp4-switched-on.json): that, with `VLLM_DSPARK_ONLINE_ADAPT=1` and
  `VLLM_DSPARK_ONLINE_DIGEST_STEPS=256`.
- [run-tp4-switched.json](run-tp4-switched.json): one measure job, no bench; on each boot, in order:
  1. [online_adaptation_workload.py](online_adaptation_workload.py): four fixed coding conversations of six turns
     (Python cache class, Express API, Rust CLI, SQL and pandas), one request at a time, temperature 0, thinking off,
     each turn the whole conversation so far with the model's own replies, at most 384 tokens a reply; then the
     sixteen held-out general prompts of `scripts/distinct_streams.py`, alone, 256 tokens each. Per request, from the
     server's counters: acceptance length (1 + accepted tokens per draft step), verified drafts per step, decode
     tok/s, and the reply's hash.
  2. The determinism gate, as candidate-54's (`token_determinism_record.py --tokens 192 --long-lines 700 --wide`),
     with log-probabilities and without, after the workload, so on the adapted state.
  3. [collect_adaptation_digests.py](collect_adaptation_digests.py): every rank's logged digests, compared at each
     update count.

```bash
bin/spark candidate queue tp4-switched experiments/2026-11-08-online-dspark-adaptation/run-tp4-switched.json --apply
bin/spark candidate verdicts tp4-switched online-dspark-adaptation-tp4-switched
bin/spark candidate gate tp4-switched online-dspark-adaptation-tp4-switched \
  --expect 18d619615e51c7df474118ed5f7d5de38dec82c3a08a8c74fb0c875824372286/d65686498eb85a2e2bde1bfb8bab84d309396c709b4916070c0b2f670fabcfd0 \
  --expect-text 0a2dacdde6ff2147367bc329818a9707862f7de08ea7e0c0513b01bfff0fd37f/12fd825767d21b022ed3fdcbbc6937289168c07882f022727d0725c86aa689fc
```

The expected hashes are candidate-54's own at TP4 (its `tp4-ring` and `tp4-switched` gates agree).

**Marks, declared before measuring:** the gate's hashes equal candidate-54's on every boot of both arms; the four
ranks' digests are equal at every compared update; then the comparison of acceptance length per turn, coding and
held-out, and c1 decode tok/s, on against off, with both boots of each.

## Results

Queue entry `20261113T092058-run-tp4-switched.json`, 2026-11-13 10:02–10:38Z, on `tp4-switched` (sp1..sp4), lab
checkout `19ff11ab2` (main), image `spark-ds41f:candidate-54` with the overlay. Boots in the order on, off, on, off;
every boot warm (no kernel compiled), ready in 126–130 s in both arms; every extra exited 0. Receipts:
[receipts/run-tp4-switched/](receipts/run-tp4-switched) (lab table, window log, every arm's workload, gate and
digest output). The tables below are [analyze.py](analyze.py)'s, also in
[receipts/run-tp4-switched/analysis.md](receipts/run-tp4-switched/analysis.md), with the per-request numbers in
[results-tp4-switched.json](results-tp4-switched.json):

```bash
python3 experiments/2026-11-08-online-dspark-adaptation/analyze.py \
  experiments/2026-11-08-online-dspark-adaptation/receipts/run-tp4-switched online-dspark-adaptation-tp4-switched \
  --json experiments/2026-11-08-online-dspark-adaptation/results-tp4-switched.json
```

**Acceptance length per turn** (coding, pooled over the four conversations; acceptance length = 1 + accepted tokens
per draft step):

| Turn | adapt-on | adapt-on-again | adapt-off | adapt-off-again | on vs off |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.403 | 4.403 | 4.540 | 4.540 | −3.01% |
| 2 | 4.651 | 4.651 | 4.742 | 4.742 | −1.92% |
| 3 | 5.786 | 5.786 | 5.815 | 5.815 | −0.51% |
| 4 | 5.746 | 5.746 | 5.704 | 5.704 | +0.75% |
| 5 | 5.023 | 5.023 | 5.141 | 5.141 | −2.29% |
| 6 | 6.287 | 6.287 | 6.448 | 6.448 | −2.49% |

**Totals:**

| | adapt-on | adapt-on-again | adapt-off | adapt-off-again | on vs off |
| --- | ---: | ---: | ---: | ---: | ---: |
| coding acceptance length | 5.231 | 5.231 | 5.319 | 5.319 | −1.66% |
| held-out acceptance length | 2.219 | 2.219 | 2.311 | 2.311 | −4.01% |
| coding c1 decode tok/s | 152.13 | 152.01 | 157.98 | 158.31 | −3.84% |
| held-out c1 decode tok/s | 80.51 | 80.44 | 85.16 | 85.22 | −5.53% |

Acceptance is the same to the last digit on both boots of an arm (the workload is one request at a time at
temperature 0, so each boot sees the same traffic in the same order); decode rate moves at most 0.21% between boots
of an arm, against an on−off gap of 3.8–5.5%.

**Where it goes** (pooled over each set's requests; decode time per step is each request's tokens after the first over
its decode rate):

| | adapt-on | adapt-on-again | adapt-off | adapt-off-again | on vs off |
| --- | ---: | ---: | ---: | ---: | ---: |
| coding draft steps | 1769 | 1769 | 1741 | 1741 | +1.61% |
| coding verified drafts per step | 5.179 | 5.179 | 5.387 | 5.387 | −3.85% |
| coding accepted / verified | 0.817 | 0.817 | 0.802 | 0.802 | +1.88% |
| coding decode ms per step | 35.28 | 35.32 | 34.47 | 34.44 | +2.47% |
| held-out draft steps | 1819 | 1819 | 1766 | 1766 | +3.00% |
| held-out verified drafts per step | 2.224 | 2.224 | 2.672 | 2.672 | −16.76% |
| held-out accepted / verified | 0.548 | 0.548 | 0.491 | 0.491 | +11.65% |
| held-out decode ms per step | 28.29 | 28.32 | 27.49 | 27.45 | +3.04% |

**Text, gates and rank state:**

| Arm | workload text | gate (log-probs) ref / long | gate (text) ref / long | ranks' digests |
| --- | --- | --- | --- | --- |
| adapt-on | `9df935a06bc6` | `18d61961` / `d6568649` | `0a2dacdd` / `12fd8257` | 4 ranks equal at all 46 logged counts (0..11,520) |
| adapt-on-again | `9df935a06bc6` | `18d61961` / `d6568649` | `0a2dacdd` / `12fd8257` | 4 ranks equal at all 46 logged counts (0..11,520) |
| adapt-off | `9df935a06bc6` | `18d61961` / `d6568649` | `0a2dacdd` / `12fd8257` | none logged (off) |
| adapt-off-again | `9df935a06bc6` | `18d61961` / `d6568649` | `0a2dacdd` / `12fd8257` | none logged (off) |

`bin/spark candidate gate` with candidate-54's hashes: **pass** on all eight gate runs (four boots, with and without
log-probabilities), every scenario `identical` with 0 token and 0 log-probability differences. The gates ran after
the workload, so on the adapted state. The workload's 40 replies hash the same on all four boots.

**Adapted state between boots.** The two on boots' rank-0 digests are equal through update 4,352, which covers the
one-request workload (about 3,600 verification steps), and differ from update 4,608 on (28 of 46 logged counts), in
the gates, which run up to 15 requests at once (`mixed15`). Within a boot all four ranks still agree at every count. The adapted
state is a function of the order in which concurrent requests share steps, so two servers given the same requests can
end with different drafters (output stays the same; only acceptance can differ).

**W1 movement.** The largest W1 row displacement, relative to the loaded row's norm, is 0.0049 after the first 256
updates, 0.0049 at the end of the workload and 0.024 after the gates (11,520 updates), far inside the 1.0 bound.

**Memory.** KV 10.5 GiB in both arms (fixed by the profile); every boot started from 115.6–116.5 GiB `MemAvailable`
on each node; no memory guard warned. The adapter adds the loaded W1 copy (63 MiB a rank) and its step buffers.

## Conclusion

Unfavourable: **adaptation on is slower than off at c1 on DS4.1**, on the coding conversations it was meant to help
(−3.8% decode, −1.7% acceptance length) and more on the held-out prompts after them (−5.5%, −4.0%), on both boots
of each arm. It does not improve with exposure: acceptance is below off on five of six turns, and turn 6 (the most
adapted) is −2.5%.

- The loss in acceptance comes from verification length, not from worse drafts. With adaptation on, `0504`'s
  per-row budgets verify fewer drafts per step (−3.9% coding, −16.8% held-out); a larger share of those is accepted
  (+1.9%, +11.7%), but not enough to make up for the drafts no longer tried. This run changes the calibration and
  the confidence head together, so it does not say which of the two lowered the budgets. W1 hardly moved (at most
  2.4% of a row norm, 0.5% during the workload), so the Markov update is an unlikely cause; the confidence path
  (calibration and head) is what sets the budgets.
- The adaptation's own work costs about 0.8 ms per step (+2.5% coding, +3.0% held-out step time), although the on arm
  verifies fewer tokens per step: on every verification step the calibrator and the graded step (support reads, one
  all-reduce, the W1 and head updates). It was not profiled further.
- Correctness holds: the four ranks' adapted state is bit-identical within each boot at TP4, the determinism gates
  equal candidate-54's on every boot of both arms (also after 11,520 updates), and the off arm, with the overlay
  mounted and the variables unset, gives candidate-54's gate hashes: off, the code changes no output.

Not adopted. The code stays behind its variables (off by default) in this experiment's patch; nothing is added to
the deployment's patch series. A calibration-only arm (`VLLM_DSPARK_ONLINE_CALIBRATION=1` against off) would say
whether the per-position calibrator alone shortens `0504`'s budgets; it was not part of this run.
