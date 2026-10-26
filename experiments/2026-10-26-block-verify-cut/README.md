# DSpark block verification ends each block at the dead-row cut

Base: vLLM `next-candidate` 1bc005688. Branch: vLLM `block-verify-cut` (three commits on
it). One variable: the sampled verification of a request with dead rows. The recipe is
unchanged (`config/cluster.json`: `draft_sample_method: greedy`,
`rejection_sample_method: block`, `SPARK_DS41F_DSPARK_DEAD_ROWS_TAU=0.2`).

## Summary

- **Before:** block verification of a sampled request scanned every verified draft,
  dead rows included. Then the insertion (or the count kernel) capped the count at the
  cut plus one. Rows past the cut skipped the routed MoE, yet their logits set the
  thresholds that decide where a block ends.
- **After:** the TileLang verify and resample kernels take the cut. A request's drafts
  from its cut on are -1 placeholders, which end a block the way the drafter's own
  placeholders do. The outcome is block verification's of the truncated draft, bit for
  bit. The TileLang path drops the insertion's cap, since nothing it commits can pass
  the cut plus one. A greedy request commits exactly what it committed before.
- **Severity, corrected.** The bug review said the cap "doesn't restore the exact target
  distribution". That holds only when a dead row's logits are non-finite. Block
  verification reaches step i with probability p_i, the product of the live rows'
  draft probabilities, whatever the later rows hold (see below). So for finite dead
  rows, which is what production computes, the capped scan already has the truncated
  block's law. The fix changes:
  - the **realization**: the same seed gives other sampled text in steps whose cut is
    below their verified drafts;
  - the **robustness**: a NaN in a dead row no longer commits the drafts as they are;
  - the **dependence**: the outcome is a function of live rows only.

  It does not change the law, the acceptance or the throughput in expectation.

## The path, end to end (next-candidate 1bc005688)

1. **The cut.** `AdaptiveVerificationManager.dead_row_args`
   (`vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:382`) hands the draft
   combine the current drafts' confidences, the padding flags, the cut buffer and
   log(tau). The TileLang combine (`vllm/v1/worker/gpu/tilelang_inputs.py:1496-1518`)
   computes the request's survival, the running sum of log confidences. The cut counts
   the drafts whose survival is at least log(tau): a prefix, from 0 to all of them. It
   is block + 1 (9 for five drafts) for a request without drafts or not decoding. The
   inputs of drafts at or past the cut are marked padding, so routed MoE skips them.
2. **The dead rows' logits.** A padding row's routes get expert -1 and weight 0
   (`vllm/models/deepseek_v4_1/tilelang/router.py:19`). The combine adds only routes
   with a layout row (`moe.py:385`, `if pos >= 0`). A dead row's hidden state is
   therefore attention, the shared expert and the residual: finite, but not the
   target's.
3. **Rows and drafts.** Request r's logits rows are `start .. start + n` for its n
   verified drafts. Row `start + k + 1` is draft k's input, and step k verifies draft k
   with row `start + k`. With cut c, rows `start + c + 1 ..` are dead and rows
   `start .. start + c` are live. Step c verifies draft c from a live row, but its
   continuation would need the dead row `start + c + 1`, so the truncated block is the
   first c drafts plus the bonus of row `start + c`.
4. **To verification.** `model_runner.py:2007` passes the cut to the rejection sampler.
   `RejectionSampler.__call__` forwarded it into verification only with logprobs or
   sink rejection on (`rejection_sampler.py:492`), and always to the count kernel
   (`tilelang_token_counts.py:46-47`, `min(count, cut + 1)`). `rejection_sample` ran
   `verify_and_resample` without it and capped in the insertion
   (`rejection_sampler_utils.py:1378`, `tilelang_resampled.py:49-53`).

### How dead rows entered the outcome

In `verify_kernel` (`tilelang_rejection.py`), every step below the request's n drafts is
live (`:601`):
- **The thresholds.** Step s's threshold is block verification's
  `p_s (1 - m_{s+1}) / (1 - p_s m_{s+1})`, from the next draft's probability under row
  s + 1, or the joint ratio p_s for the last draft (`:695`). The threshold of step c - 1
  therefore read `m_c`, and every step from c on read dead rows.
- **The count.** It is the last step whose draw is at most its threshold (`:736`), over
  all n steps.
- **The committed token at the cut.** With the cap at c + 1:
  - if the scan accepted more than c drafts, the token at c is draft c itself, since
    `sampled[i] = token` (`:747`);
  - if it accepted exactly c, the token is the residual of row c with draft c masked
    (`:948`: the row is not the request's last);
  - in the truncated block, the token at c is the bonus of row c, with nothing masked.
- **The sink's alternates** of that last draw masked draft c by the same test
  (`tilelang_sampling.py:1389`).

### Why the law was nevertheless the truncated block's (finite dead rows)

Write `P_i` for the probability that block verification accepts drafts 0 to i - 1:
the product `m_0 ... m_{i-1}` of their target probabilities, each under its own live
row. For block verification's thresholds,
`(1 - h_s)(1 - P_{s+1}) = 1 - P_{s+1} - P_s (1 - m_{s+1}) = 1 - P_s`. So
`prod_{j >= i} (1 - h_j) = 1 - P_i` and `P(count >= i) = P_i`, for any probabilities
`m_{s+1}` the later rows hold. Two consequences:
- The count, capped at c + 1, has the same law as the truncated block's.
- The token at c follows the target. Among outcomes that reach c, the scan goes past c
  with probability `P_{c+1} / P_c = m_c`, which is draft c's target probability; the
  rest of the time it emits the residual without draft c.

The identity needs every `m` to be a number in [0, 1]. A NaN draft probability breaks
it: the kernels' threshold is 1 where its denominator is NaN, and `min.f32` turns the
ratio after a NaN back to 1. Every draft is then accepted, and the cap commits the
drafts as they are.

[exact_law.py](exact_law.py) computes both procedures' laws exactly, in FP64, with the
kernels' non-finite rules: 5 tokens, 4 drafts, every draft sequence, the uniform draws
integrated out ([exact_law.txt](exact_law.txt)):

| Dead rows | Capped scan: total variation from the target | Truncated block |
| --- | ---: | ---: |
| random (scale 4) | 6e-16 to 9e-16 | 6e-16 to 9e-16 |
| the verified draft certain (+80) | same | same |
| the verified draft impossible (-inf) | same | same |
| the verified draft +inf | same | same |
| noise at 1e4 | same | same |
| NaN | **0.31 at every position up to the cut** (the drafts' own distribution) | 6e-16 to 9e-16 |

The reach probabilities agree to every printed digit for every finite kind. With NaN,
the capped scan reaches every position with probability 1.

## The fix (vLLM `block-verify-cut`)

| Commit | Change |
| --- | --- |
| 2a529d88c | `verify_kernel` and `resample_kernel` take a compile-time `cut` variant, which reads the step's cut buffer, so a captured graph reads the cut on every replay. Drafts from the cut on are placeholders, and a count at the cut draws the bonus. `rejection_sample` passes the cut to the TileLang kernels and no longer caps in their insertion. `RejectionSampler` passes the cut to verification on every step, and to the sink's candidates, for which a draw at the cut was the bonus. The step declares the cut variants beside the warmup's uncut ones, and the capped insertion only for draft-by-draft methods. Block verification with draft logits, a watermark or FP64 draws (the Triton kernels) is refused with dead rows. |
| c51d23db8 | Tests (below). |
| 4ce2b98ad | `benchmark_tilelang_rejection.py --dead-rows`. |

Kept:
- **The count kernel's clamp** (`get_num_sampled_and_rejected`). It is still the commit
  rule for greedy verification from vocabulary shards (`_verify_greedily`), and for
  block verification it is now a no-op.
- **The insertion's cap** for the Triton methods. Standard verification goes draft by
  draft, so the cap is exact there: draft c is accepted with probability
  `min(1, m_c / q_c)`, otherwise the residual is drawn, and either way the token at c
  follows row c.

Refused: the 2026-10-18 probabilistic-drafting arm ran block verification through Triton
with dead rows on. Rerunning it now needs `SPARK_DS41F_DSPARK_DEAD_ROWS_TAU=0`.

## Tests

On dgx4 (the GPU suites), in bundles from [make_bundles.py](make_bundles.py):

- `test_tilelang_rejection.py`:
  - verification's outputs, the elements it leaves alone included, equal those of the
    same drafts with -1 placeholders from the cut on, bit for bit, whatever the dead
    rows hold (random, the verified draft certain or impossible, NaN, +inf, noise at
    1e4), over 3 vocabularies, 2 dtypes and 3 logits cases;
  - resampling's block winners, for counts from 0 to every draft;
  - `rejection_sample`: for each sampled request, the count and tokens of its draft
    truncated at its cut, bit for bit; for each greedy request, the count and tokens it
    committed with the cap;
  - a positive control: the capped scan differs from the truncated block for some
    requests;
  - a captured graph reads the cut on each replay;
  - with dead rows the step declares the cut variants, without them none; both
    compile.
- `test_rejection_sampler_utils.py`:
  - chi-squared against the target at every position up to each cut, with random and
    with NaN dead rows (400,000 trials, 16 tokens, 5 drafts, cuts 0-4);
  - its positive control: the capped scan fails with NaN dead rows;
  - greedy is the minimum, as before;
  - a sampled request equals its placeholder drafts;
  - the insertion's declarations;
  - the Triton refusal.
- `test_gpu_tilelang_sampling.py`: the sink's candidates with a cut equal those of the
  placeholder drafts.
- Unchanged files that the change touches: chunking, int64 offsets, adaptive
  verification, dead rows, the sink, preparation and the input batch.

`next-tests` runs the same files on next-candidate's own tree, to separate any failure
that predates the branch.

On the laptop: [exact_law.py](exact_law.py), in FP64.

## dgx4 runs

Setup:
- **Node:** dgx4 alone, while production was down for re-cabling. dgx1-4 were powered
  off at 08:10 CET on 2026-10-26, on the owner's instruction. dgx4 was released for
  testing in the afternoon.
- **Image:** `spark-ds41f:19f2c20ed4d6-next-6`.
- **Command:** `scripts/lab.py kernel-local`, from a checkout of this branch under
  `~/spark-lab/agents/block-verify-cut/`. The outputs are in its `out/`, with copies in
  [runs/](runs/), tabulated by [bench_table.py](bench_table.py).
- **`next-tests`** runs only if a test fails in a way the branch does not explain.

### Run 1, 13:50-14:40 CET: two failures, both in the harness

vLLM 4ce2b98ad, spark-ds41f 5e1368f4.

**`cut-tests`** ([output](runs/run-1/cut-tests-output.txt)):

| Step | Result |
| --- | --- |
| verification tests | 663 passed |
| rejection sampler tests | 152 passed, 1 skipped (needs two GPUs), **2 failed** |
| sink kernel tests | 115 passed |
| sink tests | 44 passed |
| preparation tests | 71 passed |

The failure was both parametrizations of
`test_ds41f_verification_emits_same_tokens_with_insert_port`. The test's stand-in for
`insert_resampled` took the cut as a required argument, and the TileLang path no longer
passes one. Fixed in 21ce222ba: the argument defaults to None, and the stand-in still
asserts it is None. Every new test passed, the distribution test and its control
included.

**`cut-bench`** ([output](runs/run-1/cut-bench-output.txt)):
- **Without dead rows:** all 20 `rejection_sample` cases passed the gate. The first
  `local_stats` case then stopped the step. The harness adds the maximum's column back
  to our block sums before comparing them with Triton's, but next-candidate's sums
  (with the softmax fix) leave that column out too. Fixed in ab8d398c0: against a fixed
  release, the sums are compared bit for bit.
- **With dead rows:** all 60 cases (`rejection_sample`, `verify` and `resample`, 1-96
  rows, both dtypes, sampled and mixed) passed the gate. Every output was bit for bit
  that of the placeholder drafts, and every greedy request's was next-candidate's.

### Run 2

vLLM ab8d398c0: both bundles again. Pending.

## What the release comparison should expect

- **Temperature 0:** the determinism gate's hashes equal next-candidate's. Greedy
  verification is unchanged.
- **Sampled suites:** different text for the same seed. The change applies in every
  sampled step where a request's cut is below its verified drafts; there the dead rows'
  thresholds no longer take part, and the token at the cut is the bonus. The sampled
  digest changes by design.
- **Same law:** tokens a step, acceptance, sampled throughput and the model-behaviour
  metrics should be level within the suites' noise. A change beyond it is a finding to
  investigate, not an expected effect.
- **Cost:** the verification kernels gain one load and one compare. The insertion no
  longer reads the cut. The gated benchmark above measures it.
