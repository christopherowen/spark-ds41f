# The softmax fix: Barron et al.'s Eq. 5 in the logprob and verification kernels

Owner, 2026-10-25 (relayed by the release session): prepare a fix for the softmax rounding
bugs of Barron, Gnanapragasam and Tschannen's "A Fix for Softmax and Cross-Entropy"
(<https://jonbarron.info/softmaxfix/>), wherever they reach DeepSeek V4.1's serving outputs,
for correctness, though release triage bounded their effect at a few parts per million a
step. For the next release; spark-ds41f-1025 (candidate 5/6) stays frozen.

Base: vLLM `tilelang-ports` at 26382f5cf, the candidate's package tree (image
`spark-ds41f:19f2c20ed4d6-next-5`; candidate 6 adds two test patches only). The fix is vLLM
branch `softmax-fix`, four commits over 26382f5cf through 80e8c188c (local). The analysis's
numbers are from served outputs already in this repository or from a CPU emulation of the
kernels' FP32 arithmetic ([emulate.py](emulate.py)); the kernels' GPU tests and benchmarks ran
on dgx4 (see "GPU results"). The serving check needs the cluster and waits for it.

## The paper, in short

- **Collapse.** A (log-)softmax that shifts by the maximum logit and adds its exp(0) = 1 into
  the sum rounds the sum to 1 once the other columns add less than half an ULP of 1. The
  margin m of the top logit over the others' log-sum-exp past which that happens is about 5.5
  nats in BF16, 16.6 in FP32 and 36.7 in FP64. The top token's log probability then becomes
  exactly 0, its probability exactly 1, and 1 - p exactly 0.
- **The fix (Eq. 5).** The maximum's shifted logit is exactly 0, S = sum of exp(z_j - max)
  over the other columns, and logsoftmax_k = (z_k - max) - log1p(S), i.e. softplus(-m). For a
  known target, the target takes the maximum's place (Eq. 6). 1 - p_t is the others' mass,
  never 1.0 - p_t.
- **Cancellation.** q - p of nearly equal probabilities loses its digits; compute such
  differences from logs (expm1, log1p).

## Where it reaches DeepSeek V4.1's outputs

The candidate's recipe ([cluster.json](../2026-10-25-spark-ds41f-candidate/cluster.json)):
DSpark with greedy (one-hot) drafts, block verification, adaptive verification (row cost
0.035, dead rows at tau 0.2), the generation default temperature 1.0 and top-p 1.0, logprobs
from raw logits, every serving kernel TileLang. A request that sets no temperature samples, so block verification decides
its tokens.

Ranked by what users see or the engine decides:

| # | Site (vLLM 26382f5cf) | Who sees it | The bug there | Fixed |
| --- | --- | --- | --- | --- |
| 1 | `sample/tilelang_logprob.py` `topk_log_softmax` (one launch for every logprob: `compute_token_logprobs`, `compute_topk_scores`, prompt logprobs, the verified tokens' logprobs, the sink's alternates) | every logprob served: `logprobs`/`top_logprobs`, `prompt_logprobs`, `logprob_token_ids` | Collapse: the top token's log probability is exactly 0 past a margin of 16.6, and earlier: each 1024-column block added after the maximum's is rounded against 1 and loses what it holds below 2^-24, so confident tokens' values sit on a 2^-23 grid, 30-100% off at margins 12-16 depending on where the argmax sits in the vocabulary | yes |
| 2 | `spec_decode/tilelang_rejection.py` `verify` (block verification) | the tokens of every sampled request (temperature > 0, the recipe's default) | The draft's log probability is `logit - (max + log(sum))`, rounded at the magnitude of the maximum logit (an ULP of 32 is 3.8e-6) and collapsed as in 1, so log m is 0 for confident drafts; the cumulative ratio p = exp(sum of log m) then rounds to 1; the threshold p(1 - m) / (p(1 - m) + 1 - p) takes 1 - m and 1 - p of those probabilities (cancellation, or 0/0, which becomes 1) | yes |
| 3 | `spec_decode/tilelang_rejection.py` `local_stats` | feeds 2 | the block sums include the block maximum's 1 | yes |
| 4 | Gumbel sampling, temperature, resampling (`tilelang_sampling`, `resample`) | sampled tokens | none: the argmax of logits / T plus noise; a rejected draft is masked to -inf | no change |
| 5 | Greedy verification, min-p, the sink detector | greedy tokens, flags | none: argmaxes and logit thresholds | no change |
| 6 | Top-k / top-p (`tilelang_topk_topp.py`) | sampled tokens with top-p < 1 | no 1 - p: the kept mass is compared with p; its sums carry ordinary FP32 rounding (a token flips only when p lies within about 1e-7 of a cumulative mass). See "Not changed" for its probability-to-logit pivot | no change |
| 7 | DSpark's confidence head (`deepseek_v4_1/tilelang/confidence.py`, FP32 sigmoid) and adaptive verification | how many drafts are verified (speed only; the output distribution does not depend on it) | sigmoid(z) = 1 / (1 + exp(-z)) is 1.0 past z = 16.6, so a survival product overstates by up to 6e-8 a position; nothing takes 1 - conf or log(1 - conf); dead rows compare log survival with log 0.2 | no change |
| 8 | Probabilistic drafts, FP64 draws, watermark, synthetic rates (the Triton kernels of `rejection_sampler_utils`, `vocab_parallel`) | not in the recipe | the residual `max(p M_b - M_s, 0)` is the paper's q - p cancellation; vocab-parallel drafts already take `log p + log1p(-q / p)` | no change (not served) |
| 9 | The model: sparse-MLA and compressor attention, the vision tower's attention, mHC's 4x4 Sinkhorn, the sqrtsoftplus router | hidden states | attention's normalizer rounds to 1 past the cliff, which scales the other keys' contributions by 1 + O(6e-8), with no complement taken; Sinkhorn normalizes rows and columns of a 4x4; the router renormalizes its top-k scores | no change: DeepSeek's reference arithmetic, recorded only |

## How often real traffic crosses the cliff

From DS4.1 outputs already in this repository (the release's arithmetic, Triton's then, the
same since the TileLang port):

- A code completion, 175 tokens with top-5 alternatives
  (`experiments/2026-09-23-latest-head-rebase/runs/consolidated-managed/code-logprobs.json`):
  23 tokens (13.1%) report exactly 0, and 38 more (21.7%) lie in (-1e-5, 0), every one of them
  a multiple of 2^-23 (-1.19e-7, -2.38e-7, ...). The alternatives bound the true values:
  behind the 23 zeros, S is at least 7.6e-11 to 1.4e-7 (second-best log probabilities from
  -23.9 to -16.75: margins up to about 23); the quantized ones are 37% below to 80% above that
  bound.
- Prompt logprobs of a 738-token prose passage
  (`experiments/2026-10-19-sink-rejection-v2/runs/x203-served-prompt-logprobs.json`): 23 of
  737 positions (3.1%) report exactly 0, 61 (8.3%) lie in (-1e-5, 0).

So on code a third of the sampled tokens carry a log probability that is 0 or quantized, and
the fix changes all of them; on prose, a tenth.

## The size of each error (CPU emulation)

`uvx --with numpy python emulate.py --rows 96` emulates the kernels' FP32 order (Triton's
block layout) with ex2.approx as a rounded exp and libdevice's logf/log1pf/expm1f as correctly
rounded, against Eq. 5 in FP64 on the same logits.

**The top token's log probability by margin**, a 129280-column row (others N(0, 2)), the
argmax at the row's first, middle or last column (release value and relative error):

| Margin (nats) | Reference | Release: first / middle / last column | Fix, worst relative error |
| --- | --- | --- | --- |
| 4 | -1.815e-2 | 9e-6 / 3e-6 / 4e-6 | 7.6e-8 |
| 8 | -3.354e-4 | 3.8e-4 / 3.8e-4 / 3.8e-4 | 2.5e-7 |
| 10 | -4.540e-5 | 4.2e-4 / 4.2e-4 / 2.2e-3 | 2.9e-7 |
| 12 | -6.144e-6 | -1.31e-6 (79%) / -4.17e-6 (32%) / -6.08e-6 (1%) | 2.5e-7 |
| 13 | -2.260e-6 | 0 / -1.07e-6 (53%) / -2.26e-6 (0.2%) | 2.7e-7 |
| 14 | -8.315e-7 | 0 / -3.58e-7 (57%) / -8.34e-7 (0.4%) | 5.6e-7 |
| 15 | -3.059e-7 | 0 / -1.19e-7 (61%) / -3.58e-7 (17%) | 5.6e-7 |
| 16 | -1.125e-7 | 0 / 0 / -1.19e-7 (6%) | 2.8e-7 |
| 16.6 | -6.176e-8 | 0 / 0 / -1.19e-7 (93%) | 3.2e-7 |
| 17.5 | -2.511e-8 | 0 / 0 / 0 | 6.5e-7 |
| 20 | -2.061e-9 | 0 / 0 / 0 | 4.9e-7 |
| 40 | -4.248e-18 | 0 / 0 / 0 | 6.3e-7 |
| 80 | -1.805e-35 | 0 / 0 / 0 | 1.2e-6 |

Past about 87 nats the others' exponentials are FP32 subnormals: the fix's absolute error is
then at most a subnormal a column (129280 x 2^-149 = 1.8e-40), and past about 103 the
reference itself is below FP32's smallest subnormal.

**Every other output** (normal logits of sd 2, 4 and 8, the argmax and 20 random tokens, 6,048
outputs): both arithmetics' largest errors are 1.7e-6 to 4.2e-6, the rounding of the shifted
logit at its magnitude; the fix's error never exceeds the release's by more than an ULP of the
output and of the log-sum-exp plus 4e-6 of S / (1 + S) (the two sums round independently; the
largest excess was 0.86 of that allowance).

**Block verification's threshold** h = p(1 - m) / (1 - p m) over logs from 0 to -3 and -inf
(169 pairs): the release's formula is off by up to **1.0** (log p = -1e-9, log m = 0: the true
h is 0, the release's 1); with both logs at -3e-8 the release gives 1.0 for 0.5; at log p =
-1e-7, log m = -3e-8 it gives 0.5 for 0.23. The fix is within 1.5e-7 everywhere. Verification's
draft log probability at margin 14 is 0 in the release whatever the maximum logit (8, 20 or 40),
and -8.3e-7 (relative error 5e-7) in the fix; at margin 10 the release gives -4.578e-5 for
-4.540e-5 (0.8%).

**What that does to sampled outputs.** Block verification is lossless only with exact
thresholds. The release's errors live where every verified draft is near-certain, and there
the final draft's acceptance h = p decides the outcome unless it rejects, which happens with
probability 1 - p (the drafts' combined uncertainty, 1e-8 to 1e-5 in that regime). So the
distortion of the sampled distribution is at most about 1e-6 a verification step, consistent
with triage's bound; the fix removes it.

## The change (vLLM `softmax-fix`, a7b65a167 to 80e8c188c)

- `tilelang_logprob.topk_log_softmax`: the sum pass adds exp(x - max) only for columns that are
  not the maximum and counts the maximum's columns (exact in FP32); S adds the ties beyond the
  first (exp(max - max) each: 1, or NaN for an infinite maximum, as the paper's z - z_max);
  lse = log1pf(S); a token's value is (x - max, exactly 0 at the maximum) - lse. A row takes
  a block of 1024 threads, two vector loads each a step (the release took 128 threads in
  Triton's layout); the maximum's pass reads the row last step first, so the sum starts on
  the steps still in L2. One more block reduction (the tie count).
- `tilelang_rejection.local_stats`: a sampled row's block sum leaves one column holding the
  block maximum out. `verify`: S is the block sums rescaled to the row maximum, each with its
  left-out column back (1 + sum) except the first block holding the row maximum; the draft's
  log probability is its shifted logit minus log1p(S); the rejected row's log-sum-exp
  max + log1p(S); the threshold takes -expm1(log m) and -expm1(log p + log m) (IEEE division),
  each step's on its own thread (its cumulative ratio folded in the serial loop's order, the
  same bits), so thread 0 only compares the draws. A row of two or more columns, all -inf, is
  NaN, as in the logprob kernel.
- Unchanged bit for bit: argmaxes and maxima, greedy verification, the uniform draws, resampling
  and insertion. The Triton kernels stay for the configurations the recipe does not use.
- One correct path: no flag, no fallback.

Tests ([vLLM] `tests/v1/worker/test_gpu_tilelang_logprob_softmax.py`,
`tests/v1/spec_decode/test_tilelang_rejection.py`), all against FP64 references of Eq. 5 that
follow the kernels' NaN, infinity and block semantics:

- every output within its rounding bound (the shifted logit's and the output's rounding, an ULP
  of the log-sum-exp, 2e-5 of S / (1 + S), a subnormal a column), with the reference's NaN and
  infinities, over the old suites' shapes, layouts, vocabularies and special values;
- never further from the reference than the release's arithmetic beyond rounding;
- exact where the release collapses: z = [9, -9, -9] gives -3.046e-8 (and 1 - p = 3.046e-8;
  the release 0), as a logprob and as a verified draft (cumulative -3.046e-8, -6.092e-8; the
  release 0, 0); margins 4 to 100 with the argmax at the first, middle and last column;
- block verification's counts equal the FP64 reference's decisions from the same uniform draws
  wherever no draw lies within 1e-4 of its threshold (at least 90% of sampled requests
  decided); the threshold function within 1e-6 relative (plus 1.4e-7 |log p|, the prefix
  ratio's exponential) on a grid where the release's formula, emulated in FP32, is off by up
  to 1;
- a row's (a request's) bits do not depend on its batch or the run; greedy requests, sampled
  tokens, draft lse, resampling and insertion bit for bit as Triton's.

Benchmarks: `benchmark_tilelang_logprob_softmax.py` and `benchmark_tilelang_rejection.py` take
`--baseline DIR` (the release's two modules) and gate the fix against the release's TileLang
kernels, and without it against Triton's. The Eq. 5 helpers are branch-free PTX: with a C
conditional, nvcc predicated the exponential behind the compare.

## GPU results (dgx4, 2026-10-25)

dgx4 (uncabled, serving nothing) ran the kernel bundles with `scripts/lab.py kernel-local` from
a checkout of this branch under `~/spark-lab/agents/softmax-fix/spark-ds41f`, outputs under
`~/spark-lab/agents/softmax-fix/out/`, on image `spark-ds41f:19f2c20ed4d6-next-5`
(`sha256:dce880d88d8e`, the candidate's package tree 26382f5cf; -next-6 differs only in two
test files).

| When (CET) | Bundle | spark-ds41f / vLLM | Verdict |
| --- | --- | --- | --- |
| 19:58 | `softmax-logprob` | 887eb67 / a7b65a167 | **fail**: tests pass (874 logprob, 44 sink); the gate against the release fails at 1 to 16 rows (FP32 30 us against 25, bf16 30 against 20); 3x faster than Triton |
| 20:07 | `softmax-verification` | 887eb67 / a7b65a167 | **fail**: tests pass (427 verification, 146 rejection sampler, 1 skipped); `rejection_sample` 2-3% slower than the release at 16 FP32 and 48 bf16 rows |
| 20:43 | `softmax-logprob` | 243b87c / 80e8c188c | **pass** in 56 s: 874 + 44 tests; no case slower than the release or Triton (`out/softmax-logprob-2`) |
| 20:47 | `softmax-verification` | 243b87c / 80e8c188c | **pass** in 74 s: 427 + 146 tests (1 skipped); no case slower than the release or Triton (`out/softmax-verification-2`) |

Command, from the checkout:
`python3 scripts/lab.py kernel-local experiments/2026-10-25-softmax-fix/bundles/<bundle> --out ~/spark-lab/agents/softmax-fix/out/<bundle>-2`.

**Root cause of the first run's slowdown.** In SASS (`cuobjdump -sass` of the compiled
kernels), nvcc had compiled the helpers' C conditional `x == m ? 0 : exp(x - m)` by predicating
the exponential behind the compare, with predicate shuffles and moves: twice the release's
FSETP/FSEL a column. And the release's one 128-thread block a row reads the row in about 32
round trips to memory, one after another, with the compute in between, so every added
instruction lengthened the latency. The helpers are now branch-free PTX. A row takes 1024
threads, each with two vector loads a step, and the maximum's pass reads the row last step
first, as the release did. That way the sum starts on the steps still in L2, which a forward
pass had evicted at 96 rows. Verification's thresholds moved off thread 0's serial loop.

Layouts timed against the release on dgx4 (dev bundles outside git; FP32 96 rows, release
about 198 us):

| Threads x loads | Forward maximum pass | Reversed |
| --- | --- | --- |
| 512 x 4 | 258 | 200-205 |
| 512 x 4, two blocks an SM | 337 (spills) | - |
| 256 x 4 | 336 | 206-208 |
| 1024 x 2 | 229 | 199-202, level; chosen |
| 1024 x 4 | - | 209-210 (bf16 spills) |

**The passing run's logprob kernel against the release's** (us, the 95% interval of the
paired difference):

| Logits | Rows | Tokens | Release | Fix | Difference |
| --- | --- | --- | --- | --- | --- |
| FP32 | 1 | 1 / 20 | 24.6 / 24.8 | 15.9 / 16.1 | -8.7 / -8.7 |
| FP32 | 16 | 1 / 20 | 25.6 / 25.7 | 18.8 / 19.0 | -7.5 / -7.1 |
| FP32 | 96 | 1 / 20 | 197.8 / 195.6 | 202.9 / 198.6 | -1.5 to +12.1 / -2.5 to +3.3 (level) |
| bf16 | 1 | 1 / 20 | 20.2 / 20.2 | 13.4 / 13.4 | -6.9 / -6.9 |
| bf16 | 16 | 1 / 20 | 20.4 / 20.6 | 13.8 / 13.9 | -6.7 / -6.7 |
| bf16 | 96 | 1 / 20 | 53.7 / 52.7 | 54.0 / 55.4 | -2.7 to +2.9 / -2.3 to +4.8 (level) |

Against Triton it is 2.7 to 6.4 times faster. Verification (`local_stats`, `verify`,
`resample`, and `rejection_sample` end to end, 1 to 96 rows, FP32 and bf16, sampled and mixed
temperatures) is level with the release everywhere. For example, `rejection_sample` at 16 FP32
sampled rows takes 48.3 us against 46.9 (-0.7 to +3.2), and at 96 bf16 sampled rows 126.4 against
125.7 (-1.5 to +1.7).

## What changes in outputs (the next release)

- **Temperature 0 text: identical.** Greedy verification and every argmax are unchanged. The
  release protocol's text references hold: `c9cf9dbe` / `c6bb4f4e` at tp4-ring
  (`4d143977` / `12fd8257` at tp3-triangle, docs/current-state.md).
- **Logprob values: changed, by design.** Every served logprob moves by up to a few 1e-6 (the
  release's log-sum-exp error), and the exact zeros become -S (7.6e-11 to 1.4e-7 in the code
  sample above). The determinism gate's logprob digests therefore need new references in the
  release that takes this fix: `3eab5112` / `db25fde3` at tp4-ring and `62313cb5` / `9742ac2c`
  at tp3-triangle will not reproduce. The serving check records the new ones.
- **Seeded temperature 1:** the sampled digest (candidate 5: `80b3b3f3`) includes logprobs and
  changes. Its tokens change only where a draw falls between the release's threshold and the
  fixed one: about 1e-6 a verification step, so a 192-token, eight-prompt digest most likely
  keeps its text; `logprob_margins.py` compares the tokens.
- **Other digests that include logprobs** change too: prompt logprobs and the generation
  controls' logprob-id scenarios (`controls_digest.py`).
- **Speed:** gated equal or better against the release's kernels.

## Lab specs

The kernel bundles passed on dgx4 (above); [softmax-kernels.json](softmax-kernels.json) runs the
same bundles in a lab window. The serving check needs the cluster and is still to run: its arm
is candidate 5 at tp4-ring, so with the nodes cabled as the TP3 triangle it waits for the ring,
or takes a TP3 arm of the release once one is merged.

Regenerate the bundles and the arm from the vLLM branch with
`python3 make_bundles.py ~/projects/vllm-softmax-fix 26382f5cf softmax-fix`.

- [softmax-kernels.json](softmax-kernels.json): `bundles/softmax-logprob` on dgx1 (logprob,
  verified-token, rank, top-k id, preparation and sink tests; the logprob benchmark against the
  release and against Triton) and `bundles/softmax-verification` on dgx2 (verification tests,
  the rejection sampler's statistical, chunking, int64 and adaptive-verification tests; the
  rejection benchmark against the release and against Triton), on image -next-5 with the fix's
  four package files as an overlay and the release's two modules at `/b/release`.
  `tests/v1/sample/test_logprobs.py` needs the repository conftest (tblib), so the release tests
  cover its kernel tests.
- [softmax-serving.json](softmax-serving.json): the fix as an overlay over candidate 5
  ([softmax-tp4.json](softmax-tp4.json)) first, then the candidate itself
  (`experiments/2026-10-25-spark-ds41f-candidate/rc-tp4.json`), each with the determinism gate
  with and without logprobs, the seeded temperature-1 digest, and
  [logprob_margins.py](logprob_margins.py) (twelve temperature-0 prompts with top-5 logprobs,
  prompt logprobs over a repetitive code passage, eight seeded temperature-1 prompts).
  [tables_arms.py](tables_arms.py) reads them: text digests equal (and the protocol's), the new
  logprob digests, and position by position the size of every logprob difference, how many
  values each build reports as exactly 0 or on the 2^-23 grid, and what the fix reports where
  the release reported 0.

Pass: the kernel bundles pass with both gates; at tp4-ring the fix's text digests equal the
release's with and without logprobs, every arm identical within itself, and no exact zeros left
except where S underflows FP32.

## Not changed, and why

- **FP32's subnormal range.** Past margins of about 87 nats the others' exponentials are
  subnormal and S loses relative precision (absolute error up to 1.8e-40; the log probability
  is below 1.6e-38 there). The paper's LSE shifts the others by their own maximum, which needs a
  top-two pass over the row; for values this small it is not worth a hot-loop change.
- **The uniform draw's resolution near 1.** A draw is an FP32 value with 2^-24 steps below 1, so
  the last draft's acceptance u <= p resolves p to about 6e-8. Changing the draw changes every
  seeded sample; it is not this paper's bug.
- **Top-p's pivot.** vLLM's top-p (which the TileLang port follows bit for bit) converts the
  boundary probability back to a logit, `log(p * sum) + max`, which rounds at an ULP of the
  maximum logit, and finds the boundary's ties with a tolerance of 1e-9, below an ULP of any
  logit past 0.008. A probability-to-logit round trip, not collapse or cancellation; a separate
  review.
- **The Triton paths the recipe does not take** (probabilistic drafts, FP64 draws, watermark,
  synthetic rates) keep their arithmetic.
- **The model's softmaxes** follow DeepSeek's reference (row 9).

## Coordination

No file overlaps with the kernel-audit (`vllm-kernel-audit`: attention, metadata) or MoE overlap
(`vllm-moe-overlap`: routed experts) branches. The vLLM branch is local; after the lab passes it
becomes one patch of the next release's series.
