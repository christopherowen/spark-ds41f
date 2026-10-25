# Candidate 5: spark-ds41f-1022 with candidate 4's changes and the TileLang completion

Owner, 2026-10-25: shift to release stabilization and freeze the scope; build one fixed,
reproducible candidate, validate it, and benchmark it against production with the
published release benchmark. Promotion needs the owner's go.

**Status (2026-10-25 18:32 CET): promoted as `spark-ds41f-1025` and serving on the ring at 1M**
(owner: "start with TP=4 and promote when ready"; merged as 2831bc8a, #596).
- **Deployed 18:28-18:32 CET** under a deploy hold. spark-ds41f-1022 stopped at 18:28:15, and
  the API was ready at 18:31:08: 88.7 s after launch, about 3 minutes of downtime.
- **Checks:** `doctor --live` OK (its only warnings are the weight-shard extent counts);
  KV 6,512,700 tokens (vLLM's startup figure, 6.21x at 1M).
- **Boot memory service:** `spark-ds41f-kernel-memory` replaced `spark3-kernel-memory` and is
  active on every node.
- **First-prefill probe** (step 5b, below): a fresh boot's first long prefill costs about 0.1 s, once.
- **Not qualified yet:** the triangle on this image (#582, after the re-wire).

The sections below record how the candidate got there.

## What it is

- **vLLM:** production's series (spark-ds41f-1022, patch head 8fc15c18f), then the
  `tilelang-ports` branch from there through 26382f5cf, as patches 0300-0488
  ([vllm/series](vllm/series)). Those 189 commits are:
  - candidate 4's 38 (0228-0265 in experiments/2026-10-23-next-release, the same
    commits);
  - the TileLang completion ([experiments/2026-10-24-tilelang-completion](../2026-10-24-tilelang-completion/README.md)):
    every kernel serving launches is TileLang except NCCL and a few rare-path ATen
    copies; no B12X preparation runs under TileLang; the runner warm-up drops the Triton
    top-k/top-p compile; the JIT monitor has an off mode.
- **The build:** the series replays onto the pinned base with the build's fixed committer
  to patch head 5b0ec591 and tree 043c1fcb, the same tree as the branch the lab tested
  through its overlays ([source.json](source.json),
  [upstreams.lock.json](upstreams.lock.json)). Image `spark-ds41f:19f2c20ed4d6-next-5`.
  Every other source (b12x, NCCL, TileLang, TileKernels, sparknet, CUTLASS) is candidate
  4's.
- **The recipe:** [cluster.json](cluster.json), candidate 4's (rc3) with two changes:
  - **`--jit-monitor-mode off`.** Under TileLang the monitor had nothing left to catch,
    and it cost 0.4% at c8 (the tilelang-completion README's A/B).
  - **21 settings for kernels serving does not use are removed:**
    - the B12X and CuTe DSL compiler settings;
    - the FlashInfer sampler and workspace;
    - torch.compile's AOT and Inductor settings;
    - attention settings the TileLang backend never reads;
    - the NVFP4 heads' defaults;
    - the CUDA-graph memory estimate that a fixed KV size skips.
- **Arms:** [rc-tp4.json](rc-tp4.json) (tp4-ring) and [rc-tp4-dev.json](rc-tp4-dev.json)
  (the release suite's dev-mode RPC). The candidate arm's serve arguments and environment
  equal the tested cleanup overlay's
  ([ports-clean-tp4.json](../2026-10-24-tilelang-completion/ports-clean-tp4.json)).

## Validation (docs/release-benchmark.md)

1. **[candidate-tp4.json](candidate-tp4.json), one window:**
   - the image built on dgx2 and copied over the ring;
   - the release tests ([bundles/release-tests](bundles/release-tests): 97 gate files;
     the new and changed test files are classified in `config/release-tests.json`);
   - then the candidate at tp4-ring, which runs:
     - the determinism gate (`--wide` with and without logprobs, and 16/8/4 distinct
       streams), whose outputs must hash as 1022's;
     - candidate 4's qualification battery: control spellings, controls, TTFT trace,
       memory, sink counters and probes;
     - the generation controls with logprob ids, prompt logprobs, seeded sampling and
       images;
     - the startup log: no B12X preparation, no temporary KV cache, no Triton compile.
2. **[release-tp4.json](release-tp4.json), the release comparison:** the `release`,
   `single-stream` and `sampled` suites on candidate 5, spark-ds41f-1022 and candidate 5
   again, on dev arms. The single-stream suite is the published single-stream figure
   (~74 tok/s), a different measurement from the tilelang-completion checks' c1-distinct
   (~95 tok/s: 12 prompts, 256 tokens).

3. **[vision-eval-candidate5.json](../2026-10-25-vision-eval/vision-eval-candidate5.json):**
   the built image against production 1022 on the 2283 public-benchmark items (concurrency
   16).

The owner wants the release as close to perfect quality as possible: the full protocol,
with every anomaly root-caused.

## Known limitations

- **At most 32 top logprobs**, a compatibility change the owner accepted: the TileLang
  top-k ids kernel takes k up to 32, and `--max-logprobs` above 32 or -1 is refused at
  startup. Production runs the default of 20.
- **The DSpark drafter's plain argmax** trails Triton's by about 0.05 µs at 1-16 rows, once
  a decode step (accepted).
- **Image outputs differ from B12X's tower.** The TileLang tower follows DeepSeek's
  reference arithmetic. On 2283 public benchmark items its scores match B12X's within
  about one point (experiments/2026-10-25-vision-eval).
- **Seeded sampled outputs and batching:** see the seeded penalties investigation
  (tilelang-completion README, run 26) and this record's "The seeded penalties variation": an
  existing limitation of adaptive verification, spark-ds41f-1022's too.
- **6.21 full 1M contexts on the ring, not 8.18:** the 16,384 budget's admission bound (see
  candidate 5's window); the release notes explain it.

## Candidate 5's window (2026-10-25 10:30 CET, run `candidate-5-tp4`)

- **Image** `spark-ds41f:19f2c20ed4d6-next-5` (sha256:dce880d88d8e), built on dgx2 at 10:40 CET
  from deployment commit c3a63dc4; labels: vLLM patch head `5b0ec591`, tree `043c1fcb`. One ID
  and the same 54 layers on dgx1-dgx4.
- **Release tests: 6 failed** in chunk 5, in two files the kernel bundles never ran. Both are
  stale tests of ports in this candidate; no package code is wrong:
  - `tests/v1/attention/test_b12x_v41_ced_metadata.py::test_packing_trailing_dimensions_invalid_rows_and_scatter`
    (both parametrizations): `ValueError: zero_ fills contiguous 16-byte packs`. Since vLLM
    3622c1185 (in 0300-0488) `scatter_rows` zeroes its output with TileLang's fill of whole
    16-byte packs, which refuses anything else by design (its own test checks the refusal).
    Every serving call is whole packs: CED's hidden rows (BF16, 5120 a row) and the sink
    columns (FP32, `hc_mult` 4, 16 bytes a row). The round trip's 7 x 3 x 5 FP32 rows are
    420 bytes.
  - `tests/v1/spec_decode/test_dflash_dcp.py::test_dflash_uses_draft_group_dcp_slot_parameters`
    (all four): `kernel padding_rows input is_padding device_type mismatch, expected cuda`.
    Since d0f5f9e5f the DFlash speculator writes its padding rows with a TileLang launch;
    the test's speculator stub holds host tensors. The real speculator allocates them on the
    device, and DSpark (the recipe's drafter) is not DFlash.
- **Release tests, the rest:** 5,724 passed and 160 skipped in 2,311 s, every skip with its
  reason (Hugging Face models offline, ROCm, two GPUs, no GLP test file, a draft budget below the
  scheduled drafts, and the indexer NaN contract's strict xfail of 2026-10-14). #509's eight
  vocabulary-parallel draft-seed mismatches do not reproduce (`test_dspark_vocab_parallel.py`
  passes).
- **Boot:** API ready in 88.6 s (spark-ds41f-1022's TP4 boots: 87-94 s). Every rank prepared 247
  TileLang calls in 11.0-11.1 s; engine init 28.1 s. No TileLang or Triton cache file was written
  during the boot (each node's caches counted before and after), so nothing compiled: the
  overlay's 24.4 s (release check round 1) was the new kernels' first compile into the
  persistent cache. No B12X preparation, no temporary KV blocks, no JIT monitor. Zero error lines
  on any rank.
- **KV cache:** 6,512,700 tokens, 6.21 full 1M contexts, from the 10.5 GiB pool. That is the
  16,384 budget's (every saved boot at 16,384 reports it, candidate 4's image included; every one
  at 8,192 reports 8,580,566, 8.18): vLLM charges each request the sliding-window caches' window
  plus two prefill chunks in flight, at admission too. spark-ds41f-1024's draft recorded 8.18
  for candidate 4 from `mem_snapshot.py`'s log excerpt, which reads the container
  `dsv41-karmic-kraken` (a constant in the script) and so, since the rename, the stopped
  production container's log.
- **Determinism gate: identical**, zero token and log-probability differences in all six
  scenarios, with log probabilities `3eab5112` / `db25fde3` and as text `c9cf9dbe` / `c6bb4f4e`:
  spark-ds41f-1022's hashes.
- Distinct streams (a first look): 16 streams 254.5 tok/s ±0.3%, 8 streams 192.5 ±1.1%, 4 streams
  145.5 ±0.6% (the overlay's 8 streams: 192.1).
- Control spellings: all eight locations pass. Controls probe: all four pass
  (`add_generation_prompt`, `continue_final_message`, top-level `thinking` disabled beside
  `reasoning_effort` high, `thinking` "maybe" refused with 400).
- TTFT trace: isolated 122.2 ms, p50 216.9 ms at four streams and 284.4 ms at eight
  (spark-ds41f-1022's gate: 121.7 / 213.6 / 281.4), no foreign requests.
- Memory (MemAvailable after the gate's traffic): dgx1 20.76 GiB, dgx2 22.13, dgx3 22.45, dgx4 22.30
  (candidate 4, the same 16,384 budget: dgx1 18.94; spark-ds41f-1022 at 8,192: 22.99).
- Sink counters: zero flags on ordinary traffic. Probes: `Say hi.`, `Thanks!` and `ok` flagged,
  nothing collapsed, every rank agreeing; each rank's line reads "N flagged rows in executed steps
  ... vllm:sink_* counts committed outputs only" (#489).
- Generation controls (`controls_digest.py`): no failure; logprob ids succeed, alone and at eight
  streams alike; 11 of 12 digests equal the overlay's (release check round 1). The twelfth is
  `penalties-sampled/c8`, which differs between boots of one build too (see the seeded
  penalties variation).
- Prompt logprobs: succeed, the same digest alone and at eight streams.
- Seeded temperature 1: reference digest `80b3b3f3`, the overlay's; a repeat is identical. Eight at
  once differ from each request alone, as on the overlay and candidate 4.
- Images: digests equal the overlay's candidate digests (`92679390` alone and at four).
- The driver stopped after this window on the failed release tests, as its brief requires.

## Candidate 6 ([rc6/](rc6))

The smallest fix for candidate 5's release-test failures: two vLLM test patches, package files
unchanged, as run 25's test fixes were (26382f5cf).
- **0489** runs the CED round trip on 7 x 3 x 4 FP32 rows (whole packs), keeping its trailing
  dimensions, transpose and invalid rows.
- **0490** gives the DFlash slot test a host fill for its host tensors' padding rows, as the
  collectives' tests have (26382f5cf); on CUDA tensors the kernel runs.

The series ([rc6/vllm/series](rc6/vllm/series)) is candidate 5's with 0489-0490. Replayed with the
build's fixed committer it gives patch head `11171760`, tree `6f8e062f`
([rc6/source.json](rc6/source.json), [rc6/upstreams.lock.json](rc6/upstreams.lock.json)); the
diff from candidate 5's tree is the two test files. Image `spark-ds41f:19f2c20ed4d6-next-6`.
[rc6/rc-tp4.json](rc6/rc-tp4.json) is candidate 5's arm with only the image, its vLLM tree label
and the lock changed. Release tests: 154 selected files, 97 gate, 0 classification problems
([rc6/release-tests-gate-manifest.json](rc6/release-tests-gate-manifest.json),
[rc6/bundles/release-tests](rc6/bundles/release-tests)).

To qualify it: [rc6/candidate-tp4.json](rc6/candidate-tp4.json), candidate 5's window at the new
image (build, release tests, gate, qualification). Candidate 5's other checks (vision,
the seeded penalties variation, the release comparison) ran on the same package files.
[rc6/release-tp4.json](rc6/release-tp4.json) repeats the release comparison on candidate 6 if the
owner wants it measured on the image itself. After its window,
[rc6/fill_baseline.py](rc6/fill_baseline.py) writes the built image's ID, build time and
deployment commit into the promotion's baseline manifest (refusing unless every node holds one
image ID and its vLLM tree is the manifest's).

**The fixes, ahead of candidate 6's image** (run `candidate-6-tests-preview`, 11:44 CET):
candidate 5's release-test bundle on candidate 5's image, its chunk 5 (the ten files that ran
together, the two fixed among them) with 0489-0490's test files mounted over the image's
copies: 336 passed, 40 skipped, 0 failed (candidate 5's chunk 5: 330 passed, 40 skipped, the 6
failures). The bundle is outside the repository (`~/spark-release-driver/bundles/c6-tests-preview`
on dgx1), since the lab runs only origin/main's specs; candidate 6's own release tests run in its
window.

### Candidate 6's window (run `candidate-6-tp4`, 2026-10-25 13:43-14:43 CET)

The build, release tests, gate and qualification ran on candidate 6's own image,
`spark-ds41f:19f2c20ed4d6-next-6` (image `sha256:82b071b57485`, built 12:54 UTC, the same ID on
all four nodes), at deployment commit `8ee6ce53`:
- **Release tests (dgx2):** 5,730 passed, 160 skipped (each with its reason), 0 failed.
- **Determinism and greedy gate:** identical to spark-ds41f-1022's: `3eab5112` / `db25fde3`
  with log probabilities, `c9cf9dbe` / `c6bb4f4e` as text.
- **Controls:** no failure; logprob ids and prompt logprobs pass.
- **Images:** the digest is the overlay's (`92679390`).
- **Startup:** 247 TileLang calls prepared in 11 s a rank; API ready in 87 s; no B12X
  preparation and no compile lines.

### Candidate 6's release comparison (run `candidate-6-release-tp4`, 2026-10-25 14:46-18:20 CET)

[rc6/release-tp4.json](rc6/release-tp4.json): suites release, single-stream and sampled in one
session. The arms ran in this order: candidate 6, spark-ds41f-1022 (its frozen recipe, dev
mode), candidate 6 again. Verdicts are against the session's own repeat (candidate 6 → candidate
6 again).

| Metric | 1022 → candidate 6 | ±95% | Repeat | Verdict |
| --- | ---: | ---: | ---: | --- |
| Greedy responses (256) | identical | | identical | same |
| Teacher-forced KL (65,536 positions) | 0 | | 0 | same |
| Decode grid, geometric mean of 20 shapes | -2.38% | 0.15% | +1.14% | better |
| c16 output tok/s | +0.12% | 0.81% | +0.00% | same |
| c8 output tok/s | +0.88% | 0.79% | -0.29% | same |
| Poisson 0.7/s output tok/s | +0.01% | 17.72% | -0.01% | same |
| Prefill 32,768 | +12.80% | 0.95% | +0.16% | better |
| Prefill 131,072 | +13.85% | 0.87% | +0.26% | better |
| Prefill 262,144 | +13.73% | 0.79% | +0.16% | better |
| Prefill 524,288 | +12.89% | 0.34% | +0.30% | better |
| Prefill 1,000,000 | +11.81% | 0.49% | +0.26% | better |
| One stream (c1) output tok/s | +2.11% | 3.82% | -0.06% | same |
| Sampled c16 / c8 / c1 | +0.40% / +0.70% / +2.65% | 0.90 / 1.10 / 4.94% | 0.00 / +0.37 / -0.03% | same |

- **Rates:** candidate 6's decode grid steps are 1.4-3.5% shorter. At one stream it runs 75.3
  tok/s against 73.8, at 33.0 engine steps a second (+1.95%, better); its tokens a step are
  unchanged (2.28).
- **Prefill:** 6,762 / 6,796 / 6,482 / 5,863 / 4,942 tok/s from 32K to 1M, against
  5,994 / 5,969 / 5,700 / 5,194 / 4,420. This is the ring's 16,384 budget (8,192 on 1022) with
  the CED skip.
- **Time to first token:** at eight streams it is level, 297 ms against 295 (p50). The earlier
  watch item (281 ms) came from another session's 1022 run. Under Poisson the p50 is 280 ms
  against 292.

Full tables: `release-table.txt` in each suite reference, below.

The suite references are candidate 6's first arm, without the model step's generated text, plus
each suite's table:
- [manifests/suites/2026-10-25-spark-ds41f-1025-tp4](../../manifests/suites/2026-10-25-spark-ds41f-1025-tp4/)
- [-single-stream](../../manifests/suites/2026-10-25-spark-ds41f-1025-tp4-single-stream/)
- [-sampled](../../manifests/suites/2026-10-25-spark-ds41f-1025-tp4-sampled/)

## Vision on the built image (run `vision-eval-candidate5`, 2026-10-25 11:45-12:07 CET)

Candidate 5's image against spark-ds41f-1022, the 2,283 public-benchmark items of
[experiments/2026-10-25-vision-eval](../2026-10-25-vision-eval/README.md), concurrency 16, every
request answered:

| benchmark   | items | candidate 5 | spark-ds41f-1022 | answers differ | 1022 better / worse | difference, 95% interval |
|-------------|------:|------------:|-----------------:|---------------:|--------------------:|-------------------------:|
| ChartQA     |   300 | 86.00 | 85.67 |  14 |   1 / 2 | -0.33 [-1.67, +0.67] |
| DocVQA      |   300 | 94.01 | 94.74 |  27 |   6 / 7 | +0.73 [-0.31, +2.01] |
| OCRBench    |  1000 | 81.10 | 81.80 | 549 |  16 / 9 | +0.70 [-0.30, +1.70] |
| RealWorldQA |   383 | 77.02 | 77.02 |  13 |   6 / 6 | +0.00 [-1.83, +1.83] |
| TextVQA     |   300 | 85.67 | 86.20 |  10 |   3 / 2 | +0.53 [-0.73, +2.00] |
| all         |  2283 | 83.36 | 83.79 | 613 | 32 / 26 | +0.43 [-0.14, +1.02] |

**No benchmark separates the two towers**, as in the overlay run. The table is that run's to the
digit because the answers are: the built image's 2,283 answers are identical to the ports
overlay's (at concurrency 16 here, 8 there, so they do not depend on the batch), and 1022's are
identical to candidate 4's (both B12X's tower). The image ships the tower that was evaluated.

## The seeded penalties variation (run `seeded-penalties-variation`, 12:06-12:15 CET)

[penalties-variation.json](../2026-10-24-tilelang-completion/penalties-variation.json): the
penalties-sampled scenario (temperature 1, repetition 1.15, frequency 0.2, presence -0.1, seed
20261024 + i, 96 tokens) three times a boot, alone and eight at once:

| arm | alone, three runs | eight at once, three runs |
| --- | --- | --- |
| candidate (overlay `ports-nomon-tp4.json`) | `a75947f7` x3 | `79de80d9`, `dca91313`, `105c3a9f` |
| candidate, adaptive verification off | `8207f323` x3 | `8207f323` x3 (= alone) |
| spark-ds41f-1022 (`production-tp4.json`) | `b75be6b4` x3 | `ba4480ec` x3 |

**Production is stable where the candidate is not: by the brief's rule, a regression.** Without
adaptive verification the candidate is stable and batch-invariant (eight at once equal alone).
Production depends on the batch (its eight-at-once digest differs from its alone digest) but
repeats. The candidate's eight-at-once digests are drawn from a few recurring values
(`79de80d9` also in candidate 5's window, `dca91313` in release check round 1).
**Root cause: an existing limitation, which the test procedure hid on spark-ds41f-1022.** Four runs
followed (`seeded-penalties-bisect`, `-diagnosis`, `-budget-trace`; scripts and arms in
`~/spark-release-driver/` on dgx1, arms under `results/private/lab/arms/`):

1. **Bisection** (three penalties runs a boot, `controls_digest.py`'s procedure):

   | arm | alone | eight at once |
   | --- | --- | --- |
   | candidate 5, 16,384 budget | `a75947f7` | `1f94f42d`, `cd8cc5b6`, `cd8cc5b6` |
   | candidate 5, 8,192 | `a75947f7` | `cd8cc5b6`, `cd8cc5b6`, `1bd37da7` |
   | candidate 4, 16,384 | `a75947f7` | `e5844620`, `1bd37da7`, `cd8cc5b6` |
   | candidate 4, 8,192 | `a75947f7` | `cd8cc5b6`, `79de80d9`, `cd8cc5b6` |
   | candidate 4, 0228's `spec_step_seed` returning the seed | `b75be6b4` | `ba4480ec` x3 |
   | spark-ds41f-1022, 16,384 | `b75be6b4` | `ba4480ec` x3 |
   | spark-ds41f-1022, 8,192 | `b75be6b4` | `ba4480ec` x3 |

   The budget does not matter, and candidates 4 and 5 behave alike (same digests): the TileLang
   ports are not involved. With 0228's draw keying neutralized, candidate 4 equals
   spark-ds41f-1022 bit for bit. 0228 (vllm-project/vllm#61161's backport) keys block
   verification's uniform draw by the draft's index in its block, `u = rand(seed ^ (i << 32),
   pos)`, where spark-ds41f-1022 used `rand(seed, pos)`, which re-used a draw when a later round
   re-verified a position (the bias #61161 fixes). Its residual-mass bound is never read.
2. **Diagnosis** (`probe_seeded.py`: the same request bodies, four runs back to back with every
   token kept): **spark-ds41f-1022 varies too.** Eight at once, 7 of 8 requests differ between
   runs with penalties, 7 of 8 when staggered 300 ms apart, 6 of 8 at plain temperature 1;
   candidate 5: 6, 7 and 8 of 8.
3. **Budget trace** (the adaptive budget's DEBUG line per step, four runs eight at once): every
   build's output follows the batch's composition when its first decode steps run.

   | build | draft rows of the first decode step, runs 0-3 | outputs |
   | --- | --- | --- |
   | spark-ds41f-1022 | 10, 20, 15, 10 | three different; runs 0 and 3 identical (`b5f27305`) |
   | candidate 4, 0228 neutralized | 5, 10, 10, 10 | runs 1-3 identical, and equal to 1022's 10-row runs (`b5f27305`) |
   | candidate 5 | 5, 10, 5, 25 | runs 0 and 2 identical |

   How many of the eight requests have finished their prefill when the first decode steps run
   depends on arrival timing. Adaptive verification sizes each step's drafts over the whole
   batch (best-first over every request's draft slots, `1 + 0.035 x rows`), so the batch decides
   how a request's positions fall into verified blocks, and block verification's sampled
   tokens depend on that split. With the same composition every build is deterministic; with
   adaptive verification off the split is fixed and eight at once equal alone.

**Classification: an existing limitation, not a regression.** Seeded sampled requests are
reproducible alone, not under concurrency with adaptive verification, in spark-ds41f-1022 as in
this release. The penalties-variation procedure (probe, eight alone, then eight at once) happens to
give spark-ds41f-1022 the same outcome each time; under 0228's keying the compositions it meets
no longer coincide in output. Reverting 0228 would bring the bias back to make one test repeat.
Temperature 0 is unaffected (greedy verification accepts a draft only where it is the argmax, so
the split cannot change it; the gate shows it).

**Validation** (`seeded-penalties-validation`, 12:57-13:09 CET, a fourth set of boots):

| arm | probe alone x3 | `controls_digest.py` x3 | probe, penalties, eight at once x4 | probe, temperature 1, eight at once x4 |
| --- | --- | --- | --- | --- |
| spark-ds41f-1022 | stable | stable (`ba4480ec`) | 6 of 8 requests vary | 7 of 8 vary |
| candidate (overlay), adaptive verification off | stable | stable | none vary | none vary |
| candidate 5 | stable | `79de80d9`, `66514015`, `a8d79ab9` | 7 of 8 vary | 7 of 8 vary |

The probe repeats alone; spark-ds41f-1022 does not repeat under concurrency; without adaptive
verification nothing varies.

## A fresh boot of the built image (run `candidate-5-recheck-tp4`, 13:09-13:19 CET)

Candidate 5 booted again (API ready in 98.6 s; spark-ds41f-1022's TP4 boots today: 75-94 s) and
checked again:
- **Determinism gate: identical** with zero differences, the same four hashes as spark-ds41f-1022's.
- Controls without a failure, 11 of 12 digests as the overlay's (the twelfth is
  `penalties-sampled/c8`, above); prompt logprobs, the seeded reference digest and the image
  digests as before.
- Every rank prepared 247 TileLang calls in 10.9-11.1 s; no error line on any rank.
- **#455's prompt-cache report, live:** a 2,710-token request reports `cached_tokens` 0 (2,560
  created); its repeat with the same cache salt reports `cached_tokens` 2,560.
- Penalties, three runs: alone `a75947f7` x3; eight at once `79de80d9`, `cd8cc5b6`, `cd8cc5b6`.

## GLP steering on (run `candidate-5-glp-tp4`, 13:19-13:26 CET)

Candidate 5 with the synthetic GLP file (`experiments/2026-10-23-next-release/glp/make_glp.py`,
seed 0, at `/cache/glp/synthetic-0.gguf` on every node), as candidate 4's window ran it: every rank
logs "GLP steering on: alpha 0.75, layers [1, ..., 39]" and prepares 248 TileLang calls (one more
than without steering) in 11.0-11.3 s, no error line on any rank. The determinism gate with
steering on is **identical** in all six scenarios, with and without log probabilities (hashes
`7ae204c2` / `f2bbcf45`, as text `893e8b24` / `c6bb4f4e`: steering changes the outputs, and they
repeat). One stream: 109.2 tok/s on the steered text.

## Reports triaged for this release

Every open issue and pull request, the recently closed ones, their comments and the NVIDIA forum
thread (384438) were read on 2026-10-25; the table with each decision is in
[release-notes.md](release-notes.md#reports-triaged-for-this-release).

## Promotion: spark-ds41f-1025

`config/cluster.json` becomes candidate 6's recipe ([rc6/rc-tp4.json](rc6/rc-tp4.json) resolved):
- candidate 5's [cluster.json](cluster.json): spark-ds41f-1022's recipe under the spark-ds41f
  names (#484, replayed), the ring's 16,384 budget and glibc malloc pinned, the prompt-cache
  report (#469, replayed), the JIT monitor off and 21 settings for unused kernels removed;
- image `spark-ds41f:19f2c20ed4d6-next-6` and vLLM tree `6f8e062f`;
- the lock `upstreams.lock.json` names candidate 6's series ([rc6/vllm/series](rc6/vllm/series),
  patch head `11171760`), its source manifest
  `manifests/sources/2026-10-25-spark-ds41f-1025-source.json`.

The baseline is `manifests/baselines/2026-10-25-spark-ds41f-1025.json`. `tp4-ring`'s
`benchmark_reference` is `manifests/suites/2026-10-25-spark-ds41f-1025-tp4`: candidate 6's first
arm of its own release comparison (run `candidate-6-release-tp4`). `tp3-triangle` is not qualified on this image (its last
release is spark-ds41f-1022). spark-ds41f-1022's recipe and lock stay frozen in
`experiments/2026-10-23-next-release/1022` as the rollback.

### Deploying (with the rename)

The rename goes live with the promotion: the new recipe names container `spark-ds41f`, labels
`local.spark-ds41f.*` and `SPARK_DS41F_*` variables, and refuses image -1022-4 (whose labels are
`local.spark3.*`). Already done on every node (2026-10-23 21:09 CET): the cache moved to
`cache/spark-ds41f` with a relative link `cache/kkref -> spark-ds41f`. Still to do: the boot memory
service (`spark3-kernel-memory` is active, `spark-ds41f-kernel-memory` is not installed).

From dgx1, after origin/main carries the promotion commit. `scripts/lab.py window open` refuses
then (the live cluster no longer matches main's recipe), so the deploy takes the hold by hand
(AGENTS.md, shared cluster windows) and removes it at the end:

0. With no other `~/spark-hold.json` and no pending `~/spark-request.json` or
   `~/spark3-request.json`, take the hold:

   ```sh
   python3 -c 'import json,datetime,pathlib; t=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"); pathlib.Path.home().joinpath("spark-hold.json").write_text(json.dumps({"holder": "deploy spark-ds41f-1025", "since": t, "expected_end": None, "heartbeat": t, "rule": "promotion deploy: no GPU work, cluster start, stop or sync by anyone else"}) + "\n")'
   ```
1. Stop spark-ds41f-1022 with its frozen recipe, which names the running container
   (`dsv41-karmic-kraken` and its memory guard); the stopped container is kept for the rollback:

   ```sh
   bin/spark --cluster-config experiments/2026-10-23-next-release/1022/cluster.json cluster stop --apply --parallel
   ```
2. Sync every node to the promotion commit:

   ```sh
   bin/spark --cluster-config experiments/2026-10-23-next-release/1022/cluster.json cluster sync --revision origin/main --apply
   ```
3. Check the cache link on every node: each must show `kkref -> spark-ds41f` and the
   directory (moved on 2026-10-23; checked again on 2026-10-25). If a node shows a `kkref`
   directory instead, run #503's move there:
   `cd ~/projects/spark-ds41f/cache && mv kkref spark-ds41f && ln -s spark-ds41f kkref`.

   ```sh
   for host in dgx1 dgx2 dgx3 dgx4; do ssh "$host" 'ls -ld ~/projects/spark-ds41f/cache/kkref ~/projects/spark-ds41f/cache/spark-ds41f'; done
   ```
4. Replace the boot memory service on every node; the installer does not reboot or touch swap,
   and starting the service applies the values already in force:

   ```sh
   for host in dgx1 dgx2 dgx3 dgx4; do
     ssh "$host" 'cd ~/projects/spark-ds41f && sudo -n bash experiments/2026-10-02-kernel-64k/install-boot-support.sh &&
       sudo -n systemctl start spark-ds41f-kernel-memory.service && systemctl is-active spark-ds41f-kernel-memory.service'
   done
   ```
5. Start the promoted recipe, plan first (`live:` names no engine, since the stopped container
   has the old name), then apply:

   ```sh
   bin/spark --cluster-config config/cluster.json cluster start
   bin/spark --cluster-config config/cluster.json cluster start --apply
   ```

   The installer is not executable (mode 664), so `bash` runs it. A stopped lab container named
   `spark-ds41f`, such as a release comparison's last candidate arm, makes the start refuse;
   `cluster start --replace --apply` replaces it. Both happened in this deploy.

   5b. Run the first-prefill probe as soon as the API is ready, before any other request
   (doctor --live included). Reports say the first prefill after a boot is slower than later
   ones. Our release suites can't show that: their prefill scenario runs after other traffic,
   and there the first prefill of each size is level with its repeats (r6n through 1022).

   ```sh
   experiments/2026-10-25-spark-ds41f-candidate/rc6/first-prefill-probe.sh ~/spark-lab/c6/first-prefill
   ```

   It sends three cold 32K prefills of one document, then one cold 128K. Around each request it
   records every node's MemAvailable and NVMe reads, plus the server's prompt-token counter, so
   traffic from another client is visible. It then lists any kernel compile or preparation line
   logged during the probe. Compare its rates with candidate 6's warm prefill in the release
   comparison (32K 6,762 tok/s, 128K 6,796). Record the result here even if the probe is level.

   **Result (18:31-18:32 CET, no other traffic: the prompt-token counter started at 0):**

   | Request | Time to first token | Rate |
   | --- | ---: | ---: |
   | First 32K | 4.931 s | 6,644 tok/s |
   | Second 32K | 4.832 s | 6,780 tok/s |
   | Third 32K | 4.831 s | 6,782 tok/s |
   | First 128K | 19.190 s | 6,800 tok/s (the comparison's warm figure: 6,796) |

   - **Effect:** the first long prefill after a boot costs about 0.1 s (2%), once.
   - **Not memory:** free memory moved by under 10 MiB on any node.
   - **Not disk:** every 32K request read the same 1.8 GB of Engram rows from NVMe
     (O_DIRECT, so there is no page cache to warm).
   - **Not compiling:** the containers logged no compile or kernel-preparation line.
   - **Conclusion:** a much slower first prefill, as reported, is not this boot's behaviour.
     A repeated prompt hitting the prefix cache would explain a large difference.
6. `bin/spark --cluster-config config/cluster.json doctor --live` must report the live cluster
   matching (`spark-ds41f`, `spark-ds41f-memguard`), with no kernel memory service warning.
7. Release the hold (`rm ~/spark-hold.json`) and record the deploy (time, API ready, KV
   tokens, `doctor --live`) in this README, as spark-ds41f-1022's record does.

Rollback: `bin/spark --cluster-config config/cluster.json cluster stop --apply --parallel`, then
`bin/spark --cluster-config experiments/2026-10-23-next-release/1022/cluster.json cluster start --replace --apply`
and `doctor --live` with that profile; it reads its caches through `cache/kkref`. Remove the kept
`dsv41-karmic-kraken` containers only once the rollback is no longer wanted, and keep the
`cache/kkref` link while any recipe names it.

## Candidate 6 on the triangle (prepared 2026-10-25, CET)

The owner scoped this release to the ring; afterwards the cluster is re-cabled as the triangle
and TP3 is qualified quickly on the same image. Until it passes, the triangle serves
spark-ds41f-1022. Nothing here changes the recipe.

**Arms.**
- [rc6/rc-tp3.json](rc6/rc-tp3.json) is [rc6/rc-tp4.json](rc6/rc-tp4.json) at `tp3-triangle`:
  - the same image, vLLM tree label and lock;
  - the recipe's TP3 sizing unchanged: 1,048,576 tokens, eight streams, a 4,096-token budget
    (threshold 4,048), the 4.90625 GiB pool, the embedding and head in the display carve-out.
- [rc6/rc-tp3-dev.json](rc6/rc-tp3-dev.json) adds the dev-mode RPC. `tests/test_recipe.py` pins
  that the TP3 arm is the ring arm at the other shape.

The ring is the only shape where 1022's successors have run. Beside the TileLang completion, the
triangle meets these recipe changes for the first time:
- the JIT monitor off;
- the 21 inert settings removed;
- glibc malloc pinned (`MALLOC_ARENA_MAX=2`, a fixed 1 MiB `MALLOC_MMAP_THRESHOLD_`; #439's fix,
  [2026-10-23-next-release](../2026-10-23-next-release/README.md#memory-439s-growth-and-the-start-up-peak-memory));
- prompt-cache usage details;
- the renamed container, labels, variables and JIT caches.

**Specs.** None of them builds an image or runs the release tests: candidate-6-tp4 builds
`spark-ds41f:19f2c20ed4d6-next-6` onto every node, and its release tests belong to the image.
1. [rc6/candidate-tp3.json](rc6/candidate-tp3.json), run `candidate-6-tp3` (about 40 min): the
   measure job of [rc6/candidate-tp4.json](rc6/candidate-tp4.json) at three nodes.
   - The determinism gate, `--wide` with and without logprobs, with 8 and 4 distinct streams.
     TP3 serves eight streams, so there is no 16-stream run (release-tp3 drops it too).
   - The qualification battery: control spellings, the controls probe, the TTFT trace, memory,
     the sink counters, and the probes on dgx1-dgx3.
   - The generation-control digests with logprob ids, prompt logprobs, seeded sampling, images
     and the startup lines.
2. [rc6/prefill-1m-tp3.json](rc6/prefill-1m-tp3.json), run `candidate-6-1m-tp3` (about 12 min):
   one cold 1,000,000-token prefill under the bench's memory watch. It is r6x's TP3 1M check
   ([prefill-1m-suite.json](../2026-10-19-r6x-tp3-1m/prefill-1m-suite.json)). It is needed
   because release-tp3 prefills only up to 500,000 tokens, and at 1M the steady guard is TP3's
   binding constraint (#439).
3. [rc6/release-tp3.json](rc6/release-tp3.json), run `candidate-6-release-tp3` (about 4 h): the
   suites `release-tp3`, `single-stream` and `sampled`. The arms are candidate 6, spark-ds41f-1022's
   TP3 dev arm
   ([spark-ds41f-1022-tp3-dev.json](../2026-10-22-spark-ds41f-1022/spark-ds41f-1022-tp3-dev.json)),
   then candidate 6 again.

`probes.py` and `mem_snapshot.py` now take `--container`, and the TP3 spec names `spark-ds41f`.
Both scripts read `dsv41-karmic-kraken` by a constant. During a candidate's boot that is the
stopped production container, which the lab keeps. Without the option, the probes' rank lines
and the memory snapshot's KV excerpt describe 1022's last boot. The default is unchanged, so
earlier specs still reproduce. [rc6/candidate-tp4.json](rc6/candidate-tp4.json) passes no
`--container`, so its rank lines and KV excerpt are production's. Read its rank lines from the
candidate's log.

### Run order

Before re-cabling:
1. candidate-6-tp4 has passed: one image ID on dgx1-dgx4, 0 failed release tests, and the
   ring's gate. If the release tests fail, nothing here runs.
2. This branch is in main. The lab runs only origin/main's specs, and its `sync` job moves the
   node checkouts to origin/main, so merge it once the ring's release window has ended.

After re-cabling:
1. The fabric:
   - `scripts/cx7-config.py dgx1 dgx2 dgx3` (dgx-spark-fleet) finds the cables 1-2, 2-3 and
     3-1 by LLDP.
   - On dgx1 and dgx3, the two nodes the ring changed on 2026-10-23, install the triangle's
     saved `40-cx7.yaml` (`~/spark-lab/topology-tp3/` on dgx1) and `netplan apply`.
   - `ping -M do -s 8972` reaches both neighbours on both paths.
   - GID index 3 is RoCE v2 `::ffff:<address>` on every port; a reboot can shift it.
   - dgx4 stays out.
2. `bin/spark site shape tp3-triangle --apply`, then `bin/spark cluster sync --apply`.
3. Production at TP3 is spark-ds41f-1022:
   - If main's `config/cluster.json` is still 1022's recipe, its tp3-triangle names
     `manifests/suites/2026-10-22-spark-ds41f-1022-tp3` and the lab's default production
     profile is right.
   - If candidate 6 was promoted on the ring first, its recipe carries no TP3 reference until
     this qualification passes. Pass every window command
     `--production-config experiments/2026-10-23-next-release/1022/cluster.json` (1022 frozen).
   - Start it and run `doctor --live`.
4. On dgx1-dgx3, `docker image inspect spark-ds41f:19f2c20ed4d6-next-6 --format '{{.Id}}'` gives
   candidate-6-tp4's image ID.
5. `bin/spark workload --duration 15` shows no foreign requests. Dry-run each spec
   (`scripts/lab.py run <spec> --dry-run`: 20, 5 and 19 steps, no problems). Queue the first two,
   in order:

```bash
python3 scripts/lab.py queue add experiments/2026-10-25-spark-ds41f-candidate/rc6/candidate-tp3.json
```

```bash
python3 scripts/lab.py queue add experiments/2026-10-25-spark-ds41f-candidate/rc6/prefill-1m-tp3.json
```

6. Read candidate-6-tp3's determinism lines while the 1M check runs (about 12 min). Queue the
   comparison only if both hash pairs are identical: it would spend four hours on a build that has
   already failed. An empty queue keeps the window held, so the comparison starts once it is queued.

```bash
python3 scripts/lab.py queue add experiments/2026-10-25-spark-ds41f-candidate/rc6/release-tp3.json
```

**The first boot compiles.** The triangle has not booted any build since the JIT caches moved
to `/cache/spark-ds41f/jit`, and no build with the completion's TP3-shaped kernels. So
candidate-6-tp3's boot fills the TileLang, Triton and CUDA caches, and its boot time and
`tilelang_prepare_s` include those compiles. On #503, `-next-4` booted on a triangle in 141.7 s cold
and 105.6 s warm. The 1M check's boot and the comparison's boots are warm. Each boot's
ready time is in the lab log.

### Pass criteria

The last column is the triangle's earlier measurements: spark-ds41f-1022's gate (gate-1022-4-tp3,
2026-10-22), r6x's TP3 1M record, and the 2026-10-19 budget sweep (r6x with the CED skip).

| Step | Pass | Earlier on the triangle |
| --- | --- | --- |
| determinism, with and without log probabilities | `identical: true`, 0 token and 0 log-probability differences in all six scenarios; hashes `62313cb5` / `9742ac2c` with log probabilities, `4d143977` / `12fd8257` as text | 1022: the same hashes (the TP3 path's since 1017) |
| 8 and 4 distinct streams | a first look; the comparison judges | r6x's TP3 gate: 147.6 / 112.5 tok/s |
| control spellings | every location keeps the neutral request's control sequence | 1022: pass in all eight |
| controls probe | all four pass | (not in 1022) |
| TTFT trace | no foreign requests; p50 at four and eight streams and isolated mean a first look (the comparison's TTFT rows judge) | 1022: 265.8 / 355.2 ms; 147.7 ms |
| memory once serving | above the 3 GiB guard on every node; compared with 1022's | 1022: dgx1 6.54, dgx2 7.93, dgx3 8.14 GiB |
| sink counters, ordinary text / probes | 0 flags / `Say hi.`, `Thanks!` and `ok` flagged, nothing collapsed, the three rank lines (spark-ds41f's log) equal | 1022: 0 / the same three, 8 rows a rank |
| generation-control digests | no failure: only the by-design 400s for min-p and logit bias under speculative decoding; logprob ids succeed alone and at eight streams | not runnable on 1022 (logprob ids return 500 under DSpark) |
| prompt logprobs | succeed, one digest alone and at eight streams | not runnable on 1022 (the request crashes its engine) |
| seeded temperature 1 | `again` identical to alone; eight at once may differ (the seeded penalties variation, as at TP4) | |
| images | one digest alone and at four | |
| startup lines | 0 B12X preparation, temporary KV block, B12X progress, JIT monitor and JIT report lines; TileLang calls and init time recorded | |
| boot | ready under the unchanged 5/3 GiB guards; the KV line reports at least 4.5 full 1M contexts | 1022: 80.4 s; 4,765,898 tokens (4.55x) |
| cold 1M prefill | completes, nothing failed or halted (the bench floor is 4 GiB); each node's lowest MemAvailable recorded | r6x: 3,227 tok/s, dgx1 lowest 5.93 GiB; with the CED skip, three runs at 4,096: 3,569 tok/s, 5.26 GiB |
| release comparison | no verdict worse than spark-ds41f-1022 against the candidate's own repeat; greedy responses identical, teacher-forced KL 0 | |

The digests of controls, sampling and images measured here become the triangle's references for
the next build. No TileLang-complete build has run at TP3 before, and digests do not carry across
shapes: the determinism hashes already differ between the ring and the triangle.

After a pass, the owner decides the promotion. The candidate arm's release-tp3 run becomes
tp3-triangle's `benchmark_reference` (`manifests/suites/<baseline>-tp3`, with its single-stream
run beside it; [docs/release-benchmark.md](../../docs/release-benchmark.md#4-promotion)).

### The triangle's budget and memory: PR #503's evidence (recommendation, not applied)

Evidence: JordiPosthumus on #503 (2026-10-24/25: `-next-4` on their triangles at 1M with the same
sizing; their SM clock is capped at 2200 MHz and their site overlay differs) and #439. Our own
evidence is the 2026-10-19 sweep
([2026-10-19-prefill-budget-tp3](../2026-10-19-prefill-budget-tp3/README.md): r6x with the CED
skip, three repeats).

| 6,144 against 4,096 | Their triangle (`-next-4`) | Our triangle (2026-10-19) |
| --- | --- | --- |
| prefill, 32K-262K (ours to 500K) | +6.1 to +6.7% | +7.5 to +8.8% |
| cold 1M prefill | +6.3% (one run) | +6.4% (3,797 against 3,569 tok/s, three runs) |
| decode alone (closed loops at 8 and 1; our step without a prefill) | level | level (0.100 s at every budget) |
| seven streams' median step beside a cold 131K prefill | not measured | 1.143 against 0.857 s (+33%) |
| head node's lowest MemAvailable at 1M | 4.03 against 3.87 GiB (one run each) | 5.03 against 5.26 GiB (three runs; 8,192: 4.57) |
| ten hours of their production traffic at 6,144 | no guard kills, at least 4.33 GiB | |

Reading:
- **Prefill:** confirmed on both triangles, +6-9%.
- **Not free:** decode alone is level, but the other streams' step grows by a third while a long
  prompt prefills. Their decode figures are closed loops with no long prompt beside them. On the
  ring the owner took the same kind of trade at 16,384 (prefill +5-6%, stall step +78%).
- **Memory:** "6,144 leaves more headroom than 4,096" rests on one cold run per budget. Our three
  runs fall with the budget (5.26, 5.03, 4.57 GiB), and dgx1's start-up low already differed by
  0.76 GiB between two boots of one build (1022's promotion comparison). Read 6,144 as costing about 0.2-0.3 GiB at 1M.
- **#439, the steady guard first:** their heads reached the 3 GiB guard hours after boot. The cause
  was the high-water marks of the API server's and the engine core's glibc arenas. Candidate 6
  brings that fix to the triangle for the first time; on their triangle it held 4.89 GiB through a
  sweep and at least 4.33 GiB over ten hours.
- **Our headroom:** our triangle starts higher than theirs (1022: dgx1 6.54-6.70 GiB serving).
  The completion also frees memory: on the ring, candidate 5 left dgx1 1.8 GiB more than
  candidate 4 at the same budget (20.76 against 18.94 GiB). The dropped MTP buffer alone is 168 MB
  a rank at TP3's 4,096. How much of that the triangle gets is unmeasured: the qualification's
  memory snapshot and its 1M check measure it.

Recommendation for the owner:
- **Qualify at 4,096 as prepared.** It is like-for-like with 1022's references and the quickest
  path.
- **To revisit the budget in this release,** run [rc6/budget-tp3.json](rc6/budget-tp3.json) after
  the qualification and before the comparison (about 1.5 h). Then the comparison and the
  triangle's reference are measured at the budget that ships. The arms are 6,144
  ([rc6/budget-6144-tp3.json](rc6/budget-6144-tp3.json)), 4,096, and 6,144 again, on the sweep's
  suite: 32K-500K prefill, the 131K stall with seven streams, three cold 1M prefills, and the
  memory watch.
- **Take 6,144** (tp3-triangle's sizing only, threshold 6,096; tp4-ring unchanged) if all of
  these hold:
  - every boot passes the 5 GiB startup guard;
  - dgx1 stays at least 1.5 GiB above the 3 GiB guard through the 1M prefills;
  - the KV line still reports at least 4.5 full 1M contexts. A bigger budget charges each request
    more KV: the ring's 16,384 went from 8,580,566 to 6,512,700 tokens;
  - the owner accepts the stall, as on the ring.
- **Otherwise ship 4,096** and carry the A/B to the next release.
- **Not recommended:** a smaller pool (#439's 4 GiB; their traffic peaked at 9-16% of it) or other
  guards. The arena fix addresses the growth, the pool's 4.55 full 1M contexts is a capacity
  choice, and the guards stay 5/3 GiB.
- **Their first-sweep trap** (the first long prefill after a budget change ran far slower): the
  A/B repeats each size three times and repeats the 6,144 arm, so a slow first boot shows as a gap
  between `b6144` and `b6144-again`.
