# Candidate 5: spark-ds41f-1022 with candidate 4's changes and the TileLang completion

Owner, 2026-10-25: shift to release stabilization and freeze the scope; build one fixed,
reproducible candidate, validate it, and benchmark it against production with the
published release benchmark. Promotion needs the owner's go.

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
  (tilelang-completion README, run 26).

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
