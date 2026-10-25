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
2. **[release-tp4.json](release-tp4.json), the release comparison:** the `release` and
   `single-stream` suites (the optional `sampled` suite left out to fit the owner's window) on candidate 5, spark-ds41f-1022 and candidate 5
   again, on dev arms. The single-stream suite is the published single-stream figure
   (~74 tok/s), a different measurement from the tilelang-completion checks' c1-distinct
   (~95 tok/s: 12 prompts, 256 tokens).

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
