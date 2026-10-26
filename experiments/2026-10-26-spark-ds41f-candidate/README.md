# Candidate 7: the next spark-ds41f release, built from the experiments

The owner asked for a new candidate built from our experiments. The coordinator assembled its vLLM
series on `next-candidate` (head c579b3fe1). This folder is that candidate: its series, source
manifest, lock, arms, image, release tests and the specs that qualify it. It is modelled on
spark-ds41f-1025's [rc6/](../2026-10-25-spark-ds41f-candidate/rc6).

**Status (2026-10-26, CET):** image built on dgx4; release tests on dgx4 (below); cluster
qualification prepared, not run. The production cluster is down: it was re-cabled for the
four-node ring and the dgx4-dgx1 cable is missing.

## What it is

- **vLLM:** spark-ds41f-1025's series (candidate 6's
  [rc6/vllm/series](../2026-10-25-spark-ds41f-candidate/rc6/vllm/series), patch head 11171760, tree
  6f8e062f), then the 42 commits of `next-candidate` from e0ccc837d (1025's tree) to c579b3fe1, in
  application order ([vllm/series](vllm/series), written by [make_series.py](make_series.py)).
  - A commit the records already number keeps its number (0491-0507); the others are numbered from
    0508 in application order. So the file numbers are not monotonic in the series.
  - Replayed onto the pinned base with the build's fixed committer, the series gives patch head
    `8203a3b8`, tree `fbd941eb`: `next-candidate`'s tree ([source.json](source.json),
    [upstreams.lock.json](upstreams.lock.json)). `bin/spark build prepare` and `build check` on
    dgx4 reproduce both.
  - Every other source (B12X, NCCL, TileLang, TileKernels, sparknet, CUTLASS, the base image) is
    1025's.
- **Capabilities:** 1025's, without `ced-nonfinal-row-skip` (the prompt-end anchor replaces the
  non-final skip), plus one for each change below. `tilelang-only` is among them, so
  `bin/spark`'s image smoke drops the B12X RoCE import and the doctor refuses
  `VLLM_DS41_KERNEL_BACKEND`.

### The series after 1025, in order

| Patches | Commits | What | Record | Outputs at temperature 0 |
| --- | --- | --- | --- | --- |
| 0492 | f630c32cb | Window groups holding a DSpark drafter layer retain the drafter's 128 tokens (127 before) | #588 | unchanged |
| 0500 | 9b65ced6c | `VLLM_KV_CACHE_POISON`: new blocks filled with NaN (debug, off) | #595 | unchanged |
| 0501 | c35ac6c69 | DSpark draft windows start at CED's replay start | #595 | unchanged |
| 0502 | 84e3ce800 | No per-allocation KV zeroing (every cache reads only written slots) | #595 | unchanged |
| 0503 | 7a2125421 | The indexer reads no key past the visible positions | #595 | unchanged (bit-identical) |
| 0508 | fa4c25e88 | Barron et al.'s softmax fix (Eq. 5) in the logprob and verification kernels | #584 | text unchanged; every logprob moves by up to a few 1e-6 |
| 0491 | 615f22255 | The startup KV figure charges each request its window, the unsettled batches once | #586 | unchanged; the startup KV line changes |
| 0498-0499 | 89e657883, d5dbfac3f | The weight digest (`SPARK_DS41F_WEIGHT_DIGEST`, off); `--load-format direct` (B12X's bounce-mode O_DIRECT loader, ported) | #594 | unchanged (same weights) |
| 0493-0496 | 024d69e82..ae81a73b2 | TileLang is the only kernel family: the B12X family, `VLLM_DS41_KERNEL_BACKEND`, the ported kernels' Triton originals and B12X remnants go | #591 | unchanged |
| 0509 | 9730c7b62 | Test: the draft-window replay-start case on TileLang's chunk | #591 | (test) |
| 0497 | b6e396951 | `SPARK_DS41F_DS41_INDEXER_SPLIT_CHECK` is `SPARK_DS41F_INDEXER_SPLIT_CHECK` (unset in the recipe) | #590 | unchanged |
| 0510 | a7ba079c7 | mHC finalize writes its output's MXFP8 cast; the projections it feeds stop casting | #587 | unchanged (bits compared) |
| 0511 | 4209a1903 | SP prefill MoE: the combine writes the exchange slots, the exchange runs under it | #593 | unchanged |
| 0512 | e20c47020 | Test: the window reference bounds draft rows by the replay start | #591 | (test) |
| 0513-0519 | 412777ce1..16c1b8cae | Kernel audit: sparse-MLA page tables over live pages, one-launch metadata builders | #585 | unchanged (bit-identical) |
| 0504 | 9ad4a1fdb | 64-row routed-MoE tiles for prefill calls | #599 | unchanged (bits identical) |
| 0505 | a0dc662f5 | Two budget-sized buffers nothing reads dropped | #599 | unchanged |
| 0506 | 410338b04 | TileLang attention scratch reserved in its own workspace lane | #599 | unchanged |
| 0507 | 577091997 | The candidate source publishes block ids | #599 | unchanged (bit-identical) |
| 0520-0525 | 0ccbe30e5..97e204cd9 | PDL along the decode chain, its tests, and the late trigger of the routed grouped GEMMs from 16 rows | #589 | unchanged (bits identical) |
| 0526-0527 | 324ab8752, b56eef3d3 | CED replays the prompt's last 128 rows wherever its chunks end | #612 | **changed** for prompts whose final chunk has 1-128 rows |
| 0528-0532 | f6daa5aaf..c579b3fe1 | Block verification ends each block at the dead-row cut | #611 | unchanged; sampled realizations change |

## The image

`spark-ds41f:19f2c20ed4d6-next-7`, image `sha256:2319f8e7f5dc`, built on dgx4 at 15:43-15:51
CET with `bin/spark build image --apply` from deployment commit `2ba20f14` (453 s, 8 compile
jobs; dgx4's lowest MemAvailable 79.4 GiB). From
`~/spark-lab/agents/candidate-7/spark-ds41f` on dgx4, never the node's deployment checkout:

```bash
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/image-dgx4.json build prepare
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/image-dgx4.json build check
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/image-dgx4.json build image --apply
```

- **Labels:** vLLM patch head `8203a3b8`, tree `fbd941eb`; B12X `1a647a31`, NCCL `6af10aa7`,
  TileLang `222b895d`, TileKernels `cb2fa41c`, sparknet `f997a594` (1025's).
- **Smoke:** the GPU import smoke passed without the B12X RoCE import (`tilelang-only`), then the
  TileLang and sparknet smokes.
- **B12X is still in it:** b12x 1.3.0 at `/opt/spark-ds41f/candidate/b12x`, with its
  `b12x_loader` and `b12x_fp6` plugins registered; the DS4.1 Engram module still imports its
  preparation types. `vllm.distributed.device_communicators.b12x_roce_all_reduce` is gone (0494).
  Removing B12X is a later release.

## The recipe changes (the arms)

[rc-tp4.json](rc-tp4.json) (`tp4-ring`) and [rc-tp3.json](rc-tp3.json) (`tp3-triangle`, 8,192)
extend spark-ds41f-1025's recipe, frozen here in [1025/cluster.json](1025/cluster.json) with its
lock (verbatim, plus `upstreams_config`, as 1022's was), so they stay "1025 plus the series"
whatever `config/cluster.json` becomes. Each sets the image, its vLLM tree label and the lock, and
the three recipe changes the series needs:

- `--load-format direct` (0499), through `serve_options`;
- `VLLM_PLUGINS=""`: empty, not removed. Unset, vLLM loads every installed plugin (the image has
  four); 1025 loads only `b12x_loader`, which direct loading does not need;
- no `VLLM_DS41_KERNEL_BACKEND` (`unset`): 0494 removed the switch.

Rendered against 1025 at each shape (`bin/spark render dgx1`), only those lines and the image
differ. `config/cluster.json` is unchanged.

**Every other variable, checked against the series.** Each recipe variable is read by the same
number of vLLM files on 1025's tree (e0ccc837d) and the candidate's (c579b3fe1), except
`VLLM_DS41_KERNEL_BACKEND` (10 files, then none) and `SPARK_DS41F_DSPARK_DEAD_ROWS_TAU` (one more
reader: block verification at the cut). The names the series adds, removes or renames:

| Variable | In the series | In the recipe |
| --- | --- | --- |
| `VLLM_DS41_KERNEL_BACKEND` | removed (0494) | unset by the arms |
| `VLLM_ENABLE_ROCE_ALLREDUCE` | `=1` now refused (0494) | not set |
| `VLLM_ROCE_ALLREDUCE_MAX_SIZE`, `VLLM_ROCE_ALLGATHER_MAX_SIZE`, `VLLM_DS41_ATTENTION_COMPUTE` | removed (0494) | not set |
| `SPARK_DS41F_DS41_INDEXER_SPLIT_CHECK` | renamed `SPARK_DS41F_INDEXER_SPLIT_CHECK` (0497) | neither set |
| `SPARK_DS41F_WEIGHT_DIGEST` | new, debug, default off (0498) | not set (the loader A/B sets it) |
| `VLLM_KV_CACHE_POISON` | new, debug, default off (0500) | not set |
| `VLLM_PLUGINS` | unchanged | `b12x_loader` → empty |

No serve argument or choice is removed; 0494's chunk alignment moved from the variable into the
DS4.1 config (`SchedulerConfig.align_prefill_chunks`).

Other arms:
- [rc-tp4-dev.json](rc-tp4-dev.json), [rc-tp3-dev.json](rc-tp3-dev.json): the dev-mode RPC for the
  release suites' engine grid.
- [rc-tp4-12288.json](rc-tp4-12288.json), [rc-tp3-6144.json](rc-tp3-6144.json): other prefill
  budgets for the CED cross-shape check only.
- [rc-tp4-nopdl.json](rc-tp4-nopdl.json) (and `-dev`): the candidate without PDL, for #589's A/B
  ([make_nopdl.py](make_nopdl.py)). PDL changes 13 package files that no later commit touches;
  they are mounted at 577091997 from `nopdl/overlay`, and nothing else imports `pdl`.
- [1025/](1025): spark-ds41f-1025 as it serves at each shape (`tp4.json`, `tp3.json`), their
  dev-mode arms, and the two budget controls.
- [image-dgx4.json](image-dgx4.json), [image-tp3.json](image-tp3.json): the image job's profiles.

## Release tests

`scripts/release_tests.py check` for the series (19f2c20ed..c579b3fe1): 159 selected files, 104
gate, 55 excluded, 0 problems. Three files join the gate in `config/release-tests.json`:

- `tests/kernels/attention/test_deepseek_v41_ced_replay.py` (new, 0527): a decoder stack on the
  TileLang kernels under several chunkings, bit for bit against one chunk;
- `tests/kernels/test_deepseek_v41_tilelang_pdl.py` (new, 0521-0524): PDL's graph edges, no read
  before the wait, a layer's chain bit for bit with poisoned buffers;
- `tests/v1/worker/test_workspace.py` (changed by 0506): the attention scratch in its model's lane.

The direct loader's three test files were already gated (#594). Manifest:
[release-tests-gate-manifest.json](release-tests-gate-manifest.json); bundle:
[bundles/release-tests](bundles/release-tests), on the next-7 image.

**Run 1 (dgx4, `scripts/lab.py kernel-local`):** running.

## Qualification (prepared, not run)

Every spec dry-runs cleanly (`scripts/lab.py run SPEC --dry-run`; the TP3 ones with the site at
`tp3-triangle`). The lab runs only origin/main's specs, so this folder merges first. The ring
runs first: its image job copies dgx4's build to dgx3, dgx2 and dgx1 along the ring's existing
links (dgx4 → dgx3 → dgx2 → dgx1), so it does not need the missing dgx4-dgx1 cable. The triangle's
specs copy it from dgx2 again and check one image on dgx1-dgx3.

### On the ring (production config: `config/cluster.json`, 1025)

1. **[candidate-tp4.json](candidate-tp4.json)**, run `candidate-7-tp4`, about 45 min: the image
   copies, then rc6's qualification battery on the candidate:
   - the determinism gate (`--wide` with and without logprobs) and #593's 16.5K SP prompt
     (`--long-lines 1100`, a full 16,288-token chunk under the exchange);
   - 16, 8 and 4 distinct streams; control spellings; the controls probe; the TTFT trace;
     memory; the sink counters and probes (`--container spark-ds41f`, so the rank lines are the
     candidate's); the generation-control, prompt-logprob, seeded and image digests;
   - the CED probe ([ced_cross_shape.py](ced_cross_shape.py)) and the startup lines.
   The release tests are not repeated: they run on dgx4 on this image ID.
2. **[release-tp4.json](release-tp4.json)**, run `candidate-7-release-tp4`, about 3.5 h: suites
   `release`, `single-stream` and `sampled`; arms candidate 7, spark-ds41f-1025
   ([1025/tp4-dev.json](1025/tp4-dev.json)), candidate 7 again.
   `scripts/release_table.py candidate-7-release-tp4 spark-ds41f-1025 candidate-7 candidate-7-again`.
3. **[ced-cross-shape-tp4.json](ced-cross-shape-tp4.json)**, about 20 min: the CED probe on the
   candidate at 16,384 and 12,288, and on 1025 at both (the control).
4. **[deltas/deltas-tp4.json](deltas/deltas-tp4.json)**, about 40 min: candidate 7 and 1025 in
   one session with the gate, the 16K prompt, the seeded, controls and image digests and the
   softmax fix's `logprob_margins.py`; the softmax fix's table compares them (text equal, the size
   of every logprob difference).
5. **[loader/loader-tp4.json](loader/loader-tp4.json)**, about 25 min: #594's loader A/B on the
   candidate's image, direct / B12X's loader / direct, each with the weight digest. Pass: every
   rank's digest equal across loaders, outputs identical, load times recorded
   (`loader_report.py`).
6. **[pdl-ab-tp4.json](pdl-ab-tp4.json)**, about 1.5 h: #589's serving A/B on the candidate's
   image. Determinism, sampled and controls digests and distinct streams on no-PDL then PDL;
   then the `screen` and `single-stream` suites, PDL / no-PDL / PDL. Pass: digests identical
   between arms (PDL changes no bits); the attribution is read from the screen, c8 and c16
   especially (the 48-row question).

Steps 3-6 attribute and cross-check; promotion needs steps 1-2 on both shapes.

### On the triangle (after re-cabling; production config: `config/cluster.json` at `tp3-triangle`, 1025)

7. **[candidate-tp3.json](candidate-tp3.json)**, run `candidate-7-tp3`, about 40 min: step 1 at
   three nodes (no 16-stream run, probes on dgx1-dgx3), the image from dgx2.
8. **[release-tp3.json](release-tp3.json)**, about 4 h: suites `release-tp3`, `single-stream`,
   `sampled`; arms candidate 7, 1025 ([1025/tp3-dev.json](1025/tp3-dev.json)), candidate 7 again.
9. **[ced-cross-shape-tp3.json](ced-cross-shape-tp3.json)**, about 20 min: the CED probe on the
   candidate at 8,192 and 6,144, and on 1025 at both.

Commands, from dgx1 under the coordinator's window (each spec dry-runs first):

```bash
python3 scripts/lab.py run experiments/2026-10-26-spark-ds41f-candidate/candidate-tp4.json --dry-run
python3 scripts/lab.py queue add experiments/2026-10-26-spark-ds41f-candidate/candidate-tp4.json
```

### The CED cross-shape check

The brief's version asks for one token-exact 8,208-token prompt identical at TP4, TP3-8192 and
TP3-6144. TP4 and TP3 cannot be bit-identical whatever CED does: their row-parallel layers reduce
different partial sums, and 1025's determinism hashes already differ between the ring and the
triangle. So the check compares chunkings within one tensor-parallel size, and records TP4
against TP3 as information.

`ced_cross_shape.py` tokenizes a fixed synthetic document with the server, cuts it to exact
lengths and sends the ids through `/v1/completions` (temperature 0, 128 tokens, `ignore_eos`, the
chosen tokens' logprobs, a fresh cache salt; `cached_tokens` must be 0), twice each:

| Prompt | TP4 16,384 (T 16,288) | TP4 12,288 (T 12,192) | TP3 8,192 (T 8,144) | TP3 6,144 (T 6,096) |
| --- | --- | --- | --- | --- |
| 8,208 tokens | one chunk | one chunk | final chunk 64 (**affected**) | final 2,112 |
| 16,352 tokens | final 64 (**affected**) | final 4,160 | final 64 (**affected**) | final 4,160 |

Pass, on the candidate: each run's repeats identical, and
`ced_cross_shape.py compare` identical (prompt ids, output ids and logprobs) between the two
budgets of each size:

```bash
python3 experiments/2026-10-26-spark-ds41f-candidate/ced_cross_shape.py compare results/private/determinism/candidate-7-ced-tp4/ced-cross-shape-candidate-7.jsonl results/private/determinism/candidate-7-ced-tp4/ced-cross-shape-candidate-7-12288.jsonl
python3 experiments/2026-10-26-spark-ds41f-candidate/ced_cross_shape.py compare results/private/determinism/candidate-7-ced-tp3/ced-cross-shape-candidate-7.jsonl results/private/determinism/candidate-7-ced-tp3/ced-cross-shape-candidate-7-6144.jsonl
```

The control: on 1025 the affected lengths should differ between the two budgets (the
kernel-level chunking check found 17 of 25 affected cases differing on next-candidate). If 1025
compares identical too, the probe is not sensitive enough and proves nothing.

### Expected deltas against spark-ds41f-1025

| Check | Expected | Cause |
| --- | --- | --- |
| Determinism gate, as text (`--no-logprobs`) | identical to 1025: `c9cf9dbe` / `c6bb4f4e` at TP4, `4d143977` / `12fd8257` at TP3; `identical: true` | the gate's prompts are not CED-affected (one chunk at 16,288; final chunks of about 2.4K and 4.4K at 8,144 and 6,096) |
| Determinism gate, with logprobs | **new** digests (not `3eab5112` / `db25fde3` at TP4, `62313cb5` / `9742ac2c` at TP3); `identical: true` within the candidate | the softmax fix (0508) |
| Temperature-0 text in general | identical except CED-affected prompts: multi-chunk prompts whose length mod T (counted from a prefix hit) is 1-128, 0.79% of multi-chunk prompts at TP4 and 1.57% at TP3-8192 | 0526 |
| Release suites' prefill sizes | none affected at TP4 or TP3-8192 (mod T: 192, 768, 1,536, 3,072 / 3,216, 6,432) | |
| Greedy responses (256) in the release comparison | identical | |
| Teacher-forced KL | about 0, no longer exactly 0 | logprobs move by up to a few 1e-6 |
| Seeded temperature 1 (`sampled_digest.py`), the sampled suite's text | **changes** | block verification at the cut (0528) and the softmax thresholds; the law, acceptance and sampled throughput are unchanged (#611) |
| Prompt logprobs, controls digests, image digests | **change** (each hashes logprobs); their texts compare equal | 0508 |
| Startup KV line (`GPU KV cache size`) | **11,554,122 tokens, 11.02x** at TP4 16,384 (1025: 6,512,700, 6.21x); **5,366,846, 5.12x** at TP3 8,192 (1025: 4,009,281, 3.82x). The pool is unchanged | 0491 (after 0492: 65 tokens fewer than 0491 alone) |
| Startup | no B12X preparation, temporary KV block, B12X progress or JIT lines; the weights load through the direct loader; the TileLang call count recorded (1025: 247 at TP4, 246 at TP3), which the added and removed kernels may change | 0493-0496, 0499, the new kernels |
| Memory while serving | more MemAvailable than 1025: up to about 1.8 GiB a rank at TP4 16,384 (block-id candidates 56 KiB a token, the unread buffers 10 KiB a token, the drafter's lane 837 to 13 MiB) | 0505-0507 |
| Speed | prefill up (64-row MoE tiles: routed MoE -15% at TP4 16K, -12% at TP3 8K; the mHC cast -0.88 ms a TP4 SP layer; the exchange, about 5% at 15K); decode steps shorter (PDL +0.7-1.3% on the c1 chain; the cast -4.3 µs a decode layer at c1) | |

Promotion follows rc6's: `config/cluster.json` becomes the resolved arms, image retagged with the
release version (`spark-ds41f:19f2c20ed4d6-MMDD`), the source manifest under `manifests/sources`,
the comparison's first candidate arm as each shape's `benchmark_reference`, and 1025's frozen
recipe here as the rollback.
