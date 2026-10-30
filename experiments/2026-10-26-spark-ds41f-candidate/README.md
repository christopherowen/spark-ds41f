# Candidate 7: the next spark-ds41f release, built from the experiments

The owner asked for a new candidate built from our experiments. The coordinator assembled its vLLM
series on `next-candidate` (head c579b3fe1). This folder is that candidate: its series, source
manifest, lock, arms, image, release tests and the specs that qualify it. It is modelled on
spark-ds41f-1025's [rc6/](../2026-10-25-spark-ds41f-candidate/rc6).

**Released (2026-10-30):** candidate 14 is spark-ds41f-1030 on the ring
([below](#candidate-14-rc14), [release notes](release-notes.md)); the triangle is not qualified on it.

**Status (2026-10-26, CET):** candidate 7's image built on dgx4; its release tests passed on dgx4
(6,417 passed, 163 skipped, 0 failed). **Candidate 8** ([below](#candidate-8-rc8)) adds 0533, the
MoE output fix; its image is built and its release tests show two order-dependent test failures,
root-caused and fixed in test files (vLLM 31312f358). **Candidate 9** ([below](#candidate-9-rc9))
adds 0534-0536 (that fix, the MoE lease without its dead output buffer, the SageMaker import's
root logging); its image passes the release tests (6,585 passed, 0 failed). The cluster
qualification is prepared for candidate 9 ([rc9/](rc9)), not run. The production cluster is down: it was re-cabled for the
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

**Run 1 (dgx4, 15:56-16:41 CET): passed.** `scripts/lab.py kernel-local
experiments/2026-10-26-spark-ds41f-candidate/bundles/release-tests` from the dgx4 working copy at
2ba20f14, on image `sha256:2319f8e7f5dc`
([runs/2026-10-26-dgx4/release-tests](runs/2026-10-26-dgx4/release-tests)). **6,417 passed, 163
skipped, 0 failed**, in 104 files and 2,717 s. Every chunk exited 0 at its first run, so no chunk
was rerun file by file:

| Chunk | Files | Tests | Failed | Skipped | Seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 10 | 325 | 0 | 2 | 274 |
| 1 | 10 | 336 | 0 | 0 | 171 |
| 2 | 10 | 518 | 0 | 0 | 264 |
| 3 | 10 | 249 | 0 | 0 | 105 |
| 4 | 10 | 299 | 0 | 2 | 71 |
| 5 | 10 | 844 | 0 | 157 | 204 |
| 6 | 10 | 229 | 0 | 1 | 172 |
| 7 | 10 | 834 | 0 | 0 | 273 |
| 8 | 10 | 1,157 | 0 | 1 | 463 |
| 9 | 10 | 1,754 | 0 | 0 | 703 |
| 10 | 4 | 35 | 0 | 0 | 17 |

- **The new and changed gate files:** the CED replay test 4 passed, the PDL tests 10, the workspace
  tests 22; the direct loader's 44 + 38 GPU + 11 digest.
- **Every skip has its reason:** 148 need a Hugging Face model the offline container lacks; 10
  are ROCm-only; and one each for two GPUs, no GLP test file, no compute-sanitizer in the image,
  a draft budget below the scheduled drafts, and the indexer NaN contract's strict xfail of
  2026-10-14. These are candidate 6's skip classes (163 skips against its 160).
- Per-test JUnit reports stay on dgx4 in `~/.cache/spark-lab/compile/release-tests/candidate-7`.

No failure, so no fix branch.

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

## Candidate 8 ([rc8/](rc8))

Candidate 7 with one vLLM commit: `next-candidate` e846e64dc, "[MoE] A modular kernel's output
never lives in the workspace arena" (0533). Candidate 7's records above stay as they ran; candidate
8 supersedes it for the qualification.

- **The bug** (spark-ds41f #610, "c8 found a vLLM bug"): `FusedMoEKernelModularImpl.apply`
  returned `workspace[2]`, a view into the workspace arena, whenever the runner had prepared a
  shared workspace. At TP1 nothing copied it before the next decoder layer's mHC entry leased its
  scratch from the same bytes and overwrote it: NaN gate logits with three or more requests in a
  batch, silently wrong rows below that. 1025's tree has it too. TP3 and TP4 production are not
  exposed: every all-reduce path writes a fresh tensor.
- **The fix:** the output is always `torch.empty_like(hidden_states)`; the experts still write it
  directly (`output_alias`), so no copy is added.
- **Series:** candidate 7's plus 0533 ([rc8/vllm/series](rc8/vllm/series),
  [rc8/make_series.py](rc8/make_series.py)): patch head `52da6d93`, tree `e63ab9e9`
  (e846e64dc's). [rc8/source.json](rc8/source.json) and [rc8/upstreams.lock.json](rc8/upstreams.lock.json)
  are candidate 7's with the series, its digest, the head, the tree and the capability
  `moe-output-outside-workspace`.
- **Release-test classification:** 0533's `tests/kernels/moe/test_moe_output_outlives_workspace.py`
  joins the gate: 160 selected, 105 gate, 0 problems.
- **Image:** `spark-ds41f:19f2c20ed4d6-next-8`, image `sha256:225088ee6a8f`, built on dgx4 at
  17:29-17:37 CET from deployment commit `ecbbef97` (463 s; lowest MemAvailable 79.4 GiB). Labels:
  vLLM patch head `52da6d93`, tree `e63ab9e9`; every other tree candidate 7's. The smoke passed;
  B12X 1.3.0 and its plugins are still in it; the image's `modular_kernel.py` has the fix.
- **Arms and specs:** each [rc8/](rc8) arm extends candidate 7's with only the image, its vLLM
  tree label and the lock (each renders with only the image tag changed). The specs are candidate
  7's on those arms (run names `candidate-8-*`), and the qualification and release comparisons
  set `startup_memory` on every arm.
- **Outputs:** unchanged against candidate 7 at TP3 and TP4, so candidate 7's expected deltas
  against 1025 (above) are candidate 8's. At TP1 the fix changes outputs (no more corruption).

### Release tests (rc8/bundles/release-tests, dgx4, 17:40-18:07 CET)

**6,416 passed, 163 skipped, 2 failed**, in 105 files and 1,496 s
([rc8/runs/2026-10-26-dgx4/rc8-release-tests-1](rc8/runs/2026-10-26-dgx4/rc8-release-tests-1)).
Every chunk but 5 passed; 0533's new test passed. The two failures:

- `tests/v1/core/test_kv_cache_utils.py::test_mla_with_incompatible_swa_uses_one_full_allocation_group`
- `tests/v1/core/test_kv_cache_utils.py::test_unidentifiable_draft_with_mamba_warns`

Both assert a warning in `caplog_vllm.text`, which was empty. Neither is a non-finite output, and
both passed in candidate 7's run.

**Root cause: a library's import resets root logging for the rest of the pytest process, and the
release runner's chunking decides which tests come after it.** Five diagnostic runs on dgx4
([rc8/bundles/kvlog-next8](rc8/bundles/kvlog-next8) to `kvlog5-next8`, runs in `rc8/runs`):

1. Adding one gate file moved `tests/quantization/test_register_quantization_config.py` from
   chunk 4 into chunk 5, ahead of `test_kv_cache_utils.py` in the same pytest process. The two
   tests pass alone, after either of that file's passing tests, and fail after the whole file.
2. The warnings are emitted (a handler on `kv_cache_utils`' own logger receives them), and the
   `vllm` logger chain is intact; they never reach the capture handler.
3. The culprit is `test_custom_quant`'s `dist_init` fixture: after it, the root logger and every
   root handler, pytest's two session-long capture handlers included, are at level 40 (ERROR).
4. A trap on `setLevel` gives the stack: `dist_init` imports `tests.utils`, which imports
   `vllm.entrypoints.cli.serve`, the API server app and `vllm/entrypoints/serve/sagemaker/api_router.py`.
   That imports `model_hosting_container_standards` 0.1.16 (from the vLLM base image). Its
   `logging_config.py` runs `configure_root_logger()` on import: the root logger to
   `SAGEMAKER_CONTAINER_LOG_LEVEL` (else `LOG_LEVEL`, else ERROR), and every root handler below
   that raised to it.
5. pytest reuses its capture handlers for the whole session, so every later test in that process
   drops warnings. Which tests that hits depends only on the chunking.

**Not candidate 8's: candidate 7's image fails the same way in the same order**
([rootlog-order-next7](rc8/bundles/rootlog-order-next7): 2 failed). 0533 touches neither file.

**The fix (test files only):** vLLM branch `release-tests-root-logging`, commit **31312f358** on
e846e64dc (local, worktree `~/projects/vllm-root-logging`; `next-candidate` untouched), also as
[rc8/fix/0534-...patch](rc8/fix/0534-Tests-each-test-restores-the-root-logger-s-level-and.patch)
(not in rc8's series):
- `tests/conftest.py`: an autouse fixture restores the root logger's level and its handlers'
  levels after each test;
- `tests/test_root_logging_isolation.py`: raises the root levels (by hand, then through the real
  `configure_root_logger()`), and the test after each checks that `caplog` still captures a
  warning.

Validated on dgx4 with the two files mounted over the next-8 image ([rootlog-next8](rc8/bundles/rootlog-next8)):
the regression tests pass (4), and chunk 5's ten files in the release runner's order give 624
passed, 158 skipped, 0 failed. **The whole gate with the fix mounted**
([release-tests-rootlog](rc8/bundles/release-tests-rootlog): 105 gate files and the regression
test, 18:42-19:09 CET, [run](rc8/runs/2026-10-26-dgx4/rc8-rootlog-release-tests-1)): **6,422
passed, 163 skipped, 0 failed**, every chunk at its first run. That is run 1's 6,416 passes, its
two failures now passing, and the 4 regression tests; the skips are run 1's.

The same import puts `vllm serve`'s API server at root level ERROR in production too: third-party
warnings that propagate to the root logger are dropped (vLLM's own loggers have their own handler).
That is 1025's behaviour as well, and a separate change.

### Memory (0533)

The MoE output is now one fresh BF16 tensor of `hidden_states`' shape per layer call: 5,120 x 2
bytes, 10 KiB a row. The arena's slot for it (buffer 0 of the MoE lease, sized `M x 5120` BF16,
since `workspace13` is empty) is still leased and now unused. So the fresh tensor adds to peak
memory. At SP prefill the combine writes the exchange's slots (`defer_combine`), so there the
output is allocated and never written.

Measured with #610's step harness on dgx4 (one rank's shapes, six layers, dummy weights, TP1,
vLLM's profile run at the budget; [rc8/make_peak_bundles.py](rc8/make_peak_bundles.py),
`moe-peak2-*`), PyTorch's peak after startup:

| Shape, budget | Peak allocated, next-7 → next-8 | Peak reserved, next-7 → next-8 (two runs each) |
| --- | --- | --- |
| TP4, 16,384 | 24.111 → 24.268 GiB (+160 MiB: 16,384 rows x 10 KiB) | 24.689, 24.689 → 25.195, 25.082 (+0.39-0.51 GiB) |
| TP3, 8,192 | 25.380 → 25.458 GiB (+80 MiB: 8,192 x 10 KiB) | 25.604, 25.604 → 25.881, 26.133 (+0.28-0.53 GiB) |

- **Allocation** grows by exactly one budget-sized output. **Reserved** memory, what costs
  MemAvailable on GB10's shared memory, grows 0.3-0.5 GiB and varies between runs: allocator
  fragmentation around the budget-sized tensor. A one-rank load would not show any of it: the
  output is allocated in the forward, not while loading.
- **Against the guards:** the 5 GiB startup guard and the 3 GiB running guard see reserved
  memory. Candidate 7's memory cuts against 1025 (0505-0507: block-id candidates 56 KiB a token,
  the unread buffers 10 KiB a token, the drafter's lane 837 to 13 MiB) are larger, so candidate 8
  should still end above 1025. But dgx1's startup low varies by up to 0.76 GiB between boots, and
  the triangle's 1M prefills have the thinnest margin.
- **What the qualification must check:** every candidate-8 arm records `startup_memory`, so
  each boot's lowest MemAvailable on every node comes out beside 1025's in the release comparison.
  - Pass: every boot clears the unchanged 5 GiB startup guard.
  - Pass: dgx1's serving memory (`mem_snapshot.py`) is no lower than 1025's in the same session.
  - At TP3, the 1M prefills keep dgx1 at least 4.5 GiB, the fit-check rule.
  - Record the margin and its change from 1025. The guards are not loosened.
- **A follow-up that recovers it:** drop the dead output slot from the MoE lease when the caller
  supplies the output. That returns `M x 10 KiB` of arena, if the MoE lease is what sizes its lane.

### What it means for the candidate

0533's fix is in candidate 8's image. The release-test failure is a test-harness bug that predates
both candidates; its fix (31312f358) changes test files only, so the package files and serving
are unchanged. If the coordinator takes 31312f358 into `next-candidate`, it becomes 0534 and a
candidate 9 image carries its test files, as candidate 6 carried 0489-0490.

## Candidate 9 ([rc9/](rc9))

Candidate 8 with three vLLM commits, on branch `candidate-9-fixes` from e846e64dc (head
**e2b2b0b59**; the coordinator fast-forwards `next-candidate` to it). Candidate 8's records above
stay as they ran; candidate 9 supersedes it for the qualification.

| Patch | Commit | What | Outputs |
| --- | --- | --- | --- |
| 0534 | 31312f358 | Tests: each test restores the root logger's level and its handlers' levels (candidate 8's two order-dependent failures) | test files only |
| 0535 | 1fca420fa | MoE: a lease prepared for the caller's output holds no output buffer (0533 left buffer 0, `M x 5120` BF16, leased and unused) | unchanged (bit for bit, below) |
| 0536 | e2b2b0b59 | Entrypoints: the SageMaker routes' import puts the root logging back, unless `SAGEMAKER_CONTAINER_LOG_LEVEL` or `LOG_LEVEL` is set | unchanged; `vllm serve` logs third-party warnings again |

- **0535 in detail:** `_allocate_buffers` takes the caller's output shape. It leases no output
  buffer when that is the experts' output and they write a matching alias (not on ROCm without
  aiter, which copies from its own buffer). `prepare_workspace` passes `hidden_states`' shape,
  and the per-call lease passes the alias's. A lease without an output buffer refuses a call
  without a matching output. The FlashInfer NVFP4 workspace tests still asserted that the output
  was `workspace[2]`, which 0533 had already made untrue; they now assert the new contract.
- **0536 in detail:** `vllm/entrypoints/serve/sagemaker/standards.py` imports
  `model_hosting_container_standards.sagemaker` for both routers, then puts the root logger's
  level and its handlers' levels back. If either variable is set, the library's documented
  behavior applies. The routes are unchanged, and no switch is added.
- **Series:** candidate 8's plus 0534-0536 ([rc9/vllm/series](rc9/vllm/series),
  [rc9/make_series.py](rc9/make_series.py)): patch head `52ebc12e`, tree `2b0931a0` (the
  branch's). [rc9/source.json](rc9/source.json) adds the capabilities `moe-lease-without-output-buffer`
  and `sagemaker-import-keeps-root-logging`.
- **Classification:** `tests/test_root_logging_isolation.py`,
  `tests/entrypoints/serve/test_sagemaker_root_logging.py` and `tests/kernels/moe/test_flashinfer_moe.py`
  join the gate: 163 selected, 108 gate, 0 problems.
- **Before the build** ([rc9/bundles/precheck-next8](rc9/bundles/precheck-next8), `precheck2`):
  the branch's eleven files mounted over next-8 pass the new and changed tests (22) and the DS4.1
  MoE and model tests on the changed path (350). 0536's test first lived in
  `tests/entrypoints/serve/sagemaker/`, whose conftest needs `pytest_asyncio`, which the image
  doesn't install; it moved up a directory.
- **Image:** `spark-ds41f:19f2c20ed4d6-next-9`, image `sha256:22bfd440fb18`, built on dgx4 at
  19:52-20:00 CET from deployment commit `466e678b` (458 s; the build's lowest MemAvailable was
  52.5 GiB because peak runs shared the node). Labels: vLLM patch head `52ebc12e`, tree
  `2b0931a0`; every other tree candidate 8's. The smoke passed, B12X 1.3.0 is still in it, and
  all three changes are in the image.
- **Arms and specs:** each [rc9/](rc9) arm extends candidate 8's with only the image, its tree
  label and the lock (each renders with only the image tag changed). The specs are candidate 8's
  on them, with `startup_memory` on every qualification and release-comparison arm.

### Release tests (rc9/bundles/release-tests, dgx4, 20:03-20:29 CET)

**6,585 passed, 163 skipped, 0 failed** in 108 files and 1,590 s, every chunk at its first run
([rc9/runs/2026-10-26-dgx4/rc9-release-tests-1](rc9/runs/2026-10-26-dgx4/rc9-release-tests-1)).
- The new and changed files: `test_flashinfer_moe.py` 158 passed, `test_moe_output_outlives_workspace.py`
  4, `test_sagemaker_root_logging.py` 2, `test_root_logging_isolation.py` 4.
- `test_kv_cache_utils.py` 149 passed and 40 skipped, candidate 8's two failures among the passes.
- The skips are candidate 8's 163, each with its reason.

### 0535: memory and bits (run 3 of the step-harness measurement)

The same method as candidate 8's runs 1-2: #610's harness at 63fca56c, six layers at one rank's
shapes, dummy weights, TP1, max_model_len 32,768, each step in a fresh process
([rc9/make_peak_bundles.py](rc9/make_peak_bundles.py), `moe-peak3-*`, runs in `rc9/runs`).
Peaks in GiB, two repeats each:

| Shape, budget | next-7 allocated / reserved | next-8 allocated / reserved | next-9 allocated / reserved |
| --- | --- | --- | --- |
| TP4, 16,384 | 24.111 / 24.689, 24.689 | 24.268 / 25.195, 25.195 | **24.111** / 25.039, 24.926 |
| TP3, 8,192 | 25.380 / 25.604, 25.604 | 25.458 / 26.133, 26.133 | **25.380** / 26.055, 25.861 |

Run 3 repeats runs 1-2 for next-7 and next-8 to the MiB (next-8's run 2 also saw 25.082 and 25.881
reserved).
- **Peak allocated: recovered completely.** next-9's equals next-7's at both shapes. The fresh
  output now replaces the arena's slot instead of coming on top of it.
- **Peak reserved: recovered only in part.** 0535 removes the slot from reserved memory too: in
  run 3, next-9 is 0.16-0.27 GiB below next-8 at TP4 and 0.08-0.27 GiB at TP3 (the slot is
  160 MiB at TP4, 80 at TP3). But it is still
  0.24-0.35 GiB above next-7 at TP4 and 0.26-0.45 GiB at TP3, and it varies between repeats where
  next-7's never does. That is allocator fragmentation from allocating a budget-sized output
  each layer call: the slot sat inside one arena allocation, while the fresh tensor is allocated
  and freed beside other work. That it varies with timing points at reuse delayed across streams
  (the shared expert's combine runs on the aux stream). This is a likely explanation, not a
  measured one.
- **Outputs: bit for bit.** The filled runs' token ids and top-5 log probabilities
  ([rc9/compare_results.py](rc9/compare_results.py)):
  - next-9 equals next-8 for one request with the graphs, three requests eager, and the 16K
    budget request;
  - next-7 differs from both at one request, which is 0533's bug ("silently wrong rows below
    three requests");
  - next-7 still asserts on non-finite gate logits at three requests.
- **What recovering the rest would take:** the output allocated once rather than per call. One
  way is a DS4.1-only output buffer per workspace lane, outside the arena's leases, reused by
  every MoE layer. It is safe only while each layer's output is consumed (by the next mHC entry
  at TP1, by the all-reduce at TP3 and TP4) before the next MoE call writes it. That is the class
  of aliasing 0533 fixed, so it is a design change for the coordinator to decide, not part of
  this candidate.
- **For the qualification:** unchanged from candidate 8. Every candidate-9 boot records
  `startup_memory` against the 5 GiB guard beside 1025, with dgx1's serving memory no lower than
  1025's. Candidate 7's memory cuts (up to about 1.8 GiB a rank at TP4) are larger than this
  residual.

### Outputs against 1025

Candidate 7's expected deltas (above) apply unchanged. 0534 changes test files only, 0535 no
bits, and 0536 only what `vllm serve` logs: warnings from libraries that log through the root
logger appear again.

## Candidate 10 ([rc10/](rc10))

Candidate 9 plus the one change the one-at-a-time experiment
([../2026-10-27-one-at-a-time](../2026-10-27-one-at-a-time/README.md)) measured as a net gain
that candidate 9 did not already hold.

- **0537:** the MXFP8 GEMMs load the 32x32 block-scale grid at `SFB[row // 32]` instead of a
  per-row broadcast (vLLM mxfp8-block-scales, 58938a074). On candidate 9: block-32 GEMM digests
  equal over 182 cases, serving digests equal, decode grid 1.3% faster, 131K prefill 1.1% faster.
- **Series:** candidate 9's plus 0537 ([rc10/vllm/series](rc10/vllm/series),
  [rc10/make_series.py](rc10/make_series.py)): patch head `ebfa73d3`, tree `24313745`.
  [rc10/source.json](rc10/source.json) adds the capability `tilelang-block-scale-grid`.
- **Held back:** 2-bit expert scale codes (bit-exact, prefill 2.7-2.8% and c8 1.6% slower).
  Being reworked, each alone on candidate 10: 0528 (decode 1.2%), 0526 (131K prefill 0.5%),
  0506 (131K prefill 0.4%) and the expert codes.
- **Qualification** ([rc10/candidate-tp4.json](rc10/candidate-tp4.json)): the image built on
  dgx4 and copied to the ring; candidate 9's checks on candidate 9 and candidate 10 (digests must
  be equal); the lean suite on candidate 9, candidate 10, candidate 9 again.
- **Result (run `candidate-10-tp4`, 2026-10-28 17:13-19:05 UTC): qualified.**
  - Bits: every digest equals candidate 9's (determinism with and without logprobs, 16K,
    seeded sampled, prompt logprobs, images, CED cross-shape, control spellings and controls);
    model behaviour 64/64 greedy responses identical, teacher-forced KL 0 over 16,384
    positions. One controls digest differs, `penalties-sampled/c8`: candidate 10's
    (`fb36e3be`) is candidate 9's in its own qualification and both PDL arms; candidate 9 gave
    `9df5dee2` on this boot. The scenario does not repeat across boots (candidate 5 gave two
    values as well), so it is not a bit change.
  - Speed against candidate 9 (paired verdict; repeat candidate 9 to candidate 9 again):

    | Metric | Candidate 10 | Change | ±95% | Repeat |
    | --- | ---: | ---: | ---: | ---: |
    | engine grid, decode geometric mean (20 shapes) | | 1.36% faster | 0.16% | 0.10% |
    | c1 decode tok/s, paired | 77.50 | +0.90% | 0.08% | |
    | c8 decode tok/s, paired | 24.32 | +0.25% | 0.03% | +0.31% |
    | c1 TTFT, paired | 0.122 s | level | 0.37% | |
    | prefill 32K / 131K tok/s | 7,344 / 7,449 | level / +0.16% | 1.04 / 0.12% | |

    The grid gain is the block-scale reads' own (1.3% alone on candidate 9). c16, c8 and c4
    distinct streams, TTFT traces, startup and sink counters are level.

## Candidate 11 ([rc11/](rc11))

The release candidate the owner asked for on 2026-10-28: candidate 10, GLP dual serving
(`glp-3`), and the correctness and performance work that meets the release bar (temperature-0
bits exact; a correctness fix, or a measured gain). Anything that fails its check is held back
for rework, or dropped, and the series is rewritten without it before the qualification runs.

- **Series:** candidate 10's plus 0538-0570 (vLLM `candidate-11`, head 977e18a37;
  [rc11/vllm/series](rc11/vllm/series), [rc11/make_series.py](rc11/make_series.py)): patch head
  `ad7a3923`, tree `04de041e` (reproduced locally with the build's `git am`; `git diff --check`
  clean). Each part is its branch's commits, cherry-picked:

  | Patches | Part | Why it is in | Its check |
  | --- | --- | --- | --- |
  | 0538 | coded-arrive: the grouped MoE GEMM's stage arrivals wait for each thread's copies | correctness: the coded-scale expand could read rows not yet landed | `c10-coded-arrive`: same bits, same speed |
  | 0539-0540 | DeepSeek API effort aliases and prompt semantics (response_format, tool_choice none, the assistant prefix) | correctness of the API | `c10-api-semantics` (queued); oracle passed |
  | 0541 | GLP steering per request under a second served name (`glp-3`'s 0538) | the owner's request; stock rows bit-identical | `glp-3` qualified and released on 1025 (decode +0.27%, level) |
  | 0542 | GLP's launch joins the decode chain's PDL (new) | performance: outside the chain, 39 launches a step broke candidate 9's PDL overlap | this qualification's lean bracket; PDL source check |
  | 0543-0544 | the attention front in one launch | performance: +0.72% c1 in the harness, bit-identical | `c10-front` (queued) |
  | 0545 | prefill yield (`--prefill-yield short`, 10 s) | TTFT behind a long prefill | `c10-yield` (queued) |
  | 0546-0550 | copy drafts beside DSpark, **off** (`copy_drafts` unset) | built for the copy A/B; without it every size, kernel and graph is the release's | `c10-copy` (queued); enabled only if it gains |
  | 0551-0567 | IKET debug mode (`VLLM_DS41_IKET`), off | optional debugging: off, it registers four variables and imports nothing | its test: off adds neither IKET module |
  | 0568-0569 | expert codes v2: 2-bit scale codes for experts whose block scales span at most 4 values, each thread expanding its own row; the two v1 source tests brought up to v2 | performance: c8 paired decode +1.29%, c1 +0.43%, same bits | `c10-expert-codes-v2` |
  | 0570 | copy drafts: warmup steps run without adaptive verification | correctness: with `copy_drafts` set, every boot failed in warmup (an assert on the manager warmup clears); off, the line is not reached | `c10-copy` (requeued) |

  Conflicts: GLP's imports in `nvidia/model.py` and `model_state.py` (candidate 10 had retired
  the B12X attention import GLP's context named); copy drafts beside prefill yield in
  `scheduler.py` (two new methods at one place; the yield deferral runs before the copy
  widening, which skips prefill chunks); IKET's variables beside GLP's in `envs.py`.
- **Arm** ([rc11/rc-tp4.json](rc11/rc-tp4.json)): candidate 10's with the `glp` block of
  `../2026-10-28-glp-dual-serving/cluster.json` (the second served name
  `deepseek-v4.1-flash-weightless`, the published file mounted) and prefill yield.
- **Release tests:** `scripts/release_tests.py check` for 19f2c20ed..977e18a37: 171 selected,
  116 gate (eight new gate files in `config/release-tests.json`: block-scale words, front, IKET,
  weightless, copy drafts and its scheduler, prefill yield, expert scale codes), 55 excluded, 0
  problems.
  Manifest [rc11/release-tests-gate-manifest.json](rc11/release-tests-gate-manifest.json),
  bundle [rc11/bundles/release-tests](rc11/bundles/release-tests).
- **Held back:** X7 (a kernel experiment), the Markov L2 window (a probe; at most +0.5%); both
  now run after this qualification. The cost check found no real cost in 0506, 0526 or 0528
  (`../2026-10-27-one-at-a-time/README.md`); they stay unchanged.
- **Qualification** ([rc11/candidate-tp4.json](rc11/candidate-tp4.json)): the image built on
  dgx4 and copied to the ring; the release tests on dgx4; candidate 10's checks on candidate 10
  and candidate 11 (stock digests must be equal: GLP leaves unmarked rows bit-identical, prefill
  yield moves no chunk boundary); `dual_probe.py --speed` and AdvBench on candidate 11; the lean
  suite with the 131K stall probe (`../2026-10-28-candidate-10-levers/yield-suite.json`) on
  candidate 10, candidate 11, candidate 10 again. It runs after the queued A/Bs.
- **Result** (`candidate-11-tp4`, 2026-10-29): the release tests failed, 42 of 6,887 (dgx4).
  Forty were the copy patches': fakes without the attributes copy drafts added to the batch,
  the spec config and the rejection sampler (`num_draft_tokens_per_req` in the rejection
  sampler, `num_verification_tokens`, `num_draft_steps`, `max_decode_rows`). Two were IKET's:
  its scripts, run as files, imported the image's installed vLLM, which has no DS4.1 package.
  Every serving check passed: digests equal candidate 10's (greedy 64/64, teacher-forced KL 0),
  decode grid 1.91% faster, c8 paired decode 1.17%, c1 0.95%, prefill level, the 131K prefill
  under decode 9% slower (yield's trade). Copy drafts, enabled (`c10-copy-2`), still failed to
  start: the sparse-MLA kernel's packed-index check (`SwaIndices` 187 rows, 214 expected) in
  warmup. Not released; candidate 12 replaces it.

## Candidate 12 ([rc12/](rc12))

Candidate 11 without copy drafts, which fail to start when enabled; a release ships no feature
that does not work, and none switched off but IKET's debug mode. Copy drafts stay on their
branch (#619) until they start and pass their A/B.

- **Series:** candidate 10's plus 0538-0565 (vLLM `candidate-12`, head d984b2e66;
  [rc12/vllm/series](rc12/vllm/series), [rc12/make_series.py](rc12/make_series.py)): patch head
  `491ba2b7`, tree `4976b446` (the build's `git am` reproduces it). Candidate 11's patches in
  order, cherry-picked without a conflict, less 0546-0550 and 0570 (copy drafts), plus:

  | Patch | Change | Why |
  | --- | --- | --- |
  | 0565 | IKET mode tests: the subprocesses import the vLLM under test (the tested source root first on `PYTHONPATH`) | the release gate's two IKET failures |

- **Arm** ([rc12/rc-tp4.json](rc12/rc-tp4.json)): candidate 11's on candidate 12's image
  (`spark-ds41f:19f2c20ed4d6-next-12`); [rc12/source.json](rc12/source.json) drops the
  `copy-drafts` capability.
- **Release tests:** `config/release-tests.json` drops the two copy test files;
  `release_tests.py check` for 19f2c20ed..d984b2e66: 169 selected, 114 gate, 55 excluded, 0
  problems ([rc12/release-tests-gate-manifest.json](rc12/release-tests-gate-manifest.json),
  [rc12/bundles/release-tests](rc12/bundles/release-tests)).
- **Qualification** ([rc12/candidate-tp4.json](rc12/candidate-tp4.json)): candidate 11's, for
  candidate 12: the image, the release tests, candidate 10's checks on candidate 10 and
  candidate 12, GLP's probes, and the lean suite with the stall probe on candidate 10,
  candidate 12, candidate 10 again.
- **Result (run `candidate-12-tp4`, 2026-10-29): qualified on the ring.**
  - Release tests (dgx4, image `sha256:2d9c2cc6`): 6,686 passed, 163 skipped, 0 failed.
  - Bits: every digest equals candidate 10's (determinism with and without logprobs, 16K,
    greedy, seeded sampled, prompt logprobs, images, CED cross-shape, control spellings and
    controls); model behaviour 64/64 greedy responses identical, teacher-forced KL 0 over
    16,384 positions, acceptance 2.263 on both. One controls digest differs again,
    `penalties-sampled/c8`: candidate 12 gave `fb36e3be` (candidate 10's in its own
    qualification) and candidate 10 gave `44f3b309` on this boot; the scenario does not repeat
    across boots, so it is not a bit change. Seeded sampling at c8 is batch-variant on both
    builds, as before.
  - GLP: the dual-serving probe passes every check (both names served, steering moves all 8
    prompts, mixed batches and the prefix cache keep each name's text, completions and
    tokenize; beam search refused with 400). AdvBench (520 prompts, 256 tokens): replies with
    no refusal phrase 2.5% stock, 51.2% steered; judged compliant 1.4% / 38.9%.
  - Speed against candidate 10 (paired verdict; repeat candidate 10 to candidate 10 again):

    | Metric | Candidate 12 | Change | ±95% | Repeat |
    | --- | ---: | ---: | ---: | ---: |
    | engine grid, decode geometric mean (20 shapes) | | 0.25-1.14% faster | 0.15% | -0.88% |
    | c1 decode tok/s, paired | 79.14 | +0.69-0.91% | 0.08% | +0.21% |
    | c8 decode tok/s, paired | 24.87 | +0.80-0.86% | 0.50% | +0.03% |
    | c1 TTFT, paired | 0.124 s | level | 0.45% | |
    | prefill 32K tok/s | 7,317 | level | 1.24% | |
    | prefill 131K tok/s, alone | 7,383 | -0.8% | 0.48% | |
    | 131K prefill beside 11 decoding streams and five 500-token requests | 6,270 tok/s | -9.0% | 0.54% | +0.09% |
    | those short requests' TTFT p50 / max | 3.55 / 4.97 s | 3.1x / 3.4x sooner | 2.4% | |

    Ranges are against each candidate 10 boot; the second candidate 10 boot ran its grid
    0.88% faster than the first. The 131K prefill beside short requests is yield (0545) doing
    what it is for: each short request takes one step for its prefill and one for its held
    first token, about 10 steps of the long prompt's 19 s, and with nothing short waiting it
    schedules as before (level at 15 of 16 streams in `c10-yield-depth`). The lone 131K
    prefill's 0.8% has nothing to yield to; it is in 0541-0544 or the IKET patches, not yet
    bisected. c16, c8 and c4 distinct streams, the TTFT trace, startup and sink counters are
    level.
  - Memory, found after the result above: model loading took 72.72 GiB a rank on candidate 12
    against 71.67 GiB on candidate 10 (+1.05 GiB; idle free memory 0.6-0.7 GiB lower a
    node), and at TP3 (#668, 983K context) the host's low point fell from 5.24-5.35 GiB to
    3.45-3.92 GiB. 0563 gave the 2-bit scale codes their own buffer, half the 4-bit codes'
    size, for every expert, and kept the 4-bit codes for every expert too. The ring's
    lowest free memory in a boot is still 19.4 GiB, so candidate 12 is promoted there (owner, 2026-10-29); the triangle's margin
    at 983K needs candidate 13.
- **Release comparison** ([rc12/release-tp4.json](rc12/release-tp4.json), run
  `candidate-12-release-tp4`, about 3.5 h; [rc12/release-tp3.json](rc12/release-tp3.json) after
  the triangle is re-cabled): suites `release`, `single-stream` and `sampled`, arms candidate 12,
  spark-ds41f-1025, candidate 12 again, `startup_memory` on every boot. With the TP3
  qualification ([rc12/candidate-tp3.json](rc12/candidate-tp3.json)) these are what promotion
  needs.

## Candidate 13 ([rc13/](rc13))

Candidate 12 with the 2-bit expert scale codes in the 4-bit codes' buffer, for candidate 10's
memory with candidate 12's decode: the triangle's candidate (owner, 2026-10-29), candidate 12
staying on the ring.

- **Series:** candidate 12's plus 0566 (vLLM `scale-codes-one-form`, head 13793229f;
  [rc13/vllm/series](rc13/vllm/series), [rc13/make_series.py](rc13/make_series.py)): patch
  head `c1965d8e`, tree `e1a74af7` (the build's `git am` reproduces it).

  | Patch | Change | Why |
  | --- | --- | --- |
  | 0566 | A narrow expert's 2-bit codes go in the first `block_N` bytes of each of its rows of the 4-bit codes' buffer (zeros after); the grouped GEMM reads them through a uint8 view of that storage, `[rows, 2 * block_N]`. A wide expert keeps its 4-bit rows | 0563 kept a second buffer, half the 4-bit codes' size, for every expert: about 1 GiB a rank at TP4 |

  Each narrow load is still one contiguous `block_N`-byte row, so the bytes read and the
  expanded scale words are unchanged; the CPU tests check that both views share one storage of
  the 4-bit codes' size and that each expert's rows hold one form.
- **Arms:** candidate 12's on candidate 13's image (`spark-ds41f:19f2c20ed4d6-next-13`) and
  lock, at the triangle ([rc13/rc-tp3.json](rc13/rc-tp3.json), `-dev`, `-6144`) and the ring
  ([rc13/rc-tp4.json](rc13/rc-tp4.json), `-dev`); [rc13/source.json](rc13/source.json) adds the
  `tilelang-expert-scale-codes-one-buffer` capability.
- **Release tests:** `release_tests.py check` for 19f2c20ed..13793229f: 169 selected, 114 gate,
  55 excluded, 0 problems ([rc13/release-tests-gate-manifest.json](rc13/release-tests-gate-manifest.json),
  [rc13/bundles/release-tests](rc13/bundles/release-tests)); the GPU bit-match of mixed narrow
  and wide experts against the uncoded GEMM (`test_deepseek_v41_tilelang_moe.py`) is a gate
  file.
- **Qualification:** candidate 12's, for candidate 13 against candidate 12 —
  [rc13/candidate-tp3.json](rc13/candidate-tp3.json) and
  [rc13/release-tp3.json](rc13/release-tp3.json) on the triangle, and
  [rc13/candidate-tp4.json](rc13/candidate-tp4.json) on the ring (the image, the release tests,
  the checks and memory snapshot on both builds, GLP's probes, the lean suite on candidate 12,
  candidate 13, candidate 12 again). It passes with every digest equal to candidate 12's, speed
  level, and model loading about 1 GiB a rank below candidate 12's at TP4 (candidate 10's
  71.67 GiB) and about 1.4 GiB at TP3.
- **On the ring: a fix check, not a requalification** (owner, 2026-10-29: ship 0566 at TP4 if it
  works, the full qualification at TP3). [rc13/fix-tp4.json](rc13/fix-tp4.json), run
  `candidate-13-fix-tp4`, about 45 min: the image; the two scale-code test files on dgx4
  ([rc13/bundles/scale-codes](rc13/bundles/scale-codes), the GPU bit-match of mixed narrow and wide
  experts against the uncoded GEMM); one candidate 13 boot with the determinism gate (with and
  without logprobs), the 16K SP prompt, the memory snapshot and the startup lines. It works if
  every digest equals candidate 12's in `candidate-12-tp4` (the same checks on its own boot) and
  model loading is about 1 GiB a rank lower (candidate 12: 72.72 GiB). Candidate 12's release comparison
  against 1025 (`candidate-12-release-tp4`) then stands for candidate 13 on the ring.
- **Fix check result (2026-10-29, 21:01-21:30 UTC, with the repeat boot
  `candidate-13-fix-tp4-again`):** it works. The scale-code bundle passed on dgx4 (117 tests).
  The logprob, greedy and 16K digests equal candidate 12's (`fe2461d8`, `c9cf9dbe`,
  `fcd34fd2`). Model loading took 71.67 GiB on both boots (candidate 12: 72.72), with the KV
  cache unchanged at 11,554,122 tokens, so the gigabyte is headroom (dgx1 had 21.3 GiB
  available at ready). The startup lines are normal.

## Candidate 14 ([rc14/](rc14))

Candidate 13 with the Engram host hash fixed for a prefilling request that carries drafts, the
engine stop of [#677](https://github.com/christopherowen/spark-ds41f/issues/677): the 1030
candidate (owner, 2026-10-29: fold the fix in before promoting).

- **Series:** candidate 13's plus 0567 (vLLM `engram-drafted-prefill`, head d5df88d83;
  [rc14/vllm/series](rc14/vllm/series), [rc14/make_series.py](rc14/make_series.py)): patch
  head `b6de07aa`, tree `ea1281f8` (the build's `git am` reproduces it).

  | Patch | Change | Why |
  | --- | --- | --- |
  | 0567 | `_early_engram_ids` hashes on the host only the prefill rows of requests without drafts and leaves a drafted request's rows `UNKNOWN` for the GPU; drafts out of order leave the whole step to the GPU instead of raising | The scheduler pads a prompt with one token left (cached up to its last token) to the verification width with placeholder drafts, so a request can be prefilling and verifying at once; the old check raised on it and stopped every worker |

  The worker sorts verifying requests first and adaptive verification keeps their total row
  count, so a request after them starts at the host's row and its early IDs stay exact. The
  host IDs are a prefetch of the GPU's hash, so the output does not change. CPU tests:
  `tests/kernels/test_deepseek_v41_engram_host_placement.py` (5).
- **Arms:** candidate 13's on candidate 14's image (`spark-ds41f:19f2c20ed4d6-next-14`) and
  lock ([rc14/rc-tp4.json](rc14/rc-tp4.json), `-dev`); [rc14/source.json](rc14/source.json)
  adds the `engram-host-hash-drafted-prefill` capability.
- **Fix check:** [rc14/fix-tp4.json](rc14/fix-tp4.json), run `candidate-14-fix-tp4`, about an
  hour. It builds the image, then runs [rc14/mixed_load.py](rc14/mixed_load.py) for 5 minutes
  on each boot: 4 agents growing from 50K words by 20K a turn, and 8 chat streams that each
  ask every question twice (the second cached up to its last token). The boots are:
  - candidate 13, where the load should stop the engine;
  - candidate 14, with the determinism gate, the 16K prompt, the memory snapshot and the
    startup lines;
  - candidate 14 again, with the memory snapshot and the startup lines.

  Candidate 14 works if the load runs clean on both of its boots, every digest equals
  candidate 13's, and model loading stays at 71.67 GiB. Candidate 12's release comparison
  against 1025 then stands for candidate 14 on the ring.
- **Result (run `candidate-14-fix-tp4`, 2026-10-29): works.** The image
  (`sha256:7918fd6145b5`) built on dgx4 at 22:19 UTC from deployment commit `d7feb2d9` and was
  copied dgx4 to dgx3, dgx2 and dgx1; the same ID on every node.
  - The load ran clean on candidate 13 too (150 requests, 0 failed): the trigger is not
    reproduced on the ring, and the fix stands on the scheduler's padding path, which reaches the
    check whenever a prompt with one token left is padded with placeholder drafts (#677, asked of
    its reporter).
  - Candidate 14: 150 requests on each boot, 0 failed, prompts up to 311K tokens; every digest
    equals candidate 13's; model loading 71.67 GiB a rank on both boots; 277 TileLang calls
    prepared, init 28-37 s.
- **Promoted as spark-ds41f-1030** on the ring (owner, 2026-10-29: "Ship 1030 from candidate
  14"): image `spark-ds41f:1030` (this build, retagged), recipe and lock frozen in
  [releases/spark-ds41f-1030](../../releases/spark-ds41f-1030/), baseline
  [2026-10-30-spark-ds41f-1030](../../manifests/baselines/2026-10-30-spark-ds41f-1030.json),
  [release notes](release-notes.md).

## spark-ds41f-1030 on the triangle

The owner re-cabled to the dgx1-dgx2-dgx3 triangle on 2026-10-30 for the full TP3 qualification
of the release (dgx1 and dgx3 back on their `40-cx7.yaml.triangle` addressing; every cable path
passes a 9000-byte ping and each RoCE GID 3 holds its address). The arm is the frozen release
at the triangle, [rc14/rc-tp3.json](rc14/rc-tp3.json): the 6,144-token budget the recipe has
carried since 2026-10-27, which this release qualifies. The image is already on every node.

- **Checks** ([rc14/candidate-tp3.json](rc14/candidate-tp3.json), run `candidate-14-tp3`, about
  50 min): GLP's dual-serving probe and AdvBench on the release; then candidate 13's checks
  (determinism with and without logprobs, 16K, c8 and c4 distinct streams, control spellings
  and controls, the TTFT trace, memory, sink counters and probes, prompt logprobs, seeded
  sampling, images, CED cross-shape, startup lines) on three boots: spark-ds41f-1025 at its
  8,192 budget, candidate 14 at 8,192 ([rc14/rc-tp3-8192.json](rc14/rc-tp3-8192.json); one budget
  with 1025, so digests compare build to build) and the release at 6,144.
- **Release comparison** ([rc14/release-tp3.json](rc14/release-tp3.json), run
  `candidate-14-release-tp3`, about 4 h): suites `release-tp3`, `single-stream` and `sampled` on
  the release, spark-ds41f-1025, the release again. 1025's TP3 record has no request tables, so
  1025 is measured again.

It qualifies if the 8,192 digests equal 1025's apart from CED-affected prompts (0526), the
release's checks pass, the host's lowest MemAvailable stays above the 5 GiB startup guard, and
no verdict is worse than 1025's.
