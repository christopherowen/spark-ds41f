# Native full-depth graphs with wider verification

Base deployment: `2c4b23af`. One runtime variable: retain the native DSpark
full-depth graph shapes when copy drafts widen the verification limit.
Source review: https://github.com/christopherowen/vllm/pull/2 (draft).
No promotion, serving change, or measured speed claim.

## Source identity

Candidate-40 is the measured base, from its tracked lock and patch series:
upstream `19f2c20ed4d64ae22bd009ceacadd85e2a81e354`, 574 patches, head
`5f926df6d3126266a9c88c82c06544060008c33c`, tree
`ca1496c6481dc5bc4b4234b743e4ca023a6c7597`.
The experiment adds the shape fix and its warmup-fixture update, for 576 patches,
head `8dea43f62a149a3aef2be4b24fa1c68e9b33407d`, tree
`3f9b475dfcd86aabbc042299131cd38aa3612be0`. `vllm/series`, `source.json`, and
`upstreams.lock.json` are the reproducible inputs. The proposed immutable image
tag is `spark-ds41f:verification-shapes-8dea43f62a14`; it has not been built.

## Finding and change

Candidate-36 already captures every odd/heterogeneous total for one or two
requests and full-depth batches at each larger request count. Candidate-38
already gives context graphs the target's row ladder. The vocabulary head has
16/32/64/128-row tiles; wide dense and shared-expert entries already extend to
128. Routed MoE uses route density, not a 16- or 64-token cutoff: at 384 experts
and top-6 its current boundaries are 1,024 and 2,048 tokens. There is no evidence
here for simply raising verification capacity or changing those tile thresholds.

Candidate-40's copy defaults widen the maximum query from 6 to 17 rows. Its
shared shape generator consequently substitutes multiples of 17 for the native
multiples of 6. A three-request native full-depth step dispatches 18 target rows
as 20; six requests dispatch 36 as 40; seven dispatch 42 as 48. The context has
some additional exact counts from its low-concurrency set, but also pads 36 to
40 and 42 to 48.

The patch retains both sets, bounded by the existing capture limit, and supplies
the same set to target capture, context capture, and pre-KV weight preparation.
It does not change draft selection, model arithmetic, weights, norm kernels,
expert layout, the capture-row limit, or buffer-arena capacity. The native widths
now dispatch with both exact row and request counts. One-/two-request odd totals,
wider-copy shapes, larger fallback graphs, and prefill exclusion remain covered.

The current copy envelope is 59 rows for TP3 and 107 for TP4, not every request
at the maximum copy width. The older non-copy envelopes remain 48 and 96.

| Configuration | Target graphs, before → after | Context graphs, before → after |
| --- | ---: | ---: |
| 8 requests, width 6 | 34 → 34 | 23 → 23 |
| 16 requests, width 6 | 48 → 48 | 35 → 35 |
| 8 requests, width 17 | 65 → 70 | 39 → 41 |
| 16 requests, width 17 | 74 → 87 | 48 → 56 |

These are CPU-enumerated descriptors, not GPU memory or latency measurements.
The enumeration covers 2,560 feasible request/row combinations across the four
configurations. Its padding histogram is not weighted by serving traffic.

## Validation and all attempts

- Full canonical source replay reproduced candidate-40 before the new patches,
  then reproduced the exact experiment head/tree. Commands and receipts are in
  `runs/source-replay*`; the earlier independent base replay is retained too.
- Final CPU checks: **70 passed, 18 deselected**. They cover the whole 28-test
  graph-manager module, 37 warmup cases, and five context-shape cases. The 18
  unrelated embedding-collective naming cases require a global fixture omitted
  by this isolated harness and are explicitly deselected.
- The CPU container uses the pinned candidate-38 dependency image
  `sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b`,
  with the complete C38→C40 Python delta and this proposal mounted read-only.
  It has no GPU devices or network. The harness selects the real CPU platform,
  imports the installed TileLang/Triton packages for module decorators, and
  extracts the actual context shape test bodies from their GPU-guarded file.
  This is not a release-image gate.
- `doctor` passes for the experimental TP3 profile using the existing DGX site
  map kept outside Git. `git diff --check` passes. Ruff reports only three
  pre-existing findings in untouched lines: one long line in the context test
  and two unused imports in `dspark.py`; `runs/ruff.json` retains them.
- Complete locked build inputs now pass `build prepare` and `build check` for
  all three implemented policies. `seed_build_inputs.py` checks out exact Git
  objects from remote caches, verifies heads/trees/submodules, and copies only
  hash-verified wheels; the normal preparation command then exports and hashes
  the contexts. `runs/build-preparation-commands.json` records each command and
  outcome. The initial missing-context check is retained. A bounded CPU-only image build is in progress; no GPU smoke or
  serving work has run under a foreign hold.
- The first cache-seeding attempt encountered an empty nested `.gitmodules`
  file; Git returns status 1 for its empty path query. The seeder now accepts
  that empty result and safely resumes only exact clean checkouts. Both initial
  and successful logs are retained; no upstream source or cache was modified.
- Initial audit harness execution failed with `ValueError: empty body on
  ClassDef`: its method filter also stripped the descriptor dataclass. The
  harness was corrected before either recorded enumeration.
- First CPU collection failed because the CPU-only image path left TileLang
  unbound. Importing the real installed package exposed two capture tests whose
  stream placeholder required the CPU platform. After selecting that platform,
  all 28 manager tests passed. The broader attempt then recorded 65 passes,
  five context-import failures from disabled Triton, and 18 missing-fixture
  errors. Loading the installed Triton module and explicitly deselecting those
  unrelated fixture-dependent cases yielded the final 70 passes. All outputs
  are retained; no candidate GPU run failed or passed because none ran.
- Initial doctor attempts retained the missing-site, example-map, SP8-map and
  stale copied patch-count errors. The isolated profile now uses the DGX site
  map and the correct patch count. No existing site map was edited.

All substantial work after workstation recovery ran remotely. Source preparation
and tests on SP8 were capped at one CPU and 4 GiB; the test container separately
enforces those limits. Initial capacity checks observed 115 GiB available on
SP8 and 111 GiB on DGX4. These are host observations, not measured test peaks.
No GPU peak, capture-pool growth, startup fit, latency, confidence interval, or
c1/c8/c16 serving result is available yet. Original release runs were not rerun.

The separate unvalidated MXFP8 tail-read prototype is preserved outside this
PR in the isolated remote workspace under
`/home/chris/spark-work/verification-shapes/deployment-live-rows/experiments/2026-11-05-mxfp8-live-rows`.
It is not part of this profile or source PR; no GEMM kernels change here.

## Reproduction

On a remote host with the pinned dependency image and a prepared source tree:

```sh
.venv/bin/python experiments/2026-11-05-verification-shapes/audit_shapes.py /path/to/source > /path/to/new-shapes.json
.venv/bin/python experiments/2026-11-05-verification-shapes/run_cpu_checks.py --source /path/to/source --out /path/to/new-cpu-run
.venv/bin/python experiments/2026-11-05-verification-shapes/replay_source.py --cache /path/to/git-cache --destination /path/to/fresh-source --out /path/to/new-replay-receipt
bin/spark --cluster-config experiments/2026-11-05-verification-shapes/tp3.json doctor
bin/spark --cluster-config experiments/2026-11-05-verification-shapes/tp3.json build check
git diff --check
```

Use `systemd-run --user --scope -p MemoryMax=4G -p CPUQuota=100%` around the
CPU commands. The actual test container argv is recorded in
`runs/cpu-test-command.json`. Preserve each new run in a new output directory.

## Remaining qualification and decision

The source fix is supported by dispatch/warmup checks and is ready for review.
It is not ready for runtime adoption. See `lab-plan.md` for the concrete gate
and coordinated boot/repeat sequence. The added graph captures need a measured
memory budget, exact outputs under mixed batches, and performance preservation within the
predeclared comparison bounds. c1/c8/c16 alone do not isolate the repaired 3/5/6/7-request full-
depth cases, so those boundary cases are included in the plan too.

The follow-up [alternatives audit](alternatives.md) compares eight capture policies
and bounds the matrix-tile savings of this proposal at zero for four representative
wide dense projections. Native-row-only and union ladders remain credible
competitors. The source PR is a coverage repair, not a proved hardware winner.

On 2026-11-06 local time the owner explicitly authorized hardware testing,
including normal coordinated sync, stop/start/restart or replacement needed for
measurement and restoration. This removes the earlier authorization blocker;
it does not permit a foreign hold, promotion, merging, or a topology change.

The row-only and union alternatives are implemented and replayed, with final
70-pass focused CPU checks for each. See `alternatives.md`, `variants/`, and
`runs/implemented-alternative-census.json`; their actual descriptors match the
structural comparison and add no piecewise graphs. They remain unmeasured on
GPU, and no policy has been selected as a performance winner.

The strict published-branch controller has 20 passing CPU checks and retains
its failed setup attempts. Both functional/performance specs pass offline
preflight for all three policies. The CPU-only build driver uses the official
locked build command with a separate two-CPU/24-GiB worker, no GPU device
entitlement/CDI, and a mandatory worker device probe. Its output remains
unqualified until the owned-window GPU smoke, image tests, sanitizers and
serving checks finish. See `lab-plan.md` for the complete ownership, restoration
and resource contract.


The pending hardware path now has a bounded kernel-screen stage and a separate
full release-gate stage. Their resume checks bind the execution commit, source
labels, immutable image content, four required checks and log hashes. The
serving controller accepts faithful Docker-store copies by ordered layers and
runtime configuration, while recording each host's local image ID.

The latest controller/evidence preflight has **41 passing CPU checks**, clean
lint and a passing doctor. It tests image-copy drift, incomplete/replaced screen
receipts, remote cleanup identities, all boot pairings and non-inferiority
bounds. Initial lint and guardian-source-generation failures are retained in
`runs/gate-preflight*`; neither ran on hardware. Boundary replay probes remain
GPU-unvalidated. The candidate performance spec includes new 3/5/6/7-request
full-model dummy steps and c1/c8/c16 client load on both performance boots.
See `lab-plan.md` for the distinction between those paths and their bounds.

The task's disposable Mac checkout was removed after its untracked source and
evidence were copied and hash-verified on SP8, and both original commits were
verified there. That removed 578,605,056 allocated bytes. The shared local
checkout and caches were not changed. Bulk preparation, tests, build data and
unfinished work remain remote.


### Completed CPU image build (2026-11-06)

The exact native-depth image finished at 01:49:51 UTC from deployment
`eb46c4f20ea076ff48549a45e615276b1d738d86`, source
`8dea43f62a149a3aef2be4b24fa1c68e9b33407d`, and tree
`3f9b475dfcd86aabbc042299131cd38aa3612be0`.
Its SP8 image identity is
`sha256:3e03ac089d50d1408272aac0dc32ab134109e228a1d8dc8f8310688ccdb16209`.
The isolated worker exposed no GPU device paths, used one compile job within
its two-CPU/24-GiB ceiling, and stopped cleanly. Minimum host available memory
was 106.48 GiB. Exact commands, all compiler output, inspection and hashes are
retained under `runs/cpu-build-attempt-2/`.

This completes only CPU build preparation. The image remains unqualified:
native GPU replay, sanitizers, applicable release tests, serving correctness,
and repeated performance/memory preservation have not run. No production
configuration or immutable baseline has changed.

Qualification-only follow-up checks now enforce the experiment's deterministic
verdicts and sampled isolated repeats before timing, while retaining sampled
batch differences as observations. The affected-cell grid requires complete
full-graph and drafter evidence at each requested live shape. It does not claim
that its existing mode fields expose padded capture sizes. The latest bounded
CPU preflight is 54 passed, with clean lint, configuration validation and
`git diff --check`; earlier unsuccessful preflights remain recorded.

The immutable image was subsequently streamed from SP8 to DGX4, without a
workstation archive. Both endpoints recorded 10,347,186,017 compressed bytes
and SHA-256 `699f1bcf829764f677d16c954027f1b2483977e391b8d9bc02eecbb8a1e15ef8`.
DGX4 retained the same image ID. Static validation confirmed exact ordered
layers, runtime configuration, source labels and clean prepared source, and
accepted only qualification/evidence changes after the build commit.
`runs/image-transfer-1/` preserves the commands, stream receipts, loader logs
and validation record. No GPU operation was run by this transfer or validation.


### DGX4 2-boot GPU screen and shape replay benchmark (2026-11-06)

Executed the 2-boot GPU kernel screen and native-depth verification shape
replay benchmark on `dgx4` under the `dgx4-single` node-scoped hold
(`token: dgx4-verification-shapes-20261106`) against the verified image
`sha256:3e03ac089d50d1408272aac0dc32ab134109e228a1d8dc8f8310688ccdb16209`:

- **Initial screen attempt (`runs/screen-attempt-1/`)**: `gpu-smoke` passed,
  `graph-replay` passed (108/108 in 21.63s), and `memcheck` passed (108/108 in
  10.48s, `ERROR SUMMARY: 0 errors`), exposing a log-parser bug in `gpu_gate.py`
  where `suite_verdict` matched `"0 errors"` on the trailing
  `compute-sanitizer` summary line instead of the pytest summary line. Fixed in
  `98e1f70c44fbd80194c48ee97e0e35619a0f01cd` (`sanitizer_passed`).
- **2-boot GPU screen (`runs/dgx4-gpu-screen-2boot/boot-1/` and `boot-2/`)**:
  - `gpu-smoke`: passed on both boots (`exit 0`).
  - `graph-replay`: 108/108 passed on both boots (`21.6s`, exact output match
    across all 36 boundary row counts for `dspark_context.context_inputs` and
    `mxfp8` decode/split-K kernels, including poisoned-tail CUDA graph replay).
  - `memcheck`: 108/108 passed with `ERROR SUMMARY: 0 errors` on both boots.
  - `initcheck` on `test_context_inputs_replay` (`initcheck-context-inputs.log`):
    36/36 passed with `ERROR SUMMARY: 0 errors` on both boots (`6.36s`),
    confirming zero uninitialized reads in `dspark_context.context_inputs`.
  - Full `initcheck` (`initcheck.log`): 108/108 pytest output assertions passed
    on both boots, while `compute-sanitizer --tool initcheck` exited `97` with
    `27,288` uninitialized global-memory reads of size 8 bytes in
    `gemm_mxfp8_kernel` (`tl::cp_async_gs_conditional<(int)8>` at `copy.h:68`,
    launched at `test_graph_replay.py:160` `mxfp8_gemm_decode(..., padded_rows=True)`).
    As specified in `lab-plan.md`, this pre-existing MXFP8 padded-tail read
    blocks full gate promotion until paired with the separate live-row bound fix
    (`experiments/2026-11-05-mxfp8-live-rows`).
- **2-boot CUDA graph replay benchmark (`shape-benchmark-boot-1.json`, `shape-benchmark-boot-2.json`)**:
  - All repaired verification request counts (`num_reqs = 3, 5, 6, 7`,
    `native_rows = 18, 30, 36, 42` vs `wider_padded_rows = 20, 32, 40, 48`)
    verified bit-identical output (`exact_match: true`) across both boots.
  - Median CUDA graph replay latency on `dgx4` (Boot 1 / Boot 2):
    - `num_reqs=3` (`18` vs `20` rows): `context_inputs` `3.35 / 3.35 us` vs
      `3.35 / 3.35 us`; `mxfp8_decode` `3.96 / 3.96 us` vs `3.92 / 3.95 us`;
      `mxfp8_splitk` `4.08 / 4.09 us` vs `4.09 / 4.09 us`.
    - `num_reqs=5` (`30` vs `32` rows): `context_inputs` `4.09 / 4.09 us` vs
      `4.09 / 4.09 us`; `mxfp8_decode` `3.95 / 3.96 us` vs `3.95 / 4.05 us`;
      `mxfp8_splitk` `4.09 / 4.09 us` vs `4.09 / 4.09 us`.
    - `num_reqs=6` (`36` vs `40` rows): `context_inputs` `4.10 / 4.10 us` vs
      `4.09 / 4.09 us`; `mxfp8_decode` `4.09 / 4.09 us` vs `4.09 / 4.09 us`;
      `mxfp8_splitk` `4.09 / 4.09 us` vs `4.09 / 4.09 us`.
    - `num_reqs=7` (`42` vs `48` rows): `context_inputs` `4.10 / 4.11 us` vs
      `4.11 / 4.12 us`; `mxfp8_decode` `4.09 / 4.09 us` vs `4.10 / 4.10 us`;
      `mxfp8_splitk` `4.10 / 4.10 us` vs `4.09 / 4.09 us`.


### Paired live-row MXFP8 bound + 2-boot 0-error `initcheck` and `Sparse-MLA` speedup (2026-11-06)

Paired the native-depth verification shape fix with `0003-bound-mxfp8-live-rows.patch`
(`2541c9a1372f7f35589b0aafc343b00711999de9`, preserved under
`runs/paired-mxfp8-live-rows/`) and re-ran the full 2-boot GPU screen and
row-proportional `Sparse-MLA` (`vllm/models/deepseek_v4_1/tilelang/sparse_mla.py`)
benchmark on `dgx4` (`runs/dgx4-gpu-screen-v2-2boot/`):

1. **Unambiguous 2-boot GPU gate pass (`screen_passed: true`, `exit 0` on both boots)**:
   - `gpu-smoke`: passed on Boot 1 & Boot 2 (`exit 0`).
   - `graph-replay` (`test_graph_replay.py`): `108/108 passed` on Boot 1 & Boot 2 (`exit 0`).
   - `memcheck` (`compute-sanitizer --tool memcheck`): `108/108 passed`,
     `ERROR SUMMARY: 0 errors` (`exit 0`) on Boot 1 & Boot 2.
   - `initcheck` (`compute-sanitizer --tool initcheck`): **`108/108 passed` in
     `9.88s` (Boot 1) and `9.93s` (Boot 2) with `ERROR SUMMARY: 0 errors` (`exit 0`)**
     on both boots.
2. **2-boot `Sparse-MLA` (`decode` & `extend`) verification-shape speedup (`heads=24`, TP3 per-rank)**:
   - In `Sparse-MLA` (`SparseMlaV41Plan`), each token row launches its own CTAs
     (`1` CTA/row in `extend`; `5` segment CTAs/row + merge in `decode` for
     `swa_width=128, idx_width=512`), whereas `mxfp8_decode` tiles in `M = 16/32/64`.
   - Across all four affected verification request counts (`num_reqs = 3, 5, 6, 7`),
     live-row outputs are bit-identical between native and wider padded shapes
     and between `extend` and `decode` schedules (`exact_match: true`).
   - Measured `Sparse-MLA` `decode` median CUDA graph replay latency (`us`, Boot 1 / Boot 2):
     - **`num_reqs=3` (`18` vs `20` rows, `90` vs `100` CTAs)**:
       `29.09 / 29.15 us` vs `29.93 / 30.25 us` (**`+2.79% / +3.62%` speedup**,
       saves `65.0–85.3 us` across 78 layers).
     - **`num_reqs=5` (`30` vs `32` rows, `150` vs `160` CTAs)**:
       `43.14 / 43.06 us` vs `43.26 / 43.25 us` (**`+0.28% / +0.45%` median speedup**,
       and p95 latency drops from `52.24 / 53.24 us` to `43.95 / 43.65 us`,
       **`-15.9% / -18.0%` p95 reduction**).
     - **`num_reqs=6` (`36` vs `40` rows, `180` vs `200` CTAs)**:
       `45.28 / 45.10 us` vs `47.76 / 46.75 us` (**`+5.19% / +3.52%` speedup**,
       saves `128.4–193.4 us` across 78 layers).
     - **`num_reqs=7` (`42` vs `48` rows, `210` vs `240` CTAs)**:
       `50.22 / 49.71 us` vs `67.32 / 67.09 us` (**`+25.40% / +25.90%` speedup**,
       saves `17.10–17.37 us/layer` or **`1.33–1.35 ms` across 78 layers** by
       avoiding the `240`-CTA wave spill at `48` rows).
3. **Policy selection (optionality removed)**:
   - Retaining exact native full-depth shapes alongside wider copy-verification
     shapes (`0001-native-depth-captures.patch` + `0002-warmup-width-fixture.patch`
     paired with `0003-bound-mxfp8-live-rows.patch`) is selected as the single
     winning policy: it restores 100% full-depth verification graph coverage
     and achieves up to `25.9%` per-layer `Sparse-MLA` speedup with only 50
     captures (vs 65 in `shape-union`), while preserving exact request-count
     specialization (unlike `row-ladder`). Runtime `--variant` optionality has
     been removed from `gpu_gate.py`.
