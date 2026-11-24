# Candidate `candidate-72`

Date: 2026-11-23 / 2026-11-24.
Base: `candidate-71` (`731c88e75f`, built on `spark-ds41f-1122` / vLLM `0001..0865`), plus upstream `dgx-spark-networking` `v0.6.0` (`b01f350d061a6434c2186df73fa568ab4a9e4aa0`), PR #1141 (opt-in buffered atomic admission), PR #1319 (GLP-27 weightless steering + `0.3/0.95` defaults), and vLLM patch `0867` (env registration + chunked `y_all` scatter).

## Scope & Delta Over `candidate-71`

1. **Zero local `sparknet` patches (`dgx-spark-networking` `v0.6.0`)**:
   - Pins upstream `dgx-spark-networking` `v0.6.0` (`b01f350d061a6434c2186df73fa568ab4a9e4aa0`, tree `1739407706277142da7a12c95fca3c503c8cb11c`, `patch_count = 0`, `patchset_sha256 = e3b0c442...`).
   - Upstream `v0.6.0` integrates all three previously carried local patches (proxy core pinning, single-kernel multi-stage collectives, PDL chaining) plus one-shot RoCE P2P (`ROCE_P2P_ABI_VERSION = 1`, `1 MiB` ring slots, subnet-based GID resolution).
2. **vLLM patch `0866` — Opt-in buffered atomic admission (rebased PR #1141)**:
   - Adds opt-in buffered atomic admission on `/v1/chat/completions/batch` when `x-atomic-admission: 1` (or `x-tp8-atomic-admission: 1`) is sent.
   - Ordinary requests and non-header batch requests remain untouched.
   - Includes `tests/entrypoints/openai/test_buffered_atomic.py`.
3. **vLLM patch `0867` — Env var registration + chunked bit-exact prefill scatter**:
   - Registers `VLLM_PP_SPARKNET_P2P`, `VLLM_DS41_EXPERT_HOTLIST`, `VLLM_DS41_EXPERT_HOTLIST_MODE`, `VLLM_DS41_EXPERT_FAST_LOOKAHEAD`, and `VLLM_DS41_EXPERT_PREFILL_MODE` in `vllm/envs.py`, eliminating unknown-environment-variable startup warnings.
   - Chunks `y_all.index_copy_` in `_run_expert_centric_prefill` into `2048`-route slices (`~20 MiB` temporary instead of `~320 MiB`), substantially increasing `MemAvailable` headroom on `tp1`/`tp2` `131K` bit-exact prefill while preserving bit-exact output.
   - Total vLLM series: `0001..0867` (`740` patches, `patch_head = b851a087570b55a3430c7ef867735621bd1bdb07`, `expected_tree = babd6b07c4d2ef5084a0bcc97103abf0d2abc783`, `patchset_sha256 = 14a3fe59376c6b0ce12b366202a01a7ee1d3ee3fb5b3f3c291fe3881fe427a94`).
4. **Weightless steering (`GLP-27-L11-37-a3.5`, PR #1319)**:
   - All 17 `candidate-72` cluster profiles configure `GLP-27-L11-37-a3.5` (`file_sha256 = 24e97d41a9a4f77a...`, serve-time `alpha = 2.0`, `alpha_default = 3.5`) with per-model generation defaults `temperature = 0.3, top_p = 0.95`.
5. **`PP=2` KV cache & documentation**:
   - Raised `pp2-tp2-ring` and `pp2-tp2-switched` KV cache to `24.0 GiB` (`25,769,803,776` B = `30,700,347` KV tokens, `29.28x` concurrency at `1M` context), matching `pp2-tp3-switched` (`24.0 GiB`) and `pp2-tp4-switched` (`36.0 GiB`).
   - Added "When to use `PP=2` vs `TP`" guidance to `docs/switchless-topology.md`.

## Build & Unit Test Verification

- **Image**: `spark-ds41f:candidate-72`
  - OCI digest: `sha256:b9549252c02eb41731de052adbd2170c030bb62e0f203c2d72512e5011bef46c`
  - Labels: `local.spark-ds41f.vllm.tree=babd6b07c4d2ef5084a0bcc97103abf0d2abc783`, `local.spark-ds41f.sparknet.tree=1739407706277142da7a12c95fca3c503c8cb11c`
  - Built on `dgx4` in `484 s` (lowest `MemAvailable = 82.2 GiB`); `sparknet = 0.6.0`, `collective_abi = 10`, `p2p_abi = 1`.
  - Distributed along `dgx4 -> dgx3-internal -> dgx2-internal -> dgx1-internal`; identical `56` `RootFS.Layers` (`layers_sha = b5ee65d1f465301a`) verified on all 4 nodes.
- **Repository validation**: `bin/spark doctor` passed across checked `candidate-72` profiles (`tp4`, `tp3`, `pp2-tp2-ring`, `pp2-tp3-switched`, `pp2-tp4-switched`, `tp2-direct`, `tp1`).
- **Zero-overlay built-image unit tests on `dgx4`**: **`154 passed, 2 skipped, 0 failed`** across:
  - `tests/entrypoints/openai/test_buffered_atomic.py` (`5/5` CPU, `CUDA_VISIBLE_DEVICES=""`)
  - `tests/models/deepseek_v4_1/test_deepseek_v41_expert_slots.py` (`14/14` strict `torch.equal` GPU + `io_uring`)
  - `tests/models/deepseek_v4_1/test_expert_hotlist_and_fast_lookahead.py` (`3/3` GPU)
  - `tests/models/deepseek_v4_1/test_deepseek_v4_1_glp.py`
  - `tests/distributed/test_pp_sparknet_p2p.py`
  - `tests/v1/spec_decode/test_adaptive_verification.py`

## Live Feature & Batch-Invariance Verification (`spark-ds41f:candidate-72`, commit `6fa3c56b34`)

All live checks were executed from `dgx1` inside an owned cluster hold (`dgx1-1795474073`), after which production `tp4-ring` (`config/cluster.json`, `spark-ds41f:1122`, `origin/main`) was restored and verified with `bin/spark --cluster-config config/cluster.json doctor --live`.

### 1. Startup Logs & Feature Receipts
- **Zero unknown-env warnings**: confirmed clean startup logs on `tp4-ring`, `pp2-tp2-ring`, and `tp2-direct` for all newly registered env vars (`VLLM_PP_SPARKNET_P2P`, `VLLM_DS41_EXPERT_*`).
- **`GLP-27` live (`tp4-ring`, `pp2-tp2-ring`, `tp2-direct`)**:
  - Startup log on every tested shape: `GLP steering on (hc_fold_pre_engram, spec_version=3): alpha=2.0 (alpha_default=3.5, source=override), layers=27 [11..37], file_sha256=24e97d41a9a4f77a, content_sha256=f0aaba2fd1d5d52d, requests to 'deepseek-v4.1-flash-weightless'`.
  - Live auto `tool_choice` check under `0.3/0.95` defaults on `tp4-ring`: **`6/6` pass** on `deepseek-v4.1-flash` and **`6/6` pass** on `deepseek-v4.1-flash-weightless`; stock vs weightless logprobs differ (`True`).
- **PR #1141 Buffered Atomic Admission A/B on `tp4-ring`** (`/v1/chat/completions/batch`, 5 alternating pairs × 16 prompts × 192 tokens):
  - **Exact parity**: **`160/160` lanes 100% identical** across ordinary batch, `x-atomic-admission: 1`, and `x-tp8-atomic-admission: 1`.
  - **Throughput**: 5-pair mean `274.4 -> 275.5 tok/s` (`+0.38%`, neutral/slightly positive on 4-node `tp4-ring`; PR #1141 measured `+8-13%` on 8-node `tp8` buffered C16).

### 2. Temperature-0 Batch-Invariance Gates (`scripts/sonar_determinism.py`)

| Shape | Mode | Scenarios | Token Diffs | Logprob Diffs | `reference_sha256` | `long_sha256` |
| --- | --- | --- | ---: | ---: | --- | --- |
| `tp4-ring` | `--wide` (`192` tok) | `c8, staggered, mixed, cached, again, mixed15` | **`0`** | **`0`** | `18d619615e51...` | `547c1c1cbf91...` |
| `pp2-tp2-ring` (`24.0 GiB` KV) | `--wide` (`192` tok) | `c8, staggered, mixed, cached, again, mixed15` | **`0`** | **`0`** | `0c521e72c482...` | `b4cfd5f84b1f77dd...` |
| `tp2-direct` (`240/64` + `0867`) | `--tokens 128` | `c8, staggered, mixed, cached, again` | **`0`** | **`0`** | `ce46870c9bf6...` | `b4cfd5f84b1f77dd...` |

*(Note: `tp2-direct`'s NVMe-streaming expert slot cache with `0865` + chunked `0867` produces the exact same `long_sha256` (`b4cfd5f84b1f77dd2d5e9b610cea2ece69fe5cfc7e1c84f6e4366528efcc92ed`) and `long_text_sha256` (`1dd5b1dbdb63fe833df26cda175a01d7b33acea239d9c04e66c85ea702678249`) as full-GPU `pp2-tp2-ring`.)*

## Live Two-Boot Verification Screens (`spark-ds41f:candidate-72`)

Commands executed from `dgx1`:

```sh
bin/spark --cluster-config <cfg> cluster start --replace --change-shape --apply
bin/spark --cluster-config <cfg> bench --suites quality,decode,prefill \
  --concurrency <conc> --prefill-sizes 32768,131072 --prefill-repeats 2 \
  --min-samples 3 --max-samples 5 --cool-below 0 --compare none \
  --output results/private/bench/<run>
```

### 1. `tp4-ring` (`dgx1..dgx4`, `1M` ctx, `16.5 GiB` KV) vs `manifests/benchmarks/2026-11-22-spark-ds41f-1122-tp4.json`

| Metric | `1122-tp4` | `c72-tp4-b1` (`+99.4 s`) | `c72-tp4-b2` (`+83.1 s`) | 2-Boot Mean | Delta vs `1122` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Quality | `LRU 5/5` | `LRU 5/5` | `LRU 5/5` | `LRU 5/5` | pass |
| `prose-c1` (tok/s / step ms) | `84.2 ±1.1%` (`25.78 ms`) | `84.4 ±3.0%` (`25.73 ms`) | `84.6 ±0.7%` (`25.52 ms`) | **`84.5`** (`25.63 ms`) | `+0.4%` (parity) |
| `prose-c2` | `147.9 ±1.2%` (`26.79 ms`) | `149.0 ±1.1%` (`26.58 ms`) | `145.6 ±5.8%` (`27.23 ms`) | **`147.3`** (`26.91 ms`) | `-0.4%` (parity) |
| `prose-c4` | `277.0 ±1.1%` (`26.96 ms`) | `276.8 ±1.6%` (`26.99 ms`) | `277.5 ±1.4%` (`26.92 ms`) | **`277.2`** (`26.96 ms`) | `+0.1%` (parity) |
| `prose-c8` | `431.6 ±12.7%` (`36.77 ms`) | `380.0 ±1.6%` (`42.73 ms`) | `403.6 ±8.0%` (`39.86 ms`) | `391.8` (`41.30 ms`) | within ±12.7% CI (`b1` low) |
| `code-c1` | `104.1 ±0.6%` (`28.60 ms`) | `104.1 ±0.4%` (`28.50 ms`) | `104.0 ±0.5%` (`28.37 ms`) | **`104.1`** (`28.44 ms`) | `0.0%` (parity) |
| `code-c2` | `160.2 ±17.2%` (`35.14 ms`) | `171.4 ±26.3%` (`31.39 ms`) | `159.6 ±14.1%` (`34.36 ms`) | **`165.5`** (`32.88 ms`) | `+3.3%` (within CI) |
| `code-c4` | `317.2 ±4.4%` (`32.32 ms`) | `326.2 ±4.1%` (`31.21 ms`) | `312.3 ±5.0%` (`32.51 ms`) | **`319.3`** (`31.86 ms`) | `+0.7%` (parity) |
| `code-c8` | `544.4 ±4.9%` (`37.57 ms`) | `554.2 ±1.4%` (`36.59 ms`) | `539.4 ±4.0%` (`37.25 ms`) | **`546.8`** (`36.92 ms`) | `+0.4%` (parity) |
| `prefill 32K` (tok/s) | `10,185.7 ±3.9%` | `10,203.5 ±5.4%` | `10,307.8 ±11.1%` | **`10,255.7`** | `+0.7%` (parity) |
| `prefill 131K` (tok/s) | `10,543.0 ±0.9%` | `10,530.2 ±4.5%` | `10,484.9 ±3.1%` | **`10,507.6`** | `-0.3%` (parity) |
| Min `MemAvailable` (GiB) | — | `14.22–16.20` (0 swap) | `14.29–16.27` (0 swap) | `>=14.22 GiB` | 0 thermal slowdown |

*(Note: `c72-tp4-b1` recorded a slower `prose-c8` sample cluster at `380.0 ±1.6%` (`42.73 ms`), while `c72-tp4-b2` recorded `403.6 ±8.0%` (`39.86 ms`) and `c72-tp4-b1` `code-c8` hit `554.2 ±1.4%`. All other `tp4-ring` decode and prefill points are within `±0.7%` of `1122-tp4`.)*

### 2. `pp2-tp2-ring` (`dgx1, dgx2, dgx4, dgx3`, `1M` ctx, **`24.0 GiB` KV = `30,700,347` tok / `29.28x`**) vs `manifests/benchmarks/2026-11-22-spark-ds41f-1122-pp2tp2ring.json`

| Metric | `1122-pp2tp2ring` (`16.5 GiB` KV) | `c72-pp2-b1` (`+96.9 s`) | `c72-pp2-b2` (`+82.3 s`) | 2-Boot Mean | Delta vs `1122` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Quality | `LRU 5/5` | `LRU 5/5` | `LRU 5/5` | `LRU 5/5` | pass |
| `prose-c1` (tok/s / step ms) | `40.1` (`56.73 ms`) | `47.9 ±1.2%` (`44.31 ms`) | `48.1 ±1.2%` (`44.07 ms`) | **`48.0`** (`44.19 ms`) | **`+19.7%`** (`-22.1%` step) |
| `prose-c2` | `76.2` (`59.90 ms`) | `85.3 ±1.9%` (`48.18 ms`) | `85.7 ±1.6%` (`48.69 ms`) | **`85.5`** (`48.44 ms`) | **`+12.2%`** (`-19.1%` step) |
| `prose-c4` | `123.3` (`74.53 ms`) | `146.3 ±5.7%` (`54.09 ms`) | `146.8 ±5.8%` (`54.81 ms`) | **`146.6`** (`54.45 ms`) | **`+18.9%`** (`-26.9%` step) |
| `prose-c8` | `210.2` (`87.97 ms`) | `239.8 ±7.2%` (`70.99 ms`) | `225.0 ±11.6%` (`76.42 ms`) | **`232.4`** (`73.71 ms`) | **`+10.6%`** (`-16.2%` step) |
| `code-c1` | `50.3` (`59.79 ms`) | `53.5 ±1.4%` (`47.65 ms`) | `53.1 ±0.3%` (`47.97 ms`) | **`53.3`** (`47.81 ms`) | **`+6.0%`** (`-20.0%` step) |
| `code-c2` | `93.2` (`64.86 ms`) | `96.1 ±3.0%` (`52.45 ms`) | `96.1 ±2.7%` (`52.03 ms`) | **`96.1`** (`52.24 ms`) | **`+3.1%`** (`-19.5%` step) |
| `code-c4` | `148.2` (`82.38 ms`) | `146.6 ±6.9%` (`60.46 ms`) | `150.5 ±7.5%` (`59.49 ms`) | **`148.6`** (`59.98 ms`) | `+0.3%` (`-27.2%` step) |
| `code-c8` | `235.5` (`104.38 ms`) | `243.6 ±16.9%` (`76.25 ms`) | `268.9 ±11.8%` (`68.07 ms`) | **`256.3`** (`72.16 ms`) | **`+8.8%`** (`-30.9%` step) |
| `prefill 32K` (tok/s) | `5,571.7 ±4.2%` | `6,537.6 ±8.9%` | `6,502.4 ±9.3%` | **`6,520.0`** | **`+17.0%`** |
| `prefill 131K` (tok/s) | `5,727.5 ±0.2%` | `6,789.2 ±0.9%` | `6,792.6 ±2.3%` | **`6,790.9`** | **`+18.6%`** |
| Min `MemAvailable` (GiB) | — | `10.97–13.93` (0 swap) | `11.00–13.86` (0 swap) | `>=10.97 GiB` | 0 thermal slowdown |

### 3. `tp2-direct` (`dgx1, dgx2`, `256K` ctx, `6.0 GiB` KV = `6.44M` tok, `240/64` slots + bit-exact `0865` + chunked `0867`) vs `manifests/benchmarks/2026-11-20-spark-ds41f-1120-direct2.json`

| Metric | `1120-direct2` | `c72-tp2-b1` (`+114.7 s`) | `c72-tp2-b2` (`+89.9 s`) | 2-Boot Mean | Delta vs `1120` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Quality | `LRU 5/5` | `LRU 5/5` | `LRU 5/5` | `LRU 5/5` | pass |
| `prose-c1` | `17.7` (`121.37 ms`) | `28.0 ±5.0%` (`74.69 ms`) | `27.8 ±5.1%` (`76.25 ms`) | **`27.9`** (`75.47 ms`) | **`+57.6%`** (`-37.8%` step) |
| `prose-c4` | `53.0` (`153.91 ms`) | `92.5 ±4.6%` (`83.10 ms`) | `90.8 ±6.2%` (`85.29 ms`) | **`91.7`** (`84.20 ms`) | **`+72.9%`** (`-45.3%` step) |
| `code-c1` | `22.2` (`127.53 ms`) | `31.1 ±4.1%` (`76.16 ms`) | `31.0 ±4.0%` (`76.24 ms`) | **`31.1`** (`76.20 ms`) | **`+39.9%`** (`-40.2%` step) |
| `code-c4` | `64.2` (`166.48 ms`) | `92.3 ±4.2%` (`83.34 ms`) | `91.3 ±3.8%` (`85.59 ms`) | **`91.8`** (`84.47 ms`) | **`+43.0%`** (`-49.3%` step) |
| `prefill 32K` (tok/s) | `258.8 ±73.8%` | `4,443.9 ±40.8%` | `4,378.6 ±40.1%` | **`4,411.3`** | **`17.0x`** |
| `prefill 131K` (tok/s) | `235.6 ±15.9%` | `4,733.2 ±17.2%` | `4,705.2 ±26.3%` | **`4,719.2`** | **`20.0x`** |
| Min `MemAvailable` (GiB) | — | `8.55 / 10.32` (0 swap) | `9.56 / 10.88` (0 swap) | `>=8.55 GiB` | 0 thermal slowdown |

*(Chunked `0867` `y_all` scatter raised `tp2-direct` `131K` prefill `MemAvailable` floor from `6.48 / 7.84 GiB` in `candidate-71-r2` to `8.55–9.56 / 10.32–10.88 GiB` while keeping bit-exact determinism.)*

## Revision `candidate-72-r2` (Dropped Opt-In Buffered Atomic Admission)

Date: 2026-11-24.
Commit: `ff1c376962` (on `origin/candidate-72`).

After reviewing the opt-in buffered atomic admission patch (PR #1141): it only applies to `/v1/chat/completions/batch` under narrow experimental constraints (`temperature=0`, `max_tokens<=256`, `prompt<=2048`, `batch<=16`, custom header) with `+0.38%` delta on 4-node `tp4-ring` and zero effect on standard `/v1/chat/completions`. Removed rather than carrying an unused opt-in wire protocol:

- **vLLM series (`0001..0866`, `739` patches)**: dropped `0866-perf-serving-admit-a-buffered-text-batch-in-one-engi.patch` and renumbered `0867` -> `0866-chore-ds41-register-DS4.1-PP-env-vars-in-envs.py-and.patch`.
  - `patch_head = 11ff09dccfa89264f50089c46710bdc30ee75fb0`
  - `expected_tree = 74e59cbfe31df3bd67fb75de6a165647b400e96f`
  - `patchset_sha256 = f67d540529599a3a29dadd5fd07d1cf2446222dd9fc07a8e34e88aa0cc215bd9`
- **Image**: `spark-ds41f:candidate-72-r2` (superseded `spark-ds41f:candidate-72` deleted from `dgx1..dgx4`).
  - OCI digest: `sha256:bca67b12033acedd4160e8e411e7ee99aa9fc53b5b54861e4ad9c426f728af4b`
  - Built on `dgx4` in `474 s` (lowest `MemAvailable = 82.6 GiB`); GPU import smoke passed (`DS4.1 SM12x`, `TileLang`, `sparknet 0.6.0` ABI `10`, B12X-free).
  - Distributed along `dgx4 -> dgx3-internal -> dgx2-internal -> dgx1-internal`; identical `56` `RootFS.Layers` (`layers_sha = a539fbd431f3f898`) and labels (`vllm=74e59cbfe31d`, `sparknet=173940770627`) verified on all 4 nodes.
- **Validation & Verification**:
  - `bin/spark doctor` passed (`configuration OK`) across `tp4`, `tp3`, `pp2-tp2-ring`, `pp2-tp3-switched`, `pp2-tp4-switched`, `tp2-direct`, `tp1`, and `image-dgx4`.
  - Zero-overlay built-image unit tests on `dgx4`: **`145 passed, 2 skipped, 0 failed`** (`81.60 s`), plus verified `ATOMIC_TEXT_BATCH_LIMIT` and `test_buffered_atomic.py` are absent (`atomic_absent=True`).
  - Live `tp4-ring` smoke (`+81.4 s`): `GLP-27 alpha=2.0` loaded cleanly, zero unknown-env warnings, zero `TP8_*` debug logs; `scripts/sonar_determinism.py --wide`: **`0` token diffs, `0` logprob diffs** across all 6 scenarios (`reference_sha256 = 18d619615e51c7df...`, `long_sha256 = 547c1c1cbf911045...`, bit-identical to `candidate-72` r1).
  - Production `tp4-ring` (`config/cluster.json`, `spark-ds41f:1122`, `origin/main` @ `945d95d03`) restored (`+80.0 s`) and verified with `bin/spark --cluster-config config/cluster.json doctor --live`.

## Revision `candidate-72-r3` (Full Step 9 Release Qualification)

Date: 2026-11-24.
Commit: `5099b083f6` (on `origin/candidate-72`, rebased onto `origin/main` `75fb3e72c6` after reverting premature `spark-ds41f-1122`).

### Delta Over `candidate-72-r2`

Running the full 174-file `release-tests` bundle on `candidate-72-r2` uncovered 6 test/stub compatibility failures (`6 failed, 12073 passed, 356 skipped`). Added vLLM patch `0867` (`0867-fix-tests-restore-release-test-compatibility-for-PP2.patch`, `740` patches total, `patch_head = 8d5315174cb88b84d48305e3418f757b471d7f22`, `expected_tree = 7eef5cb24aa415faa91b62cc89da810255944bcf`, `patchset_sha256 = 90c4f7f875d45a6ace0e6f51cfca7777b6190365ba2e62acb06f2de519456afa`):
- Defensive `getattr(...)` guards for optional stub attributes in `vllm/v1/core/sched/scheduler.py` (`force_thinking_end`, `_last_prefill_output_id`, `num_verification_tokens`) and `vllm/v1/worker/gpu/model_runner.py` (`is_last_pp_rank`, `pp_handler`, `speculative_config`) — zero runtime behavior change.
- Updated `tests/models/test_deepseek_v4_1_engram.py` (mock local Engram decoder layer under `0853` PP stage ownership), `tests/v1/worker/test_gpu_model_runner_v2.py` (assert `0860` `broadcast_drafts` confidence kwargs), and `tests/v1/worker/test_prepare_kernels.py` (`speculative_config` on `is_last_pp_rank=False` stub).
- Classified all 10 new vLLM test files in `config/release-tests.json` (`174` gate files, `0` problems).

### Image & Static / Release-Test Verification

- **Image**: `spark-ds41f:candidate-72-r3` (superseded `candidate-72-r2` deleted from `dgx1..dgx4`).
  - OCI digest: `sha256:3587a2b177a6e7fcff8abccfd90d79fe9b87dfd5220037494eff2ea4b1f36fda`
  - Labels: `local.spark-ds41f.vllm.tree=7eef5cb24aa415faa91b62cc89da810255944bcf`, `local.spark-ds41f.sparknet.tree=1739407706277142da7a12c95fca3c503c8cb11c`
  - Distributed along `dgx4 -> dgx3-internal -> dgx2-internal -> dgx1-internal`; identical `56` `RootFS.Layers` (`layers_sha = 85aa1db09dcbd8cf`) verified on all 4 nodes.
- **Full Linux repo unit suite (`dgx4`)**: **`929 passed, 0 failed`** (`32.30 s`).
- **Full 174-file `release-tests` bundle (`dgx4`, `receipts/release-tests/verdict.json`)**: **`12,079 passed, 356 skipped, 0 failed`** (`2,965.6 s`, `passed: true`).

### Determinism & Formal `tool_gate` Receipts (`candidate-72-r3`)

| Shape | Determinism Gate | Token / Logprob Diffs | `tool_gate` (`T=0.0, 0.3`, Stock + Weightless) |
| --- | --- | ---: | --- |
| `tp4-ring` | `--wide` (`192` tok, 6 scenarios) + `--no-logprobs` | **`0 / 0`** | **`80/80` strict PASS (`0` derailments)** |
| `pp2-tp2-ring` (`24.0 GiB` KV) | `--wide` (`192` tok, 6 scenarios) + `--no-logprobs` | **`0 / 0`** | **`80/80` strict PASS (`0` derailments)** |
| `tp2-direct` (`240/64` + `0866`) | `--tokens 128` (5 scenarios) | **`0 / 0`** | — |
| `tp1` (`96/48` + `0866`) | `--tokens 128` (5 scenarios) | **`0 / 0`** | — |

### `tp4-ring` Two-Boot Sonar Release Qualification vs `spark-ds41f-1120`

Receipts in `receipts/tp4-ring/{release,single-stream,sampled}-table.txt` and `c{16,8,4}-distinct.json`:
- **Startup time**: `+80.0–82.0 s` vs `spark-ds41f:1120`'s `+113.0–116.3 s` (**`-33 s` faster**).
- **`distinct_streams` (`4` samples × `256` tok)**: `c16 = 281.23 ±0.39% tok/s`, `c8 = 209.92 ±0.65% tok/s`, `c4 = 163.32 ±0.19% tok/s` (`0` foreign requests).
- **Vs recorded Nov 19 `manifests/suites/2026-11-20-spark-ds41f-1120-tp4`**:
  - `prefill 1,000,000`: **`+1.24% ±0.63%` better** (`32K..512K`, `poisson`, `stall-3/11/15`, and `grid` geometric mean `+0.20%`: all `same`).
  - Decode showed `-0.85%..-1.22%` vs Nov 19 recorded `1120` (`c1 -0.93%`, `c8 -1.22%`, `c16 -0.85%`, `sampled c1/c8/c16 -0.88%/-1.13%/-1.06%`).
- **Same-session `spark-ds41f:1120` control (`docs/release-benchmark.md` ABBA confirmation)**:
  - `1120` itself ran `-1.65%` slower on the aged host (`c16`: `280.43 -> 275.79 tok/s`), confirming session drift rather than a candidate regression.
  - Head-to-head in the same session (`1120-ctrl -> c72r3`, both candidate boots): **`single-stream c1 +0.82% ±0.11%` better** (`86.76 -> 87.36 / 87.60 tok/s`, `TTFT paired -0.52% better`, `TPOT p50 -1.16%`), **`c16 +0.61%` faster** (`275.79 -> 277.51 / 277.42 tok/s`), and engine grid `4x24 -2.62%`, `8x24 -2.38%`, `4x12 -1.28%`, `12x72 -1.16%`.

### Two-Boot `bin/spark bench` Summary Across All Four DGX Shapes (`candidate-72-r3`, `receipts/benchmarks/`)

| Shape | Quality | `prose-c1` | `prose-c4` | `prose-c8` | `code-c1` | `code-c4` | `code-c8` | `prefill 32K` | `prefill 131K` | Min `MemAvailable` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tp4-ring` (`16.5 GiB` KV) | `LRU 5/5` | `83.8` | `276.9` | `395.7` | `104.6` | `321.4` | `530.1` | `10,185.2` | `10,453.3` | `>= 14.29 GiB` |
| `pp2-tp2-ring` (`24.0 GiB` KV) | `LRU 5/5` | `48.0` | `145.0` | `240.1` | `53.7` | `153.4` | `248.3` | `6,585.2` | `6,798.4` | `>= 10.86 GiB` |
| `tp2-direct` (`240/64` + `0866`) | `LRU 5/5` | `28.1` | `90.7` | — | `30.4` | `91.3` | — | `4,436.6` | `4,730.6` | `>= 8.52 GiB` |
| `tp1` (`96/48` + `0866`) | `LRU 5/5` | `9.2` | `29.8` | — | `8.4` | `28.2` | — | `2,079.4` (`1K: 330.9`) | — | `>= 8.57 GiB` |

*(Zero swap growth and `0.0 ms` thermal slowdown across all 8 boots. Notes: Spaceports `sp1..sp8` were unreachable during this session because the WireGuard container on `storage` was stopped; AgentX corpus `prepare` on `dgx1` previously OOM'd at `64G` on Oct 25 so AgentX could not run on `dgx1`, matching `1120` which also has no AgentX record.)*

## Revision `candidate-72-r5` (Reworked PR `#1323` Empty-Index-Slice Decode RoPE Fix)

Date: 2026-11-24.
Commits: `06eebd1314` (cherry-picked PR `#1323` diagnostic record under `experiments/2026-11-23-decode-rope-empty-index-slice/`, authored by James Meadlock), `77e6b7154b`, `13d226bea2` (on `origin/candidate-72`).

### Delta Over `candidate-72-r3`

Added reworked vLLM patch `0868` (`0868-fix-dsv41-rotate-decode-queries-on-a-rank-dealt-no.patch`, preserving James Meadlock's authorship; `741` patches total, `patch_head = 32d6ab273e89923c6fea2fa8c415f8887b1da325`, `expected_tree = 0aa091fa2b3dc5ab239816682a4b0f84e209cc58`, `patchset_sha256 = 314d65b785222c70e3401050d193551917ef134cb9f0d0cea4afae18c15f0695`):
- Extracted `DeepseekV4Attention._launch_pending_decode_aux(...)` in `vllm/models/deepseek_v4_1/attention.py` and guarded it with a single-shot `issue_pending_aux()` helper in `forward_mqa` invoked beside the first `_index_chunk` on non-empty slices, right after `select(...)` before `gather` on empty rank-split slices (so deferred `_rope(q)` + `_shared_decode_metadata` overlap with the all-gather), and before every `wait_stream(pending_aux)` join.
- Prevents skipped decode query RoPE on ranks dealt `0` index rows (e.g. `TP8` rank 7 at `25..28, 33..35, 41..42, 49` rows; eager long-context `TP3`/`TP4`), prevents duplicate RoPE launches across multi-chunk scoring or `_INDEX_SPLIT_CHECK` reference passes, and skips empty aux-stream work on headless ranks (`n_local_heads == 0`).
- Extended `tests/models/test_deepseek_v4_1_attention.py::test_decode_two_stream_overlap_schedules_disjoint_workspaces` to cover empty rank-split slices (asserting `("rope", "q", "aux")` runs before `("gather", "main")` and `("attend", "main")`), non-empty rank-split slices, multi-chunk idempotence (`DECODE_CHUNK=2` on `4` rows), and headless (`n_local_heads=0`) empty slices.
  - Note: intermediate build `candidate-72-r4` (`sha256:dd75ce7cb82b...`) caught a test-only bug where `monkeypatch.setattr(cls, "DECODE_CHUNK", 2)` was shadowed by the instance attribute `module.DECODE_CHUNK = 64` and `_can_overlap_decode`'s `rows <= self.DECODE_CHUNK` guard; fixed in `candidate-72-r5` by calling `forward_mqa` directly with `module.DECODE_CHUNK = 2` and `_pending_decode_aux` set, and superseded `candidate-72-r4` was deleted from `dgx1..dgx4`.
- Corrected the `When to use PP=2 vs TP` table in `docs/switchless-topology.md` to remove the inaccurate `TP6`/`TP8` NCCL batch-invariance claim.

### Verification (`candidate-72-r5`)

- **Image**: `spark-ds41f:candidate-72-r5` (superseded `candidate-72-r3` and `candidate-72-r4` deleted from `dgx1..dgx4`).
  - OCI digest: `sha256:cc5107cc88e14bbd0c81d3e3e8dfe58bbeb8ac8977dd02622d7d92332d15a082`
  - Labels: `local.spark-ds41f.vllm.tree=0aa091fa2b3dc5ab239816682a4b0f84e209cc58`, `local.spark-ds41f.sparknet.tree=1739407706277142da7a12c95fca3c503c8cb11c`
  - Distributed along `dgx4 -> dgx3-internal -> dgx2-internal -> dgx1-internal`; identical `56` `RootFS.Layers` (`layers_sha = 1fc69a3f3bb0af19`) verified on all 4 nodes; GPU import smoke passed.
- **Zero-overlay built-image pytest (`dgx4`, `tests/models/test_deepseek_v4_1_attention.py`)**: **`80 passed, 0 failed`** (`61.59 s`).
- **Full Linux repo unit suite (`dgx4`)**: **`929 passed, 0 failed`** (`33.80 s`).
- **`tp4-ring` strict temperature-0 determinism (`receipts/determinism/tp4-ring-r5-wide{,-nologprobs}.jsonl`)**: **`0` token diffs, `0` logprob diffs** across all 6 scenarios (`c8`, `staggered`, `mixed`, `cached`, `again`, `mixed15`), bit-for-bit identical SHA-256 outputs and logprobs to `candidate-72-r3` (`reference_sha256 = 18d619615e51...`, `long_sha256 = d65686498eb8...`, `reference_text_sha256 = 0a2dacdde6ff...`, `long_text_sha256 = 12fd825767d2...`).

## Conclusion

`candidate-72-r5` (`spark-ds41f:candidate-72-r5`, `sha256:cc5107cc88e1...`) carries the full `candidate-72-r3` Step 9 qualification plus the reworked PR `#1323` empty-index-slice decode RoPE fix (`0868`, `80/80` built-image attention tests, `929/929` Linux repo tests, `0/0` bit-identical `tp4-ring` determinism). Production `spark-ds41f:1120` is restored (`doctor --live OK`) and the cluster hold is released. Ready for release promotion.
