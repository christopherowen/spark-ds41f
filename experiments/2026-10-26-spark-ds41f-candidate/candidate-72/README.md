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

## Conclusion

`candidate-72` is verified end-to-end across unit tests (`154/154`), zero-patch `dgx-spark-networking v0.6.0`, live `GLP-27` (`alpha=2.0`, `6/6` auto `tool_choice`), opt-in buffered atomic admission (`160/160` exact parity), strict temperature-0 batch invariance (`0` token diffs, `0` logprob diffs across `tp4-ring`, `pp2-tp2-ring`, and `tp2-direct`), `24.0 GiB` KV (`30.70M` tokens) on `pp2-tp2-ring` with `>=10.97 GiB` free RAM, and chunked `0867` prefill scatter on `tp2-direct` with `>=8.55 GiB` free RAM. Ready for promotion review.
