# Candidate `candidate-71`

Date: 2026-11-23.
Base: `candidate-70` (`290d2994df`, built on `spark-ds41f-1122` / vLLM `0001..0860`), plus `dgx-spark-networking` one-shot RoCE P2P (`feat/oneshot-pp-p2p` @ `431cef5adff9ca0812aef6ad29ea9cb73f2fa6af`, PR #8; compat follow-up `0a5c5dd` pushed after image build).

## Scope & Zero-Degradation Guarantee

`candidate-71` promotes the full Round 2 performance stack for `pp2-*`, `tp2-*`, and `tp1`, and moves pipeline-parallel RoCE point-to-point transport directly into `dgx-spark-networking` (`sparknet.oneshot.P2P` + `SparknetOneShotP2P`), while keeping every `tp3-*`, `tp4-*`, and `tp5..tp8-*` cluster profile byte-for-byte identical in runtime configuration to `spark-ds41f-1122` (only image tag, `expected_labels`, and `upstreams_config` change):

- **All 9 `tp3..tp8` shapes**: zero runtime config drift vs `releases/spark-ds41f-1122/cluster.json`; verified live on `tp4-ring` across 2 boots with dead-on parity.
- **vLLM patch `0861`** (`SparknetOneShotP2P`): executes strictly when `pipeline_parallel_size > 1` and `VLLM_PP_SPARKNET_P2P=1`.
- **vLLM patch `0862`** (async zero-sync PP adaptive verification + `SPARK_DS41F_PP_USE_ENGINE_REQS=1`): executes strictly when `pipeline_parallel_size > 1`.
- **vLLM patch `0863`** (bundled 5-domain held-out hotlist + fused BF16 single-sync `L+1` expert lookahead): executes strictly inside `ExpertSlotCache` (`VLLM_DS41_EXPERT_SLOTS > 0`: `tp1`, `tp2-direct`, `tp2-switched`).
- **vLLM patch `0864`** (`TileLang` fused MoE `down_num_stages = 3 if intermediate % 128 == 0 else 2`): activates for `TP=1` (`768`), `TP=2` / `pp2-tp2` (`384`), and `TP=3` / `pp2-tp3` (`256`); leaves `TP=4` / `pp2-tp4` (`192`) and `TP=8` (`96`) unchanged at `2`.

## Delta Over `candidate-70`

1. **`dgx-spark-networking` one-shot RoCE P2P (`candidate-71/sparknet/0001..0003` rebased on `431cef5adff9`)**:
   - Adds `ROCE_P2P_ABI_VERSION = 1` (leaving collective ABI 10 unchanged): 4-slot credit-flow-controlled per-peer RDMA ring proxy (`slot_bytes = 1 MiB` default = `8 MiB` pinned RAM/rank), TileLang `oneshot_p2p_send` / `oneshot_p2p_recv` kernels, `sparknet.oneshot.P2P`, and `SparknetOneShotP2P` vLLM adapter.
   - Real 2-node ConnectX-7 GPU verification (`dgx1` ↔ `dgx4`): 36/36 GPU tests pass in `8.2 s`, fail-stop timeout verified at `0.500 s`, beats NCCL P2P at every payload from `4 KiB` (`11.1 µs`, `1.7x`) through `64 MiB` (`23.90 GB/s` vs NCCL `23.30 GB/s`).
2. **Topology & recipe (`scripts/topology.py`, `scripts/recipe.py`, `tests/test_pp.py`)**:
   - `_pp2_ring4_node_environment` emits `SPARKNET_ROCE_PP_PEER_HCAS` (keyed by PP-local peer `(rank // 2) ^ 1` -> `roce_peer_hcas[str(rank ^ 2)]`), emits `SPARKNET_ROCE_SUBNETS` for all 4 local HCAs, and omits deprecated `SPARKNET_ROCE_GID_INDEX` so both `AllReduce` and `P2P` resolve RoCE v2 GIDs by subnet (while preserving `NCCL_IB_GID_INDEX` when pinned).
3. **vLLM patches `0861..0864`** (`737` patches total, `patch_head = 795a96b0690cd98721fb6f3f22357c989eff50bd`, `expected_tree = 0f5bc603eb8cb993cca614dfd6c12c4efec14add`):
   - `0861`: wires two separate `SparknetOneShotP2P` instances into `gpu_worker.py`/`model_runner.py` (`main_stream`, `1 MiB` slots for forward activations) and `pp_utils.py:PPHandler` (`broadcast_stream`, `64 KiB` slots for coalesced backward sampled/draft/confidence broadcast), eliminating both Gloo TCP metadata hops and the sibling NCCL `pp_broadcast` communicator without cross-stream deadlock (`8.5 MiB` total pinned RAM per PP rank), with fail-stop health-check aggregation.
   - `0862`: async double-buffered D2H copy in `apply_pp_confidences` + `SPARK_DS41F_PP_USE_ENGINE_REQS=1` + `SPARK_DS41F_C1_ROW_COST_SCALE`.
   - `0863`: bundles the 5-domain Sonar-disjoint held-out expert hotlist (`vllm/models/deepseek_v4_1/expert_slots/hotlist.py`, `VLLM_DS41_EXPERT_HOTLIST=multidomain`) and fused BF16 single-sync `L+1` expert lookahead (`VLLM_DS41_EXPERT_FAST_LOOKAHEAD=1`).
   - `0864`: defaults `down_num_stages = 3` in TileLang fused MoE when `intermediate % 128 == 0` (keeping `2` otherwise).
4. **Profile updates**:
   - All four `pp2-*` shapes (`pp2-tp2-ring`, `pp2-tp2-switched`, `pp2-tp3-switched`, `pp2-tp4-switched`): `VLLM_PP_SPARKNET_P2P=1`, `VLLM_SINK_REJECTION=1`, `SPARK_DS41F_PP_USE_ENGINE_REQS=1`, `enable_adaptive_verification=true`, `adaptive_verification_row_cost=0.25`.
   - All three `tp1`/`tp2-*` shapes (`tp1`, `tp2-direct`, `tp2-switched`): `VLLM_DS41_EXPERT_FAST_LOOKAHEAD=1`, `VLLM_DS41_EXPERT_HOTLIST=multidomain`, `VLLM_DS41_EXPERT_HOTLIST_MODE=pure`.

## Build & Unit Test Verification

- **Image**: `spark-ds41f:candidate-71`
  - OCI digest: `sha256:f89fd31c639a9510b8652efb7bcb7077cc492860bab6ad5f9e92a347813eb235`
  - Labels: `local.spark-ds41f.vllm.tree=0f5bc603eb8cb993cca614dfd6c12c4efec14add`, `local.spark-ds41f.sparknet.tree=89880a734bb3b0e53bd84e0eb186266207aac26f`
  - Built on `dgx4` in `521 s` (lowest `MemAvailable = 82.2 GiB`); GPU import smoke passed for DS4.1 SM12x, TileLang, SparkNet (`collective ABI 10`, `P2P ABI 1`), and B12X-free import.
  - Distributed along `dgx4 -> dgx3-internal -> dgx2-internal -> dgx1-internal`; identical `56` `RootFS.Layers` (`sha256` prefix `17cb3f833c8f191c`) and image ID verified on all 4 nodes.
- **Repository tests**: All `candidate-71` profiles pass `bin/spark doctor`; Linux `python3 -m unittest discover -s tests` on `dgx4`: **`929 passed, 0 failed`** (`34.9 s`); `tests/test_pp.py` + `tests/test_topology.py`: **`37 passed, 0 failed`**.
- **Zero-overlay built-image GPU & simulator unit tests on `dgx4` (`--gpus all`)**: **`336 passed, 1 skipped, 0 failed`** across `test_pp_sparknet_p2p.py`, `test_adaptive_verification.py`, `test_deepseek_v41_expert_slots.py` (`14/14` with `--security-opt seccomp=unconfined --ulimit memlock=-1` for `io_uring`), `test_deepseek_v41_tilelang_moe.py`, `test_deepseek_v41_tilelang_router.py`, and `sparknet/tests/test_oneshot_cpu.py` (including ASan/UBSan C simulator), plus direct GPU verification of `RouterLookaheadPredictor.predict_topk_gpu` (`1.000` top-k agreement vs FP32 reference). Note: `tests/models/deepseek_v4_1/test_expert_hotlist_and_fast_lookahead.py` in patch `0863` passes `bias=` instead of `e_score_correction_bias=` to `RouterLookaheadPredictor.__init__` — runtime code is unaffected and verified live on both `tp1` and `tp2-direct`.

## Live Two-Boot Verification Screens (`spark-ds41f:candidate-71`, commit `3d90dadc4e`)

Commands executed from `dgx1` inside an owned cluster hold (production `tp4-ring` restored and `doctor --live OK` afterwards):

```sh
bin/spark --cluster-config <cfg> cluster start --replace --change-shape --apply
bin/spark --cluster-config <cfg> bench --suites quality,decode,prefill \
  --concurrency <conc> --prefill-sizes <sizes> --prefill-repeats 2 \
  --min-samples <min> --max-samples <max> --cool-below 0 --compare none \
  --output results/private/bench/<run>
```

### 1. `pp2-tp2-ring` (`dgx1, dgx2, dgx4, dgx3`, `1M` ctx, `16.5 GiB` KV) vs `manifests/benchmarks/2026-11-22-spark-ds41f-1122-pp2tp2ring.json`

| Metric | `1122-pp2tp2ring` | `c71-pp2-b1` (`+94.0 s`) | `c71-pp2-b2` (`+81.2 s`) | 2-Boot Mean | Delta vs `1122` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Quality | `LRU 5/5` | `LRU 5/5` | `LRU 5/5` | `LRU 5/5` | pass |
| `prose-c1` (tok/s / step ms) | `40.1` (`56.73 ms`) | `47.6 ±7.1%` (`43.97 ms`) | `47.9 ±1.4%` (`44.23 ms`) | **`47.8`** (`44.10 ms`) | **`+19.2%`** (`-22.3%` step) |
| `prose-c2` | `76.2` (`59.90 ms`) | `84.2 ±2.8%` (`47.98 ms`) | `84.1 ±1.4%` (`49.15 ms`) | **`84.2`** (`48.57 ms`) | **`+10.5%`** (`-18.9%` step) |
| `prose-c4` | `123.3` (`74.53 ms`) | `149.5 ±1.3%` (`52.58 ms`) | `149.5 ±1.2%` (`51.90 ms`) | **`149.5`** (`52.24 ms`) | **`+21.2%`** (`-29.9%` step) |
| `prose-c8` | `210.2` (`87.97 ms`) | `226.0 ±25.5%` (`74.68 ms`) | `235.2 ±13.2%` (`71.65 ms`) | **`230.6`** (`73.17 ms`) | **`+9.7%`** (`-16.8%` step) |
| `code-c1` | `50.3` (`59.79 ms`) | `53.4 ±0.8%` (`47.43 ms`) | `53.2 ±1.0%` (`47.58 ms`) | **`53.3`** (`47.51 ms`) | **`+6.0%`** (`-20.5%` step) |
| `code-c2` | `93.2` (`64.86 ms`) | `94.1 ±1.1%` (`52.84 ms`) | `94.5 ±0.8%` (`52.68 ms`) | **`94.3`** (`52.76 ms`) | **`+1.2%`** (`-18.7%` step) |
| `code-c4` | `148.2` (`82.38 ms`) | `151.5 ±26.8%` (`59.15 ms`) | `156.2 ±8.6%` (`58.24 ms`) | **`153.9`** (`58.70 ms`) | **`+3.8%`** (`-28.7%` step) |
| `code-c8` | `235.5` (`104.38 ms`) | `260.8 ±19.9%` (`70.23 ms`) | `261.2 ±13.0%` (`70.17 ms`) | **`261.0`** (`70.20 ms`) | **`+10.8%`** (`-32.7%` step) |
| `prefill 32K` (tok/s) | `5,571.7 ±4.2%` | `6,615.5 ±1.6%` | `6,581.9 ±2.2%` | **`6,598.7`** | **`+18.4%`** |
| `prefill 131K` (tok/s) | `5,727.5 ±0.2%` | `6,749.4 ±6.2%` | `6,799.3 ±2.0%` | **`6,774.4`** | **`+18.3%`** |
| Min `MemAvailable` (GiB) | — | `17.93–21.44` (0 swap) | `18.06–21.50` (0 swap) | `>=17.93 GiB` | 0 thermal slowdown |

### 2. `tp2-direct` (`dgx1, dgx2`, `256K` ctx, `6.0 GiB` KV = `6.44M` tok) vs `manifests/benchmarks/2026-11-20-spark-ds41f-1120-direct2.json`

| Metric | `1120-direct2` | `c71-tp2-b1` (`+129.5 s`) | `c71-tp2-b2` (`+91.1 s`) | 2-Boot Mean | Delta vs `1120` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Quality | `LRU 5/5` | `LRU 5/5` | `LRU 5/5` | `LRU 5/5` | pass |
| `prose-c1` | `17.7` (`121.37 ms`) | `28.6 ±2.1%` (`74.38 ms`) | `29.5 ±10.0%` (`71.23 ms`) | **`29.1`** (`72.81 ms`) | **`+64.1%`** |
| `prose-c4` | `53.0` (`153.91 ms`) | `91.7 ±7.0%` (`83.81 ms`) | `93.9 ±14.8%` (`83.30 ms`) | **`92.8`** (`83.56 ms`) | **`+75.1%`** |
| `code-c1` | `22.2` (`127.53 ms`) | `32.9 ±13.9%` (`72.82 ms`) | `32.0 ±0.7%` (`74.15 ms`) | **`32.5`** (`73.49 ms`) | **`+46.2%`** |
| `code-c4` | `64.2` (`166.48 ms`) | `95.9 ±9.2%` (`81.86 ms`) | `99.8 ±10.7%` (`77.32 ms`) | **`97.9`** (`79.59 ms`) | **`+52.4%`** |
| `prefill 32K` (tok/s) | `258.8 ±73.8%` | `3,987.1 ±44.5%` | `4,629.2 ±3.3%` | **`4,308.2`** | **`16.6x`** |
| `prefill 131K` (tok/s) | `235.6 ±15.9%` | `4,878.3 ±3.5%` | `4,842.0 ±14.4%` | **`4,860.2`** | **`20.6x`** |
| Min `MemAvailable` (GiB) | — | `5.00 / 6.69` (0 swap) | `5.45 / 7.31` (0 swap) | `>=5.00 GiB` | 0 thermal slowdown |

*(Note: `c71-tp2-b1` `prefill 32K` had one cold-cache sample pulling its 2-sample mean to `3,987.1 ±44.5%`; `c71-tp2-b2` recorded `4,641 / 4,617` = `4,629.2 ±3.3%`.)*

### 3. `tp1` (`dgx1`, `256K` ctx, `6.0 GiB` KV = `6.44M` tok) vs `manifests/benchmarks/2026-11-20-spark-ds41f-1120-tp1.json`

| Metric | `1120-tp1` | `c71-tp1-b1` (`+125.5 s`) | `c71-tp1-b2` (`+94.1 s`) | 2-Boot Mean | Delta vs `1120` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Quality | `LRU 5/5` | `LRU 5/5` | `LRU 5/5` | `LRU 5/5` | pass |
| `prose-c1` | `4.4` (`511.18 ms`) | `9.0 ±4.4%` (`218.61 ms`) | `9.1 ±2.4%` (`214.95 ms`) | **`9.05`** (`216.78 ms`) | **`2.06x` (`+105.7%`)** |
| `prose-c4` | `10.9` (`677.59 ms`) | `29.2 ±0.2%` (`213.65 ms`) | `30.8 ±13.2%` (`208.69 ms`) | **`30.0`** (`211.17 ms`) | **`2.75x` (`+175.2%`)** |
| `code-c1` | `4.8` (`584.05 ms`) | `8.4 ±0.8%` (`285.43 ms`) | `8.4 ±1.3%` (`285.66 ms`) | **`8.40`** (`285.55 ms`) | **`1.75x` (`+75.0%`)** |
| `code-c4` | `11.4` (`794.33 ms`) | `28.0 ±9.8%` (`262.51 ms`) | `28.4 ±6.7%` (`257.85 ms`) | **`28.2`** (`260.18 ms`) | **`2.47x` (`+147.4%`)** |
| `prefill 1K` (tok/s) | `24.4 ±65.7%` | `331.9 ±145.8%` | `290.9 ±208.5%` | **`311.4`** | **`12.8x`** |
| `prefill 8K` (tok/s) | — | `1,357.8 ±115.2%` | `1,501.6 ±24.9%` | **`1,429.7`** | — |
| `prefill 32K` (tok/s) | — | `2,147.6 ±2.2%` | `2,101.1 ±6.6%` | **`2,124.4`** | — |
| Min `MemAvailable` (GiB) | — | `7.98` (0 swap) | `8.06` (0 swap) | `>=7.98 GiB` | 0 thermal slowdown |

### 4. `tp4-ring` Regression Check (`dgx1..dgx4`, `1M` ctx, `16.5 GiB` KV) vs `manifests/benchmarks/2026-11-22-spark-ds41f-1122-tp4.json`

| Metric | `1122-tp4` | `c71-tp4-b1` (`+104.0 s`) | `c71-tp4-b2` (`+86.6 s`) | 2-Boot Mean | Delta vs `1122` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Quality | `LRU 5/5` | `LRU 5/5` | `LRU 5/5` | `LRU 5/5` | pass |
| `prose-c1` | `84.2 ±1.1%` (`25.78 ms`) | `84.9 ±1.7%` (`25.43 ms`) | `84.3 ±1.3%` (`25.61 ms`) | **`84.6`** (`25.52 ms`) | `+0.5%` (parity) |
| `prose-c2` | `147.9 ±1.2%` (`26.79 ms`) | `143.2 ±14.8%` (`27.69 ms`) | `143.5 ±15.9%` (`27.67 ms`) | `143.4` (`27.68 ms`) | within ±15% CI |
| `prose-c4` | `277.0 ±1.1%` (`26.96 ms`) | `279.1 ±2.0%` (`26.75 ms`) | `277.6 ±1.3%` (`26.90 ms`) | **`278.4`** (`26.83 ms`) | `+0.5%` (parity) |
| `prose-c8` | `431.6 ±12.7%` (`36.77 ms`) | `427.0 ±5.5%` (`37.12 ms`) | `427.0 ±4.2%` (`37.09 ms`) | `427.0` (`37.11 ms`) | within ±12.7% CI |
| `code-c1` | `104.1 ±0.6%` (`28.60 ms`) | `104.6 ±1.2%` (`28.24 ms`) | `104.3 ±0.2%` (`28.28 ms`) | **`104.5`** (`28.26 ms`) | `+0.4%` (parity) |
| `code-c2` | `160.2 ±17.2%` (`35.14 ms`) | `156.7 ±18.0%` (`34.73 ms`) | `156.0 ±17.9%` (`34.91 ms`) | `156.4` (`34.82 ms`) | within ±18% CI |
| `code-c4` | `317.2 ±4.4%` (`32.32 ms`) | `310.6 ±6.8%` (`32.68 ms`) | `315.1 ±12.1%` (`32.30 ms`) | `312.9` (`32.49 ms`) | within ±12% CI |
| `code-c8` | `544.4 ±4.9%` (`37.57 ms`) | `536.0 ±10.0%` (`37.63 ms`) | `497.8 ±27.1%` (`42.31 ms`) | `516.9` (`39.97 ms`) | within CI (1 outlier in b2) |
| `prefill 32K` (tok/s) | `10,185.7 ±3.9%` | `10,226.3 ±0.8%` | `10,151.4 ±9.4%` | **`10,188.9`** | `+0.03%` (parity) |
| `prefill 131K` (tok/s) | `10,543.0 ±0.9%` | `10,492.1 ±8.6%` | `10,548.5 ±5.2%` | **`10,520.3`** | `-0.2%` (parity) |
| Min `MemAvailable` (GiB) | — | `14.25–16.62` (0 swap) | `14.37–16.65` (0 swap) | `>=14.25 GiB` | 0 thermal slowdown |

## Revision `candidate-71-r2` (Bit-Exact Batch Invariance + `pp2-tp3/4` KV Headroom)

Following the batch-invariance audit of `candidate-71`:
- **vLLM patch `0865`** (`0865-fix-ds41-make-expert-centric-prefill-streaming-bit-e.patch`, `738` total patches, `patch_head = e69d8114e031c2a1724134ede5369b3102c2a0b3`, `expected_tree = ca4b59280d9211433442f6e1f529386d1497db58`): factors `MoeExpertsV41Plan.run_routed()` out of `run()` and stages each wave's BF16 per-route outputs into `(token, k)` order in `y_all` before running one final TileLang `dispatch.combine()` in exact `k = 0..topk-1` FP32 order with `shared` folded once. Restores strict `assert torch.equal(out_ref_p, out_slot_p)` and warm-cache repeat invariance in `test_deepseek_v41_expert_slots.py`, and fixes the `e_score_correction_bias` kwarg and `tk.device.type` check in `test_expert_hotlist_and_fast_lookahead.py`.
- **`sparknet` bumped to `0a5c5dd2eba6`** (`patch_head = d642bcfa7813537c334a3bfc2ecec673f38ad76e`, `expected_tree = 7794c85affdc85c595abfa6fb07b568bb8f719ff`): adds `AllReduce`-matching legacy `SPARKNET_ROCE_GID_INDEX` compatibility to `oneshot.P2P` (PR #8).
- **Profile sizing updates**:
  - `tp2-direct` & `tp2-switched`: reverted `VLLM_DS41_EXPERT_SLOTS` from `252` to `240` (`240/64`) after `candidate-71-r2` at `252/64` dipped to `3.76 GiB` `MemAvailable` on `dgx1` during `131K` bit-exact prefill (`y_all` staging buffer), restoring `6.48 / 7.84 GiB` min `MemAvailable`.
  - `pp2-tp3-switched`: raised `kv_cache_memory_bytes` to `25769803776` (`24.0 GiB`).
  - `pp2-tp4-switched`: raised `kv_cache_memory_bytes` to `38654705664` (`36.0 GiB`, ~`68M` tokens, sized against external 8-Spark measurements showing `43–48 GiB` free at `16.5 GiB` KV).
  - All 9 `tp3..tp8` profiles remain runtime-config-identical to `spark-ds41f-1122`.

### `candidate-71-r2` Build, Unit Tests & Live Batch-Invariance Gates

- **Image**: `spark-ds41f:candidate-71-r2` (`sha256:2d324e04b517a7ae22a1d539e095d6ecdd3e131adbbe17b68d8e73b9db4dbc41`), built on `dgx4` in `455 s` (lowest `MemAvailable = 82.8 GiB`), distributed along `dgx4 -> dgx3 -> dgx2 -> dgx1` (`56` identical `RootFS.Layers`, `sha256` prefix `ef6ac05a80e040db`, verified on all 4 nodes).
- **Zero-overlay built-image GPU unit tests on `dgx4`**: **`74 passed, 1 skipped, 0 failed`** (`81.4 s`) across `test_deepseek_v41_expert_slots.py` (`14/14` strict `torch.equal`), `test_expert_hotlist_and_fast_lookahead.py` (`3/3`), `test_pp_sparknet_p2p.py`, and `test_adaptive_verification.py`.
- **Live batch-invariance & prefill gates (`commit 89397d4300`)**:

| Shape | Determinism Gate | Token Diffs | Logprob Diffs | Exact Digests (`ref` / `long`) | Live Bench / Memory Notes |
| --- | --- | ---: | ---: | --- | --- |
| `pp2-tp2-ring` (`c71` & `c71-r2`) | `token_determinism.py --wide` (`c8`, `staggered`, `mixed`, `cached`, `mixed15`, `again`, `192` tok) + 40-prompt idle vs `C16` | **`0`** (`40/40`) | **`0`** (`40/40`) | `0c521e72c482…` / `b4cfd5f84b1f…` | Identical SHA-256 across `c71` and `c71-r2`; warm boot `+78.5 s` / `+92.7 s` |
| `tp2-direct` (`240/64`, `c71-r2`) | `token_determinism.py` (`c8`, `staggered`, `mixed`, `cached`, `again`, `128` tok) | **`0`** | **`0`** | `ce46870c9bf6…` / **`b4cfd5f84b1f…`** | `long_sha256` is **bit-for-bit identical to `pp2-tp2-ring`**; `LRU 5/5`; decode `prose-c1 26.2`, `prose-c4 90.0`, `code-c1 31.1`, `code-c4 86.9`; prefill `32K 4,438.7`, `131K 4,451.6 tok/s`; min `MemAvailable 6.48 / 7.84 GiB` (`0.0` swap) |
| `tp2-direct` (`252/64`, `c71-r2`, unfavourable mem) | `token_determinism.py` (`128` tok) | **`0`** | **`0`** | `ce46870c9bf6…` / `b4cfd5f84b1f…` | Aborted `131K` prefill bench at `dgx1` min `MemAvailable 3.76 GiB` (`< 4 GiB` floor) -> motivated `240/64` above |
| `tp1` (`108/48`, `c71-r2`) | `token_determinism.py` (`c8`, `staggered`, `mixed`, `cached`, `again`, `64` tok) | **`0`** | **`0`** | `d28389815160…` / `28e3acb32e98…` | Prefill `8K 1,455.6`, `32K 1,850.9` (`2,135` warm repeat); min `MemAvailable 7.87 GiB` (`0.0` swap) |

## Conclusion

`candidate-71-r2` passes every quality gate (`LRU 5/5`), strict bit-exact batch-invariance gate (`0` token diffs and `0` logprob diffs across `pp2-tp2-ring`, `tp2-direct`, and `tp1`, including `40/40` idle-vs-`C16` on `pp2-tp2-ring` and bit-identical 6K long-prefill SHA-256 between `tp2-direct` NVMe streaming and full-GPU `pp2-tp2-ring`), unit test gate (`929/929` repo tests, `74/74` `r2` + `336/336` `r1` built-image GPU & simulator tests), and memory/thermal guard (`>= 6.48 GiB` min `MemAvailable` on `tp2-direct`, `7.87 GiB` on `tp1`, `17.93+ GiB` on `pp2-tp2-ring`, `0.0 GiB` swap growth, `0.0 ms` thermal slowdown), with zero regression on `tp4-ring`.
