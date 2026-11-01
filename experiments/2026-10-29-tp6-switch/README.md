# Switched TP3..TP8: rank-order sum, mHC, GLP steering, and MoE padding up to eight ranks

## What this changes

Five vLLM commits on top of candidate 24's series (`rc24/vllm/series`, `0625`–`0629`), together with switched-topology and collective-probe support for `TP3..TP8`:

- **`0625` — TileLang `rank_order_sum` for 2..8 ranks** (`vllm/models/deepseek_v4_1/tilelang/collectives.py`, `tests/kernels/test_deepseek_v41_tilelang_collectives.py`).
  - Raises `MAX_RANK_ORDER_WORLD` to `8`, bounding `rank_order_sum_kernel`, `rank_order_sum_calls`, and `rank_order_sum` together so the kernel launched in the forward pass is always declared during warmup.
  - Unrolls `sum_rows` over a tuple of distinct `T.Tensor` parameter annotations per source rank (`s0 .. s{world-1}`). Source `0` is cast directly to `float32` (preserving `-0.0`); sources `1 .. world - 1` accumulate in `float32` in rank order and round once to `bfloat16`.
  - `world in (2, 3, 4)` remain `structural_equal` to `rc24`.
- **`0626` — TileLang mHC `post_pre` and GLP steering for 2..8 ranks** (`vllm/models/deepseek_v4_1/tilelang/mhc.py`, `vllm/models/deepseek_v4_1/tilelang/glp.py`, `tests/kernels/test_deepseek_v41_tilelang_mhc.py`).
  - Updates `project_streams`, `project_and_finalize_streams` (introduced in `0606` for single-row fused entry), and `glp_steer_kernel` to assert `1 <= world <= MAX_RANK_ORDER_WORLD`.
- **`0627` — Sequence-parallel row exchange and all-to-all tests at 3..8 ranks** (`tests/models/test_deepseek_v41_sp_project_reduce_scatter.py`).
  - Extends `test_row_exchange_gives_each_rank_its_rows_of_every_block` and `test_row_all_to_all_over_cabled_pairs` across `world in [3, 4, 5, 6, 7, 8]`.
- **`0628` — `moe_intermediate_size` padding and TP-padded expert scale codes for TP5, TP7, TP8** (`vllm/model_executor/models/config.py`, `vllm/models/deepseek_v4_1/nvidia/model.py`, `vllm/models/deepseek_v4_1/tilelang/scale_codes.py`, `tests/models/test_deepseek_v4_1_config.py`, `tests/kernels/quantization/test_deepseek_v41_expert_scale_codes.py`).
  - `DeepSeek-V4.1-Flash` has `moe_intermediate_size = 2304`, which divides into 64-element expert GEMM tiles at `TP3` (`768`), `TP4` (`576`), and `TP6` (`384`), but not at `TP5`, `TP7`, or `TP8` (`2304 // 8 = 288`, which is not a multiple of 64).
  - `DeepseekV41ForCausalLMConfig.update_model_config_for_parallelism` now pads `moe_intermediate_size` to `round_up(original_moe_intermediate, tp_size * 64)` (`2304 -> 2560` at TP5 and TP8, `2304 -> 2688` at TP7; unchanged at TP3, TP4, TP6), marks `shared_experts` with `allow_tp_padding=True`, and updates `_row_deltas` in `scale_codes.py` so trailing zero-padded `w2_weight_scale` columns expand to `base` (multiplying zero `w2_weight` columns) without widening the row's UE8M0 exponent span.
- **`0629` — Batch strided `--load-format direct` reads into column-padded 2D destinations** (`vllm/model_executor/model_loader/direct_io/_batch.c`, `tests/model_executor/model_loader/test_direct_loader_cuda.py`).
  - When `moe_intermediate_size` is padded (`TP5` rank 4, `TP7` rank 6, and `TP8` all ranks), `RoutedExperts._load_w2` narrows `w2_weight` and `w2_weight_scale` along the column dimension (`destination[:, :narrow_size]`), so `destination_stride > bytes * (1 + expand_bf16)`. Previously, `_batch.c` only allowed `chunk_rows > 1` when `destination_stride == bytes * (1 + expand_bf16)`, splitting every 2560-row `w2.weight` (`144` bytes/row into stride `160` at `TP8`) and `w2.weight_scale` (`9` bytes/row into stride `10` at `TP8`) into `2560` single-row jobs (`~112.7M` 4 KiB `O_DIRECT` reads = `684.55 GB` read from disk and `1540.5 s` target weight load on `TP8`).
  - `_batch.c` now batches `chunk_rows > 1` whenever `source_stride >= bytes` and `destination_stride >= bytes * (1 + expand_bf16)`, and checks 2D strided row intervals when destination bounding boxes overlap so disjoint column slices batch cleanly while overlapping 2D slices are still rejected before writing any destination bytes.
  - On `spaceport` (`sp1`–`sp8`, `tp8-switched`), this reduces `TP8` target weight loading from `1540.48 s` (`684.55 GB` read from disk) to `18.61 s` (`125.55 GB` read from disk, `2.10 GB/s`), and total model loading from `1550.3 s` to `28.44 s` (`54.5x` speedup).

### Per-rank geometry across `TP3..TP8`

| Shape | Attention (`heads`, `o_groups`) | Per-rank (`heads`, `o_groups`) | `moe_intermediate_size` | Per-rank MoE intermediate | Engram (`24` heads × `128`) |
|---|---|---|---|---|---|
| **TP3** | `64 -> 72`, `8 -> 9` | `24, 3` | `2304` | `768` (`6 × 128`) | `8` heads (`1024` cols/rank) |
| **TP4** | `64`, `8` | `16, 2` | `2304` | `576` (`4 × 128 + 64`) | `6` heads (`768` cols/rank) |
| **TP5** | `64 -> 80`, `8 -> 10` | `16, 2` | `2304 -> 2560` | `512` (`4 × 128`) | `640` cols/rank (padded to `3200`) |
| **TP6** | `64 -> 96`, `8 -> 12` | `16, 2` | `2304` | `384` (`3 × 128`) | `4` heads (`512` cols/rank) |
| **TP7** | `64 -> 112`, `8 -> 14` | `16, 2` | `2304 -> 2688` | `384` (`3 × 128`) | `448` cols/rank (padded to `3136`) |
| **TP8** | `64`, `8` | `8, 1` | `2304 -> 2560` | `320` (`2 × 128 + 64`) | `3` heads (`384` cols/rank) |

## Evidence

### 1. Single-node GPU kernel and model test verification (`dgx4`)

All 541 tests across the modified kernel, model, config, and PDL suites were executed on `dgx4` against `spark-ds41f:19f2c20ed4d6-next-24` with `0625`–`0628` mounted (`540 passed, 1 skipped, 0 errors`; see [`dgx4-pytest-receipt.txt`](dgx4-pytest-receipt.txt)):
- `tests/kernels/test_deepseek_v41_tilelang_collectives.py` (all world sizes `2..8` + 9-part fallback + CUDA graph replay)
- `tests/kernels/test_deepseek_v41_tilelang_mhc.py` (`project_streams`, `project_and_finalize_streams`, and `post_streams` with `RankOrderParts` across `world in 2..8`)
- `tests/models/test_deepseek_v41_sp_project_reduce_scatter.py` (including `test_moe_combine_writes_the_exchange_slots`, row exchange, and all-to-all across `3..8` ranks)
- `tests/models/test_deepseek_v4_1_config.py` (`TP3..TP8` head and `moe_intermediate_size` padding)
- `tests/kernels/quantization/test_deepseek_v41_expert_scale_codes.py` (`TP5`, `TP7`, `TP8` zero-padded `w2_weight_scale` columns)
- `tests/kernels/test_deepseek_v41_tilelang_pdl.py` (PDL source audit and layer-chain bit-equality)
- `tests/models/test_deepseek_v4_1_glp.py`

### 2. Six-node switched hardware measurements (`vedcsolution`, TP6 on `rc10`)

Measured on 6 Sparks behind one switch (`sparknet` 0.5.0 `direct`):

| | Measured |
|---|---|
| Decode, closed loop (60 s warm + 240 s, 10 windows), 16 seats | aggregate: c1 83.7 · c8 234.6 · c16 305.9 tok/s (c1 is also per request) |
| Prefill, real text (lab boot) | 32K 7,391 (±7 %) · 131K 7,648 · 262K 7,383 · 524K 6,845 tok/s; 1M was skipped by servebench (prompt = max_model_len) |
| Determinism, `token_determinism_text.py --wide` | `identical: true` with and without logprobs (text digests `7fde24b3` and `12fd8257`, the same in both runs) |
| Against a TP4 (1025) corpus, `servebench model --limit 64 --top 20` | `servebench compare`: KL mean 0.0239 nats (p99 0.295), top-1 94.65%, acceptance length +0.3% (2.259) |
| tool-eval-bench 3.0.1.dev6, 104 scenarios | final score 90/100 (89 pass, 10 partial, 5 fail) |
| Weights per rank (model loading) | 48.9 GiB (TP4 ≈ 72) |
| Overnight serving (8 h, real traffic) | ~280-293 tok/s hourly mean at 16-25 concurrent requests; no 5xx and no restart |

**Not paired.** Candidate-10 TP4 ring numbers (c1 77.50; c8 24.32 per stream ≈ 195 aggregate) come from another fabric and another session. TP6 and TP4 cannot match bit for bit because they reduce different partial sums.

## Tooling and qualification status for Spaceport `TP3..TP8`

- `scripts/probe_collectives.py` now accepts `--world-size` in `{2, 3, 4, 5, 6, 7, 8}` and supports `--numerics` across all 2..8 ranks.
- `scripts/topology.py` now accepts switched node maps (`oneshot-switched`, `nccl-switched`) of `3..8` nodes while keeping switchless topologies (`oneshot-direct`, `oneshot-ring4`, `nccl-ring`) at `3` or `4` nodes.
- Before relying on any `TP3..TP8` switched deployment in Spaceport, run `bin/spark topology probe --port-samples --numerics` across all participating ranks to qualify the switched fabric and `sparknet`'s multi-group all-gather above 4 ranks.
