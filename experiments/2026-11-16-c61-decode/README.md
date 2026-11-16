# 2026-11-16-c61-decode (`candidate-61`)

- **Base commit**: `ac768604d` (`c60-prefill` / `candidate-60`, `origin/main`)
- **Branch**: `c61-decode` (vLLM working branch `c61-decode` on top of `c60`'s `0001..0816`, tree `afcaac3594acdea245d9543dbbea2376104470ba`)
- **Primary Goal**: Beat **`100.0 tok/s`** single-stream (`prose-c1`) decode throughput on `tp4-ring` (`dgx1..dgx4`) while ensuring zero quality, determinism, memory-headroom, or multi-stream (`c8`/`c16`, `tp4-switched`, `tp3-switched`) regressions.
- **Baseline reference (`c58-final` / `c60` on `tp4-ring` and `tp4-switched`)**:
  - `tp4-ring` (`c58-final`): `prose-c1 = 79.93 tok/s` (`step_ms = 26.52 ms`), `c1-decisions-tp4 = 83.29 tok/s` (`35.88 steps/s`, `2.330 tok/step`), `code-c1 = 98.57 tok/s`.
  - `tp4-switched` (`c60`): `prose-c1 = 79.23 tok/s` (`step_ms = 26.73 ms`), `c1-decisions-tp4-switched = 82.91 tok/s` (`35.74 steps/s`, `2.328 tok/step`), `code-c1 = 98.23 tok/s`.

---

## Ranked Backlog of Open Decode & Spec-Decode Items (`TODO.md` + Open PRs + Unmerged Branches)

Every open decode item in `TODO.md`, open GitHub PR, and unmerged local decode branch is ranked below in descending order of likelihood and expected magnitude to improve **single-stream (`prose-c1`) decode TPS** on `tp4-ring` without regressing quality, determinism, or multi-stream (`c8`/`c16`) performance. We work through them strictly in this order.

### Tier 1 — High-Likelihood Single-Stream (`c1`) Winners

1. **Rank 1 — PR `#1003` (`perf/dspark-sunk-cost-and-early-halt`) + `TODO.md` Spec-Decode ("Marginal verification rule")**
   - **Expected `c1` impact**: **`+6.2%` steady-state (`+9.6%` cold-100)** measured on `TP=4`, `KL = 0`, `+1%` at `c8`/`c16`.
   - **Mechanism**: Prices single-tile (`<= 16` rows) sunk DSpark draft cost (`DRAFT_COST_RATIO = 0.55`) and within-tile marginal row slope (`SINGLE_TILE_ROW_COST_SCALE = 0.42`) in `AdaptiveVerificationManager._step_cost` and `effective_dead_rows_tau(num_reqs=num_reqs)`, warm-starts `OnlineAcceptanceEstimator` confidence intercepts, and adds early draft halting / batch-bounded draft depth (`early_halt_tau`, `max_draft_steps_for_batch`). Reconcile cleanly with `0786` (`_resolve_active_spec_steps`) and `c60`'s `SINGLE_REQ_TILE_SPILL_SCALE = 2.0`.

2. **Rank 2 — `TODO.md` Spec-Decode ("Extend certified `W_q` Cauchy–Schwarz screening to DSpark `project_head_markov_greedy_pairs`" + "Per-row $\ell_2$-optimal scale & finite outlier bounds in `build_screen_weights`")**
   - **Expected `c1` impact**: **`-0.45 to -0.75 ms/step` (`+1.8% to +3.0%` `c1` TPS)**, 100% bit-exact (`KL = 0`).
   - **Mechanism**: Eliminates reading the full `248 MB` 12-bit `lm_head` shard a second time per decode step (`1,020.3 µs` in `compute_draft_logits_local`) by using `lm_head`'s 8-bit `W_q` screen (`165 MB`) + Markov upper bound to refine only `<2%` active columns in `project_head_markov_greedy_pairs`, and tightens `build_screen_weights()` (per-row $\ell_2$ scale grid search + finite outlier bounds replacing the `498` `inf`-bound rows).

3. **Rank 3 — `TODO.md` Decode #13 ("Mid-network Layer 16/20 dead-row routed-MoE pruning at `c1`") + Decode #1 ("Routed MoE continuous `FC1 -> SwiGLU -> FC2 -> Combine` / `M=6` kernel optimization")**
   - **Expected `c1` impact**: **`-1.0 to -3.5 ms/step` (`+4% to +14%` `c1` TPS)** (`40`-layer routed MoE is `24.17 ms` isolated / `~65%` of `c1` step time).
   - **Mechanism**:
     - Part A (Lossless `M=6` MoE kernel & epilogue tuning): optimize `moe_gate_up` (`417 µs/layer`) / `moe_down` (`183 µs/layer`) / `moe_swiglu_cast` (`5.4 µs/layer`) at `M=6`.
     - Part B (Mid-network Layer 16/20 dead-row pruning at `c1`): detect diverged suffix draft rows at `c1` before Layer 20, zero their routed-MoE weights/indices for Layers `20..39` so `moe_dispatch` skips dead-only experts, and mark pruned rows dead before `rejection_sampler` (`0` false prunes on accepted tokens, `KL = 0`).

4. **Rank 4 — `TODO.md` Spec-Decode ("Fixed-width tree side-candidate leaves in low-confidence tail slots" & "Tail-slot `survival < tau` short-suffix `CopyDrafter` splice")**
   - **Expected `c1` impact**: **`+2% to +6%` `c1` TPS** (if acceptance $\tau$ increases by `>= +0.10 tok/step` inside the existing 16-row single-stream MMA tile).
   - **Mechanism**: Uses the remaining free rows in `decode_16` (or `top2=True` already supported by `project_head_markov_greedy_pairs`) to verify runner-up branch alternatives or short `CopyDrafter` suffixes at ambiguous draft positions.

### Tier 2 — Solid Bit-Exact Kernel & Collective Wins (`+0.5% to +2.5%` `c1` TPS)

5. **Rank 5 — `feat/tp3-decode-recovery` Unported Items #1 & #4 (`compressor.py` replicated uncompressed BF16 up to `tp <= 4` + `input_batch.py` `num_reqs_after_padding` alignment)**
   - **Expected `c1` impact**: **`-110 to -140 µs/step` (`+0.4% to +0.6%` `c1` TPS on `TP4` and `TP3`)**.
   - **Mechanism**: Replicates the 4 compressor layers (`30 MiB` total across the model) up through `tp <= 4` in uncompressed BF16 to eliminate `4` RoCE `all_gather`s + `4` `gather_columns_into` launches + 12-bit unpack per decode step (reconciled cleanly with `c60`'s `compress_ratio in (1, 2)` SP prefill path).

6. **Rank 6 — PR `#1086` (`perf/indexer-multirow-and-decode-split`) + `TODO.md` Decode #5/`#9` ("DSA indexer `KPool` intra-request multi-row reuse + 6-shuffle butterfly")**
   - **Expected `c1` impact**: **`-50 µs/step` at `4K` (`+0.2%`) up to `-0.8 ms/step` at `64K–131K` (`+1.5% to +3%+` at medium/long context)**, `BitExact = True`.
   - **Mechanism**: Dense `indexer_score_v41` 2-row key-tile reuse (`#1086` Part A) plus extending intra-request multi-row register-resident `KPool` reuse & 6-shuffle butterfly across both dense & candidate indexers; carefully threshold `#1086` Part B (`DECODE_INDEX_SPLIT`) so `[local_rows, 512]` all-gather never slows short `M=6` steps.

7. **Rank 7 — PR `#1015` (`tilelang-upstream-patch-audit`, `0014..0019` 2-split `round="rn"` TF32 `mhc.project_streams`) + `TODO.md` TileLang #1 / Decode #10**
   - **Expected `c1` impact**: **`-0.25 to -0.45 ms/step` (`+1.0% to +1.7%` `c1` TPS)** (`mhc.project_streams` runs 86×/step; `12.19 -> 5.92 µs` at `M=1..4`, 42/42 bit-exact).

8. **Rank 8 — PR `#967` (`feat/tilelang-tirx-sparse-mla-opt`: `TVectorizable` on `tirx.reinterpret` + `MergeIfStmt` `kReadState`) + `TODO.md` Decode #6 & #10 ("Two-phase sparse-MLA KV + PDL prologue prefetch" & "Same-request multi-row SWA KV reuse")**
   - **Expected `c1` impact**: **`-0.10 to -0.40 ms/step` (`+0.4% to +1.5%` `c1` TPS)**, bit-exact.

9. **Rank 9 — `TODO.md` Decode #7 & #8 ("Forward-fusing `mHC` stream finalize + MXFP8 quant into consumer projections + Layer-0 entry fusion" & "Piggybacking KV-RoPE onto `wq_b` via tail CTAs")**
   - **Expected `c1` impact**: **`-0.15 to -0.50 ms/step` (`+0.6% to +1.9%` `c1` TPS)** ceiling (removing 61 KV-RoPE launches and 80+ BF16 finalize/cast global round-trips).

10. **Rank 10 — `TODO.md` Spec-Decode ("Grouped 3-layer DSpark `context_kv` projection")**
    - **Expected `c1` impact**: **`-14.4 µs/step` (`27.9 -> 13.5 µs` measured on `sp8`)**, 100% bit-exact.

11. **Rank 11 — `TODO.md` Decode #2 ("Direct producer writes `WO-B` / `moe_combine` into registered `sparknet` RoCE send buffers")**
    - **Expected `c1` impact**: Up to `-0.2 to -0.4 ms/step` ceiling if unmasked `cudaMemcpyAsync` (116 launches/step) shows up in `IKET` trace.

12. **Rank 12 — `TODO.md` Decode #5 remaining ("2-stream shared-expert / DSpark context-KV overlap & `cudaGraphInstantiateFlagUseNodePriority`")**
    - **Expected `c1` impact**: Low-to-moderate (`~0.1–0.3 ms/step` ceiling, must screen for 48-SM contention at `M=6`).

### Tier 3 — Copy-Specific, Multi-Stream (`c8`/`c16`), Sampled (`temp > 0`), Host/Memory, Diagnostic & Negative Items

13. **Rank 13 — PR `#995` (`feat/copy-drafter-wide-ramp-and-guards`) + `TODO.md` Spec-Decode ("Hash-indexed `O(1)` `CopyDrafter`, forward cursor `last_src`, payoff-gated 32/48/64 passes")**
    - Improves `code` / boilerplate / long-context copy matching and adds `slot >= 0` safety + gentle `doublings - 1` decay (minimal impact on `prose-c1` short context).

14. **Rank 14 — PR `#1072` (`perf/shm-broadcast-busy-loop`, `busy_loop_s = 2 ms`)**
    - Eliminates `-90%` host reader CPU spin on `dgx1` during GPU decode steps; reduces host jitter with zero GPU penalty.

15. **Rank 15 — PR `#988` (`feat/sampler-dead-row-skip-and-sparse-topk`) & PR `#1081` (`perf/shard-local-sampled-logits`) & `TODO.md` Spec-Decode ("Coupled Gumbel-Max & block verification for `temp > 0`")**
    - Huge speedup for **`temp > 0` sampled decode** (`-0.13 to -2.15 ms/step` collective savings + dead-row skip in `Sampler`); `0%` on greedy (`temp = 0`) `prose-c1`. (Note: `#988` Part 2 `_sample_sequential_topk` is partially present in `c60`; port `Sampler` `dead_row_cut` + `apply_markov_bias_topk` cleanly.)

16. **Rank 16 — High-Concurrency & Long-Context Calibration (`PR #1078` `48-row decode tile`, `PR #986` `adaptive-verify-graph-context-cost`, `PR #953` `cohort-sync-verification`, `TODO.md` "Draft depth by load" & "Cross-request verify-row allocation")**
    - Targets `c8`/`c16` (`-2.40 ms/step` at `M=42`, `-98%` second-wave spills at `c16`, `CONTEXT_ROW_COST_KNEE = 262144`) and `c1` `CopyDrafter` burst plateau (`20 -> 16`, `+1.1%`). Reconcile `#986`, `#953`, and `#1003` cleanly in `adaptive_verification.py`.

17. **Rank 17 — PR `#984` (`feat/uma-allocator-hygiene`: `libc10.so` `mi_collect(True)` + `malloc_trim(0)`)**
    - Reclaims `198–391 MiB/rank` host `VmRSS` and `1.92 GiB` post-prefill CUDA slack; `0%` change on `c1` TPS.

18. **Rank 18 — Scope-Limited / Opt-In / Zero-Impact-on-Default-`tp4-ring` Items**:
    - `PR #1004` (`exp/dspark-topk-selection-reuse` — `1107` uses `compress_ratios=[0,0,0]` window-only drafting); `TODO.md` `spec_near_tie_k` (`#1211`, opt-in); `TODO.md` `answer-only repetition_penalty`; `PR #981` (`TP=2` NVMe expert cache); `PR #958` (`TP=3` NCCL); `PR #954`/`#1014` (`LSA`, default off); `PR #962` (`IKET`/`ROCE_TRACE` diagnostic); `0647` NVFP4 scale search.

19. **Rank 19 — PR `#1002` (`feat/online-dspark-adaptation`) — Previously Measured Negative on DS4.1 (`-3.84%` coding `c1`, `-5.53%` held-out `c1`)**
    - Retained only as opt-in reference unless calibration-only (`VLLM_DSPARK_ONLINE_CALIBRATION=1`) is screened separately.

20. **Rank 20 — PR `#1141` (`buffered-atomic-admission`)**
    - Multi-stream gateway batching (`c16` waves); `0%` effect on `c1`.

---

## Already Integrated in `candidate-50..60` (Clean Up in `TODO.md` & PRs After Verification)
- `0762` (`feat/moe-l2-eviction-policy`), `0763` (`feat/certified-lm-head-screen`), `0764` (`feat/async-output-fastpath`), `0765` (`perf/decode-2stream-kv-indexer-overlap`), `0766` (`perf/qb-indexer-qb-tile-contiguous`), `0767`/`0809` (`sparse-mla-stage1-kv-gather` + unroll), `0769` (`spec-decode-elide-tp-conf-broadcast` & Decode #12 broadcast skip when `tp_invariant_confidence & rank_order_reduce_scatter`), `0776` (`vectorize-indexer-KPool-loads`), `0777` (`hw-e2m1x2-indexer-gemm-quant`), `0781`/`0806` (`64k` L2 prefetch), `0782` (`pack12-unconditional-loads-swar`), **`0783` (= PR `#987` `WO-B` 128-row `WIDE_DECODE` & high-concurrency cliffs)**, **`0784` (`Decode #4` 128-row vocab head)**, `0786` (`remaining-token-draft-cap`), `0788` (`cudagraph-decode-optimizations`), `0789`/`0790` (`copy-drafter-survival-and-tile-cap` & `long-context-scan`), `0791` (`compact-packed-vocab-tables`), **`0801` (`Decode #11` RoCE `SPARKNET_ROCE_PROXY_CPU=big` & API 2 pipelined RoCE)**, `0805` (`glp-spec-v3`), `0808` (`indexer` sink score reduction), `0810` (`MoE` blockscaled K-loop unroll), **`0814` (`TileLang #2` pre-multiplied `q_weight * k_weight` + fused `dspark_confidence_calibrate_kernel` + `_accumulate_kernel` from `tp3-decode-recovery`)**.

---

## Log
- **Initialized**: Branch `c61-decode` at `ac768604d`; pushed finalized `candidate-60` and `c61-decode` to `origin`.
- **Rank 1 (`PR #1003` sunk-cost `DRAFT_COST_RATIO=0.55`, `SINGLE_TILE_ROW_COST_SCALE=0.42`, warm-start calibration) + Rank 2 (certified 2-pass `W_q` Markov vocab screen) — Screen `20261116T185316-screen-r1-2-tp4.json` (`tp4-ring`)**:
  - Determinism (`token_determinism_record.py`): **PASS** (`reference_sha256 = 0a2dacdd...`, `long_sha256 = 12fd8257...` exact match between `c60-base` and `c61-r1-2`, 0 token diffs across all 6 scenarios).
  - `c16` (`90s + 30s`): `c60-base = 272.92 tok/s` (`7.87 steps/s`, `127.12 ms/step`, `34.72 tok/step`) vs `c61-r1-2 = 272.17 tok/s` (`7.90 steps/s`, `126.58 ms/step`, `34.48 tok/step`).
  - `c1` (`150s + 30s`): `c60-base = 83.23 +/- 4.84 tok/s` (`35.86 steps/s`, `27.89 ms/step`, `2.329 tok/step`, `accept = 1.333 / 2.647`, `coverage_at_1/2/3/4 = 0.860 / 0.464 / 0.224 / 0.098`) -> **`c61-r1-2 = 80.53 +/- 5.49 tok/s` (`-3.25%` regression)** (`33.70 steps/s`, `29.67 ms/step`, `+1.78 ms/step`, `2.397 tok/step`, `accept = 1.398 / 3.396`, `coverage_at_1/2/3/4 = 0.981 / 0.771 / 0.430 / 0.213`).
  - **Root cause & decision**: Because `Rank 2` is 100% bit-exact (`-218.4 us/step` on `sp8`: `1145.1 -> 926.7 us` in CUDA Graph, `163 passed`), the `+0.749` extra draft rows/step (`2.647 -> 3.396`) and `-6.02%` step rate came solely from `Rank 1` lowering `effective_dead_rows_tau` (~3.7x lower threshold at `c1`), pushing low-survival tail rows into 384-expert top-6 MoE (`+1.78 ms/step`). **Reverted `Rank 1`** (`612a3ca09`) while keeping bit-exact `Rank 2` (`0f2eb0a85`).
- **Rank 3 (`M=6` MoE stage/epilogue tuning & Layer 16/20 dead-row pruning) — Evaluated on `sp8` (No Change)**:
  - `M=6` MoE breakdown (`E=384, topk=6, H=5120, I=576`): `disp=4.1 us`, `gate_up=392.2 us`, `moe_swiglu_cast=4.1 us`, `down=167.9 us`, `combine=4.1 us` (`graph=564.3 us`). `c60`'s `s3` (`num_stages=3, down_num_stages=2`) beats `s2` (`2` stages) and `s2+fuse_swiglu` at `M=2` (`219.2` vs `223.8` / `223.1 us`), `M=6` (`599.7` vs `607.2` / `615.1 us`), and `M=48` (`3722.3` vs `3744.9` / `3735.0 us`). `DeepSeek-V4.1-Flash` safetensors carry no intermediate Layer 16/20 LM head/probe weights; Layer-0 `DEAD_ROWS_TAU` governs dead-row pruning.
- **Rank 4 (`CopyDrafter` tail splice / tree side-candidate leaves) — Evaluated on `sp8` (Non-viable in `c61`)**:
  - Target verification kernels (`sparse_mla.py`, `indexer.py`, SWA, causal KV slot mapping, `rejection_sampler.py`) strictly require linear causal chains without non-causal tree bitmasks; `top2_branch` Markov confidence promotion is already present in `dspark/speculator.py` and `tilelang_sampling.py` (off by default); and moving `CopyDrafter.prepare_alloc` after `DSpark.propose` would stall async `alloc` D2H delivery behind DSpark.
- **Rank 5 (BF16 compressor replication up through `TP<=4`) — Verified on `sp8` (`b901076ea`)**:
  - Set `disable_tp = tp_size <= 4` and `pack_bf16 = not disable_tp` in `compressor.py` + guarded `pack_bf16` in `tilelang/linear.py` (making `_sp_weight = weight` zero-copy for SP prefill and eliminating 4 decode RoCE `all_gather`s + 4 `gather_columns_into` launches + 12-bit unpacks). **Verified**: `test_deepseek_v4_1_compressor.py`, `test_deepseek_v4_1_compressor_prepare.py`, `test_deepseek_v4_1_sp_prefill.py`, and `test_deepseek_v41_tilelang_linear.py` -> **327 passed, 0 failed**.
- **Rank 6 (`indexer.py` `<2>` decode multi-row reuse + `attention.py` `DECODE_INDEX_SPLIT`) — Verified on `sp8` (`a9b2c07fa`)**:
  - Paired consecutive query rows in dense `indexer_score_v41` (`<2>` for `rows < 64`), cutting CUDA graph latency by `-10.8%` to `-33.1%` across `rows=6..48` (`2K..131K` context); kept `candidates=True` at 1-row. Added `_decode_index_split` gated to `rows >= 24 or eager_long (>= 16384)` so captured short `M=6` decode avoids synchronous RoCE `all_gather` overhead. **Verified**: `19 passed, 1 skipped` (`test_deepseek_v41_tilelang.py -k "indexer or pages or paired"`) and `264 passed, 0 failed` (`test_deepseek_v4_1_attention.py` + `test_deepseek_v4_1_sp_prefill.py`).
- **Rank 7 (`mhc.py` 2-split branchless IEEE RN TF32) — Verified on `sp8` (`ef64f9f42`)**:
  - Replaced 3-split `& 0xFFFFE000` with 2-split branchless uint32 IEEE round-to-nearest-even in `load_split_tf32`, dropping `F_low` SMEM and the 3rd `T.gemm` call. CUDA graph latency (`H=5120`): `pre` `4.98->4.27 us` (`M=1`), `5.09->4.41 us` (`M=6`), `5.70->4.77 us` (`M=16`), `19.15->11.98 us` (`M=64`); `post_pre` `5.35->4.77 us` (`M=1`), `5.49->4.91 us` (`M=6`), `6.23->5.48 us` (`M=16`), `20.86->14.14 us` (`M=64`), with lower FP64 relative error across all cases. **Verified**: `test_deepseek_v41_tilelang_mhc.py` -> **180 passed, 0 failed** (`128.02s`).
- **Rank 8 (`PR #967` `98df13420` `sparse_mla.py`) — Evaluated on `sp8` (Already Superseded)**:
  - Diffed all 6 `SparseMlaV41Plan` routes (`idx_width=0, 512` x `decode, extend, split`) before vs after `_install_tirx_compiler_fixes()` inside `candidate-60`: **byte-for-byte identical CUDA source (`6/6 identical=True`)**, because `candidate-60` already hoists `dequant_swa` scales out of the vectorized loop (`0` `LegalizeVectorizedLoop` warnings) and already rewrote Stage 1/Stage 2 (`PartialStats > 0`), leaving only 1 `swa_active` check in Stage 1 decode (`0` in Stage 2).
- **Rank 9 (`rope_(heads=16, dim=512)` on `q`) — Measured on `sp8`**:
  - 40-layer CUDA graph replay runs in `23.58 us` total at `rows=6` (`0.59 us/layer`), already overlapped on `aux_stream` with `indexer.wq_b` (`<0.02 ms/step` ceiling); KV-RoPE is already fused into `attention_front`.
- **Rank 10 (Grouped 3-layer DSpark `context_kv` via `TileLangGroupedBlock32Rows`) — Verified on `sp8` (`9a20bb954`)**:
  - Stacked homogeneous `TileLangBlock32Rows` context-KV projections along N using per-slice `SPLIT_FP8[(512, 5120)]` (`TileLangGroupedBlock32Rows`, `_grouped_context_kv_projection`). CUDA graph speedup (`3x512 -> 1x1536`, 100% bit-exact `int16 torch.equal`): `rows=1: 18.56->16.12 us (-13.1%)`, `rows=6: 14.83->14.00 us (-5.6%)`, `rows=12: 11.58->10.68 us (-7.8%)`, `rows=24: 16.59->15.16 us (-8.6%)`, `rows=48: 25.80->24.47 us (-5.2%)`. **Verified**: `test_deepseek_v41_tilelang_dspark_context.py` + `test_deepseek_v4_dspark_context_kv.py` -> **30 passed, 0 failed** (`28.31s`).
- **Rank 14 (`PR #1072` `36d9084c7`: `SHM_SPIN_BUSY_LOOP_S = 0.002` in `shm_broadcast.py`) — Verified on `sp8` (`34eaef2f6`)**:
  - **Verified**: `tests/distributed/test_shm_broadcast.py` -> **32 passed, 0 failed, 0 skipped** (`19.98s`).
- **Rank 15 (`PR #1081` `827f82449`: shard-local sampled verification for `temp > 0`) — Verified on `sp8` (`1b7d6dded`)**:
  - **Verified**: `test_greedy_verification.py`, `test_gpu_verified_logprob_token_ids.py`, `test_dead_rows.py` -> **57 passed, 0 failed** (`65.57s`).
- **Rank 16 Part A (`PR #1078` `545c67138`: 48-row decode projection tile tier) — Verified on `sp8` (`b8aeac1d7`)**:
  - Added `block_M = 48` tier to `DECODE_TILE_ROWS`, `PARTIAL_ROW_TILES`, `DECODE_FP8_CONFIGS`, `SPLIT_FP8`, `SPLIT_BF16`, `split_bucket`, and `single_heights`/`multi_heights` validation in `linear.py`. **Verified**: `test_deepseek_v41_tilelang_linear.py` + `test_deepseek_v41_tilelang_dspark_context.py` (`-k "test_decode_48row or block32 or tile_contiguous or grouped_context"`) -> **10 passed, 0 failed** (`113.57s`).
- **Additional `c60` Audit**: Confirmed `8071740d9` (`moe-swiglu-and-dual-stream-graph`), `d2bcc24f0` (`0045` `bf16x2` producer MXFP8 casts), and `2dd7c601c` (`NewRequestData` int32 `PickleBuffer` IPC) are already integrated in `candidate-60`.
- **Recorded `c60-base` `c1` Style Breakdown (`c61-r1-2-screen-tp4/c60-base/c1.json`)**:
  - `prose-c1 = 78.4 tok/s` (`38` reqs), `code-c1 = 111.6 tok/s` (`8` reqs), overall `c1 = 83.23 tok/s`, `c16 = 272.92 tok/s`. Survival profile: `survival_at_0..4 = 0.659 / 0.375 / 0.173 / 0.092 / 0.043`.
