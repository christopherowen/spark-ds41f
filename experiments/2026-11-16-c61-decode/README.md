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
- **Additional `c60` Audit**: Confirmed `8071740d9` (`moe-swiglu-and-dual-stream-graph`), `d2bcc24f0` (`0045` `bf16x2` producer MXFP8 casts), `2dd7c601c` (`NewRequestData` int32 `PickleBuffer` IPC), `PR #987` (`128`-row `WO-B`, `_DENSE_VARLEN_DECODE_MAX_REQS = 2`, `reduce_scatter_rank_order`), and `PR #988` part 1 (`dead_row_cut` verification sampler skip) are already integrated in `candidate-60`.
- **`front.py` Strided `kv_raw` Fix (`e652d7d2f`)**: First `c61-r2-screen-tp4` boot failed at `warmup_kernels` (`RuntimeError: kernel attention_front input kv_raw strides[0] violates packed ABI constraint`) because `TileLangGroupedBlock32Rows` emits `stride(0) == 1536` slices (`out.split(512, dim=1)`). Updated `context_kv_front_kernel` to use `T.StridedTensor((M, KV_LATENT), (kv_stride, 1), T.bfloat16)` with `T.assume(kv_stride % 16 == 0)` (vectorized `uint4` loads preserved). **Verified**: `test_deepseek_v41_tilelang_front.py` + `test_deepseek_v41_tilelang_dspark_context.py` -> **99 passed, 0 failed**.
- **Rank 13 (`PR #995`: CopyDrafter negative-slot guards, `doublings - 1` decay, wide `greedy_rejection_kernel`, `SPARK_DS41F_COPY_MAX_DRAFTS`) — Verified on `sp8` (`57faad2c2`)**:
  - Guarded negative slots in `copy_alloc_kernel`/`copy_apply_kernel`, stepped down `doublings` gently on partial acceptance `>= INITIAL`, expanded `greedy_rejection_kernel` to `width <= 65` (`passes = -(-width // 32)`), and added `SPARK_DS41F_COPY_MAX_DRAFTS` override in `speculative.py`. **Verified**: `test_copy_drafts.py` + `test_greedy_verification.py` -> **73 passed, 0 failed** (`114.31s`).
- **Recorded `c60-base` `c1` Style Breakdown (`c61-r1-2-screen-tp4/c60-base/c1.json`)**:
  - `prose-c1 = 78.4 tok/s` (`38` reqs), `code-c1 = 111.6 tok/s` (`8` reqs), overall `c1 = 83.23 tok/s`, `c16 = 272.92 tok/s`. Survival profile: `survival_at_0..4 = 0.659 / 0.375 / 0.173 / 0.092 / 0.043`.

- **Batch 2 (`c61-r2-screen-tp4`, `20261116T203227-screen-tp4.json`, deploy commit `8b8c2c3f0`) — Screen & Diagnosis (`tp4-ring`)**:
  - Booted cleanly (`+126.0s`), `c16 = 273.23 tok/s` (vs `c60-base` `272.92`), `greedy-c61-r2` `identical: true` (0 token diffs across all 6 scenarios).
  - **Findings & Reverts**:
    1. **Stalls (`Rank 14` / PR `#1072` `34eaef2f6`)**: Lowering `SHM_SPIN_BUSY_LOOP_S = 0.002` in `shm_broadcast.py` caused 9 five-second ZMQ PUB/SUB lost-notify stalls (`ttft_s p90 = 5.122s, p99 = 5.128s`, dragging raw `c1` to `62.97 tok/s`) because `last_read` is recorded before the `~27 ms` GPU step (`> 2 ms`), skipping `sched_yield()` and hitting `SHM_READER_RECHECK_INTERVAL_MS = 5000`. Excluding stalls, `c61-r2` improved both `prose-c1` decode (`80.47 -> 81.07 tok/s`, unstalled e2e `78.45 -> 78.83`) and `code-c1` decode (`116.21 -> 119.19 tok/s`, unstalled e2e `111.59 -> 113.95`, `+2.56%`). Reverted `shm_broadcast.py` (`33a662b12`).
    2. **Determinism hash shift (`Rank 7` `ef64f9f42`)**: 2-split TF32 in `mhc.py` shifted `reference_sha256` from `0a2dacdd...` (`c60-base`) to `81a029f1...` (`hc_*_fn` weights use low 8 bits). Reverted `mhc.py` (`99fa40533`).
    3. **Indexer `<2>` decode split (`Rank 6` `a9b2c07fa`)**: Reverted `indexer.py` + `attention.py` (`49ffbed7a`) because `<2>` included `2 <= rows < 64` (avoiding `+0.34% @4K` risk).
- **Batch 2 Follow-up (`c61-r3-screen-tp4`) Prepared**:
  - **Rank 16 Part B (`PR #986`, `274e85635`)**: Calibrated `adaptive_verification.py` (`SATURATED_ROW_COST_KNEE=64`, `COALESCED_ROW_COST_SCALE=0.28`, `COALESCED_MIN_REQS=10`, `SECOND_WAVE_STEP_ROWS=6.714`, `CONTEXT_ROW_COST_KNEE=262144`, `SINGLE_REQ_TILE_STEP_ROWS=1.0`, preserving `num_sampling_requests <= 1` for copy-spill tests). Verified: `test_adaptive_verification.py` -> `51 passed, 1 skipped` (`11.55s`).
  - **Rank 17 (`PR #984`, `ca879889a`)**: Applied UMA allocator hygiene (`reclaim_uma_memory` / `trim_host_heap`, `freeze_gc_heap` ordering). Verified (`sp8` container `spark-ds41f:candidate-60` with `vllm_test_utils` on `PYTHONPATH`): `241 passed, 1 deselected, 14 warnings` (`63.48s`).
  - **Rank 6 (`indexer-prefill-exact` `pipe`, `00f2fd101`)**: Applied `experiments/2026-11-16-indexer-prefill-exact/overlay/vllm/models/deepseek_v4_1/tilelang/indexer.py` (pipelined `__tl_fetch_k_dense`, `bf16x2` epilogue, 32-bit paired head sum; avoids `<2>` short-decode row-pair assumptions). Verified on `sp8`: `test_deepseek_v41_tilelang.py -k "indexer or topk"` -> `28 passed, 1 skipped, 41 deselected, 1 xfailed` (`40.90s`); `test_deepseek_v41_tilelang_pdl.py` -> `15 passed` (`29.08s`).
  - **Rank 15 Part 2 (`PR #988` part 2) + Rank 16 Part C (`PR #953`) (`c4094a105`)**:
    - Sparse top-k Markov bias and sampling (`apply_markov_bias_topk` in `qwen3_dspark.py` and `nvidia/dspark.py`, `topk_greedy_sample` / `topk_gumbel_sample` / `topk_drafting_calls` in `tilelang_sampling.py`, `_can_sample_sparse_topk` and `_sample_sequential_topk` preserving `active_steps` truncation in `dspark/speculator.py`).
    - Cohort-atomic verification budgeting + phase-locking catch-up merged with `#986` `_step_cost` in `adaptive_verification.py`, plus `_coalesce_idle_burst` (`idle_wakeup_first_poll_s = 0.0003`, `idle_wakeup_burst_poll_s = 0.0015`) in `v1/engine/core.py`.
    - Verified on `sp8`: `171 passed, 1 skipped` (`test_adaptive_verification.py`, `test_late_admission_sink.py`, `test_worker/test_late_admission.py`, `test_dspark_topk.py`, `test_dspark_v41_gathered_markov.py`, `test_dead_rows.py`, `test_dspark_greedy_markov.py`) and `test_late_admission.py` -> `7 passed` (`7.05s`).
  - **Ranks 8, 9, 11, 12, 18–20 Evaluation Summary**:
    - **Ranks 8, 9, 12**: `c60` already includes `s3` MoE, `mhc_finalize + cast_block_mxfp8` (producer `bf16x2` MXFP8 casts), `attention_front` KV-RoPE, 2-stream `_forward_decode_overlapped`, `pdl.chain()` + async `issue_stage_chunk` in `sparse_mla`, `sink-wq-screen`, and `dspark-markov-trunc`.
    - **Rank 11**: `SparknetOneShotAllReduce` exposes no external producer `send_buf` API.
    - **Ranks 18–20**: Already opt-in (`compress_ratios=[0,0,0]`, `spec_near_tie_k`, `TP=2`/`TP=3`, `LSA`, `IKET`, `PR #1002`), negative on DS4.1 (`PR #1002`, `2026-11-15-decode-prefix-reuse` regressed prefill `-0.73%..-1.88%`), or multi-stream gateway only (`PR #1141`).
  - **Isolated Non-MoE `M=1..6` CUDA-Graph Profiler on `sp8` (`µs/call` -> `×40 layers ms`)**:

    | Kernel | `M=1` | `M=3` | `M=4` | `M=6` | `40L` range |
    | --- | --- | --- | --- | --- | --- |
    | `mhc.pre` | `10.32 µs` | `11.04 µs` | `11.11 µs` | `10.88 µs` | `0.41–0.44 ms` |
    | `mhc.post_pre` | `10.89 µs` | `11.25 µs` | `12.27 µs` | `11.35 µs` | `0.44–0.49 ms` |
    | `wqa_wkv(448)` | `8.21 µs` | `8.16 µs` | `9.35 µs` | `7.12 µs` | `0.28–0.37 ms` |
    | `front(1280+512)` | `4.10 µs` | `4.09 µs` | `5.84 µs` | `4.10 µs` | `0.16–0.23 ms` |
    | `wq_b(8192)` | `13.67 µs` | `14.96 µs` | `14.76 µs` | `14.32 µs` | `0.55–0.60 ms` |
    | `rope_(16,512)` | `4.26 µs` | `2.71 µs` | `2.68 µs` | `2.68 µs` | `0.11–0.17 ms` |
    | `sparse_mla(512)` | `12.24 µs` | `12.30 µs` | `12.29 µs` | `12.30 µs` | `0.49 ms` |
    | `sparse_mla(swa)` | `10.25 µs` | `10.56 µs` | `11.00 µs` | `10.74 µs` | `0.41–0.44 ms` |
    | `wo_proj` | `22.62 µs` | `22.62 µs` | `22.55 µs` | `22.53 µs` | `0.90 ms` (warm) |
    | `shared_experts` | `20.79 µs` | `20.50 µs` | `19.89 µs` | `20.94 µs` | `0.80–0.84 ms` (warm) |
    | `idx.dense16K` | `17.13 µs` | `20.92 µs` | `22.96 µs` | `27.40 µs` | — |
    | `idx.dense64K` | `26.26 µs` | `41.81 µs` | `51.70 µs` | `62.63 µs` | — |

    Physical byte floor (`5,708 MB/step` at `c1` across 39 routed MoE layers + dense projections/heads/drafter) is `21.41 ms` at `266.6 GB/s`.

- **Batch 3 (`c61-r3-screen-tp4`, `20261116T213738-screen-tp4.json`, deploy commit `c08714c26`) — Screen & Diagnosis (`tp4-ring`)**:
  - **`c1` (`150s + 30s`, `0` stalls)**: `overall = 83.68 tok/s` (`CI95 ±5.90%`, `35.95 steps/s` = `27.81 ms/step`, `2.336 tok/step`, `ttft_p50 = 0.111s`, `ttft_p99 = 0.159s`, `ttft_max = 0.159s`, `0` failed, `0` foreign requests) vs `c60-base` (`83.23 tok/s`, `35.86 steps/s`, `2.329 tok/step`, `ttft_p99 = 0.162s`). Code/non-code (`idx % 16 in (2, 14)`): non-code (`n=48`) `dec = 82.67` / `e2e = 80.05 tok/s` (`c60-base`: `82.47 / 79.87`); code (`n=11`) `dec = 116.93` / `e2e = 111.35 tok/s` (`c60-base`: `116.78 / 111.22`). All 5.0 s stalls eliminated.
  - **Determinism (`greedy-c61-r3.jsonl`)**: `identical: true`, `token_diffs: 0`, `logprob_diffs: 0` across all 6 scenarios (`c8`, `staggered`, `mixed`, `cached`, `again`, `mixed15`), exact match on `reference_sha256 = 0a2dacdde6ff2147367bc329818a9707862f7de08ea7e0c0513b01bfff0fd37f` and `long_sha256 = 12fd825767d21b022ed3fdcbbc6937289168c07882f022727d0725c86aa689fc`, and `0` content diffs across all 81 recorded calls.
  - **`c16` (`90s + 30s`) Regression & Root Cause**: `269.62 ± 1.04% tok/s` (`7.278 steps/s` = `137.4 ms/step`, `sched/step = 40.05`, `out/step = 37.06`, `draft/draft = 2.471`, `acc/draft = 1.420`) vs `c60-base` (`272.92 tok/s`, `127.1 ms`, `sched = 37.51`, `draft/draft = 2.123`) and `c61-r2` (`273.23 tok/s`, `127.5 ms`, `sched = 37.59`, `draft/draft = 2.140`). Lowering `33..64`-row marginal slope to `COALESCED_ROW_COST_SCALE = 0.28` (`PR #986`) plus cohort-atomic grouping (`PR #953`) inflated `draft/draft` and pushed `sched/step` past `40`, stalling step latency by `+9.9 ms/step` with zero gain in `acc/draft`.
- **PR `#1225` (`indexer-prefill-exact`, `0817..0819`) + `PR #986`/`#953` Revert (`c61-r4-screen-tp4`, `20261116T215612-screen-tp4.json`, deploy commit `0816e9600`)**:
  - Integrated `0818` (`max.NaN.bf16x2` score ReLU, `835ccd250`) and `0819` (`uint4` 8×BF16 top-$k$ key loads, `3caf20087`) on `sp8` and merged PR `#1225` (`f381e0b13`). Verified on `sp8`: `test_deepseek_v41_tilelang.py -k "indexer or topk"` -> `30 passed, 1 skipped` (`test_indexer_nonfinite_query_scale` and `test_indexer_candidates_nonfinite_query_scale` passing; `PDL sources: 34 kernels launched, 34 with PDL, 0 violations`), `test_deepseek_v41_tilelang_pdl.py` -> `15 passed`, plus `5/5` bit-exact tensor checks (`rows=6, 96, 256`, `width=8K..1M`) vs `candidate-60`.
  - Reverted `PR #986` and `PR #953` (`adaptive_verification.py` and `v1/engine/core.py`) on `sp8` (`cc3114976`) and ran `c61-r4-screen-tp4`:
    - `c1` (`150s + 30s`, `0` failed, `0` foreign): **`83.70 ± 5.85% tok/s`** (`35.953 steps/s` = `27.81 ms/step`, `sched/step = 2.525`, `acc/draft = 1.3411`, `ttft_p50 = 0.111s`, `ttft_p99 = 0.133s`, `tpot_p50 = 11.65 ms`).
    - `c16` (`90s + 30s`, `0` failed, `0` foreign): **`271.94 ± 1.13% tok/s`** (`7.878 steps/s` = `126.94 ms/step`, `sched/step = 37.307`, `acc/draft = 1.4000`, `ttft_p50 = 0.409s`, `ttft_p99 = 0.495s`, `tpot_p50 = 56.95 ms`) — `c16` regression from `#986`/`#953` confirmed resolved.
    - Determinism (`record-greedy-c61-r4.json`): `identical: true`, `0` token/logprob diffs across all 6 scenarios (`reference_sha256 = 0a2dacdde6ff2147367bc329818a9707862f7de08ea7e0c0513b01bfff0fd37f`, `long_sha256 = 12fd825767d21b022ed3fdcbbc6937289168c07882f022727d0725c86aa689fc`).
- **Pre-Release Cleanups, `shm_broadcast` Fix, & `Rank 6` `<2>` Indexer (`c61-r5-screen-tp4` Staged, `23` overlay files)**:
  - Guarded unwritten bonus/greedy global reads in `tilelang_rejection.py` (`705 passed`, `compute-sanitizer initcheck`: `6 passed, 0 errors`), deduplicated `qwen3_dspark.py` (`apply_bias_gathered` -> `apply_bias_topk`), removed test-only `SPARK_DS41F_COPY_MAX_DRAFTS` from `vllm/config/speculative.py` (`acefbc852`).
  - Fixed `PR #1072` (`shm_broadcast.py`, `f2621b38c`): added per-wait spin anchor (`wait_started` / `_wait_epoch`) so `2 ms` busy spin runs post-wakeup rather than staling across the `~27.8 ms` GPU step, raised `PUB` `SNDHWM` (`1 -> 64`), and drained stale `CONFLATE=1` notifications before polling (`test_shm_broadcast.py`: `32 passed`).
  - Integrated and verified `Rank 6` (`PR #1086` / `adf31bf6f`) `<2>` dense `indexer_score_v41` pairing on top of `00f2fd101`'s BF16 8 KiB `contrib` pipeline plus `_decode_index_split` (`rows >= 24 or eager_long >= 16384`): CUDA graph indexer latency dropped at every width (`width=4096 rows=6`: `11.55 -> 10.34 us`, `-10.5%`; `width=65536 rows=6`: `60.39 -> 44.85 us`, `-25.7%`; `width=131072 rows=6`: `96.74 -> 68.27 us`, `-29.4%`). Verified: `test_deepseek_v41_tilelang.py -k "indexer or pages or paired"` -> `21 passed, 1 skipped`, `test_deepseek_v4_1_attention.py tests/models/test_deepseek_v4_1_sp_prefill.py` -> `264 passed`, `test_deepseek_v41_tilelang_pdl.py` -> `15 passed`.
  - Cherry-picked `PR #1222` (`2cad2e9cf`, per-profile `head_url` resolution) and `PR #1223` (`1369922ca`, per-writer atomic JSON temp files); `tests/test_lab.py tests/test_source_locks.py` -> `124 passed`.
