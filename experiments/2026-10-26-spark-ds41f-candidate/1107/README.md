# Candidate 1107

Unpromoted release candidate combining the validated `C40` runtime base with the complete 2-boot GPU-qualified post-`1030` serving, kernel, compiler, speculative decoding, sequence-parallel, multimodal, and collective optimizations.

## Source Identity

| Component | Base Revision | Patch Series | Patch Count | Patch Head | Expected Tree | Patchset SHA-256 |
|---|---|---|---:|---|---|---|
| `vllm` | `19f2c20ed4d64ae22bd009ceacadd85e2a81e354` | `experiments/2026-10-26-spark-ds41f-candidate/1107/vllm/series` (`0001..0761`, `641` active) | `641` | `2756478b60b90700b9918b457aeb38657da637b5` | `ed1bdb7a7684da33d403a055521ac0c4dd7ca670` | `26fac4e56977e6f2dc057821f8fa5d10f1d6159861843e8f725bb8a2a10cef38` |
| `tilelang` | `2123e412ce2217b324c50c4ec967213906456ac5` | `experiments/2026-10-26-spark-ds41f-candidate/1107/tilelang/series` (`0001..0013`) | `13` | `92a1d3a0eba72b440297d4090bbfc498a47b0eb6` | `e2b001828fd4e38e6a2eee61f818d5083cb1335f` | `2142ea448ceec22c0643badfd24bba129eb4b15e3f03a0c4c8c3ccb468b8e0d1` |
| `sparknet` | `1fd6a1fbd8dc4884057046d1ebd93d585c9e4bcf` | `none` (`refs/heads/main`) | `0` | `1fd6a1fbd8dc4884057046d1ebd93d585c9e4bcf` | `8cc28a5cd3e683c1f114ed843453e3ce34f323e0` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

## Included Workstreams and 2-Boot GPU Qualifications

1. **Base Stack (`C40`, `0001..0689`, 574 patch files)**:
   - Validated `C38` base (`0001..0687`) plus default `CopyDrafter` enablement and safe envelope (`0688`) and lazy top-k scratch-read guard from PR `#878` (`0689`).
2. **TileLang Runtime Closure & Single-Rank Path (`#885`, `#886`, `#887`, `#893`, `#895`, `#912`, `#915`, `0690..0713`, `0717`, `0752..0754`)**:
   - Replaces ATen compute kernels on the request path with TileLang or DMA (`copy_logits` with pitched 2D `cudaMemcpy2DAsync` for aligned FP32 matrices `>= 8 MiB`, 8-D strided `rank_order_strided`, magic-multiplier `gather_columns_into` / `concat_rank_shards` with TensorIterator bypass for tiny tails `<= 128` vector units in `0713` and PDL chaining in `0753`, `gather_logits_kernel`, native strided `int32`/`int64` token-ID score inputs, single-pass `prompt_scores_and_ranks`, single-rank fused Markov drafting in `0717` with warp-shuffle vocab greedy epilogue and L2-hot step dispatch in `0752`, and warp-broadcast target logit + fused barriers in `topk_log_softmax` in `0754`).
3. **Serving Contract & Dead-Code Cleanup (`#894`, `0720..0722`)**:
   - Enforces the sequence-parallel MoE contract (`tp_size > 1` with `use_sequence_parallel_moe=False` rejected), removes dead `VLLM_DS41_ENGRAM_OVERLAP`, `VLLM_DS41_MARKOV_NVFP4`, and `VLLM_DS41_DRAFT_NVFP4_HEAD` fallback branches, and records capability `ds41-serving-contract`.
4. **Frontend Request & Utility Lifetime Cleanup + Worker Test Isolation (`#890` + `C46`, `0723..0724`, `0730`)**:
   - Rolls back partially registered `OutputProcessor` / `CoreClient` state on failed `generate()` submission (`0723`), pops abandoned `call_utility_async` futures on cancellation or send failure (`0724`), and isolates Triton reference monkeypatching (`0730`).
5. **HTTP Disconnect & Repeated Cancellation Ownership (`#918`, `0725..0726`)**:
   - Cancels and awaits both `handler_task` and `cancellation_task` inside a shielded `CancelScope`, closes abandoned streaming generators (`aclose()`), and absorbs repeated `asyncio.CancelledError` interrupts in `finally:` until child cleanup finishes (`0726`).
6. **CED Encoder Prefix Endpoint Retention & Finite Decoder Recovery (`#923`, `#919`, `0727`, `0731..0734`)**:
   - Retains `(resend, request.num_prompt_tokens)` in `KVCacheCoordinator._get_retention_endpoints` only when `resend // block != num_prompt_tokens // block` (`0727`, saving `-10.22 to -10.95 ms` / `+26.0% to +32.8%` on cross-block GPU CED prefills), reads live activation rows in padded MXFP8 decode tiles (`0731`), and reconstructs private finite decoder history after preemption (`0732..0734`).
7. **Native-Depth Verification CUDA Graph Shapes & Replay (`#921`, `0728..0729`, `0735`)**:
   - Adds native verification row totals `[18, 30, 36, 42]` (`+4` target graphs on TP3, `-2.97 ms` / `+22.52%` at `42` rows in `0728`) and replays DSpark drafter metadata inside query graphs (`0735`, `-0.39 ms/step`).
8. **Sequence-Parallel Prefill & mHC/MLA Kernel Optimizations (`0736..0740`, `0742..0743`, `0755..0756`)**:
   - Selects `prefill_swizzle_panel=2` for `M=64` routed MoE GEMM (`0736`), exchanges WO-A MXFP8 rows and projects WO-B on local SP rows (`0737`), exchanges SP Engram prefill columns and slices projection (`0738`), removes fused mHC entry optionality for 2-launch PDL (`0739`), right-sizes sparse-MLA head block (`block_H=16`) for 16-head TP3 ranks (`0740`), keeps CED boundary normalization and global KV projection on local SP rows before latent gather (`0742..0743`), fuses rank-major shards in `mhc.post` and Engram post-gate (`0755`), and chains routed MoE decode kernels with PDL (`0756`).
9. **Opt-In `VLLM_DS41_LSA` Shortlist Indexing (`0741`)**:
   - Adds opt-in `VLLM_DS41_LSA=1` BF16 block-8 shortlist indexing while keeping `VLLM_DS41_LSA=0` default-off bit-identical.
10. **CopyDrafter Caller-Stream Preparation, Continuation Priority, Warp Pruning & Expected-Yield Gating (`0744..0745`)**:
    - Runs CopyDrafter preparation on the caller stream (`0744`), ranks longest-level matches by available continuation length (`1..lookahead`) before recency, prunes shorter-level backward suffix scans via warp-minimum `need_len` and 2-token anchor probes (`28–90x` faster on 1M-token contexts), and gates copy replacement against DSpark's expected yield (`offered * estimate >= sum(dspark_confidences[r, :block])` in `0745`).
11. **DSpark Online Confidence Calibration & Adaptive Verification Alignment (`0746`)**:
    - Calibrates DSpark confidence head probabilities online via identity-initialized logit-space Platt scaling (`OnlineAcceptanceEstimator.for_confidence_head`), feeds calibrated survival into `select_verification_counts`, and aligns the `c1..c2` threshold to `0.06335` (`43.67x` survival MAE reduction, `+1.21%` expected tokens/step cost on heterogeneous `c8`).
12. **Cross-Topology TP2..TP8 Parity & Selective 12-Bit `pack12` Compressor (`0747..0748`)**:
    - Extends column-sharded compressor `fused_wkv_wgate`, `fused_wqa_wkv` decode tiles, and `_trim_padded_groups` across `TP2..TP8` (`0747`), and packs lossless 4-zero-nibble BF16 compressor weights to 12-bit (`0748`, `2.38x` total TP3 compressor speedup).
13. **Indexer `wq_b` Decode RoPE + MXFP4 Epilogue Fusion (`0749`)**:
    - Fuses decode RoPE rotation and scale-matched MXFP4 quantization into `wq_b`'s 256-thread TileLang GEMM epilogue (`100% torch.equal`, eliminating 42 `quantize_mxfp4` launches/step and saving `-0.11 to -0.24 ms` per captured 42-layer step).
14. **Multi-CTA Disk Engram Step, Packed Rows, Negative-Delta Guard & Full-Batch Cache (`0750`)**:
    - Fuses disk Engram wait, FP8 gather/dequantize, and release into a single multi-CTA kernel (`2.14x–2.41x` Eager, `1.07x–1.18x` CUDA Graph), wires 4 KiB-aligned single-plane packed rows with `col_delta[c] < 0` overflow validation and startup telemetry, and keeps the 17.3 MiB 4-way user-space row cache active across all batch sizes.
15. **Deep-Context `>=256K` Chunk Step-Down & `>=1.0s` Stall Decode Interleave (`0751`)**:
    - Preserves full `long_prefill_token_threshold` chunks below `256K` context and on solo prefills while stepping down onto the `2,048`-token grid at `>=256K` and interleaving `K=2` pure decode steps once accumulated prefill stall reaches `>=1.0s` (`-56% to -82%` deep-context decode ITL).
16. **Row-Parallel Vision (`wo`/`w2`), Multi-Image Packing, JPEG `Image.draft()` Downscaling & `{"image": 16}` Cap (`0758` + `1107/{tp3,tp4}.json`)**:
    - Switches vision `attn.wo` and `mlp.w2` to row-parallel `all_reduce`, packs multi-image encoders into a single flash-attn launch, adds `Image.draft()` decode-time downscaling (`7.77x` at 4K), and raises `--limit-mm-per-prompt` to `{"image": 16}`.
17. **IKET Runtime Loop & Phase Profiler (`0757`) + TileLang Multi-Warp TMA Producer Rendezvous (`tilelang/0013`)**:
    - Adds `tilelang/0013-rendezvous-multi-warp-tma-producers.patch` so multi-warp TMA producer groups synchronize at `mbarrier.arrive_expect_tx` (`0` hangs across 2-boot 10-suite release gate and 2-boot IKET verification in `experiments/2026-10-26-spark-ds41f-candidate/1107/receipts/`).
18. **TP3 6-Channel Bidirectional 8 MiB NCCL Ring (`1107/tp3.json` + `0759..0760`)**:
    - Configures `NCCL_ALGO=Ring`, `NCCL_MIN_NCHANNELS=6`, `NCCL_MAX_NCHANNELS=6`, `NCCL_BUFFSIZE=8388608`, `NCCL_P2P_NET_CHUNKSIZE=262144`, `NCCL_SWITCHLESS_BIDIRECTIONAL=1`, `NCCL_MIN_TRAFFIC_PER_CHANNEL=512`, and `NCCL_THREAD_THRESHOLDS="-2 -2 -2 1 1 1"` (`-16.1%` 42-layer collective time per step on TP3).
19. **Startup Allocator & Host Heap Flush Across Preparation, Profiling, Warmup and Capture (`0761`)**:
    - Calls `gc.collect()`, `torch.cuda.empty_cache()`, and `libc.malloc_trim(0)` after TileLang kernel preparation, after memory profiling, before CUDA graph capture, and between backbone and DSpark speculator graph capture. Verified across 2 cold/warm TP3 boots (`sp5..sp7`): head `MemAvailable` stays `>= 6.03 GiB` (`Boot 1: min=6.45 GiB, ready=6.68 GiB`; `Boot 2: min=6.03 GiB, ready=6.60 GiB`) and workers stay `>= 7.94 GiB` (`+1.2–1.6 GiB` headroom above the `5.0 GiB` startup guard, `0` swap growth).

## Verification

```bash
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/1107/tp3.json doctor
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/1107/tp4.json doctor
python3 scripts/release_tests.py check /path/to/vllm a75357bba81e31e4f3085700f94b6a0b29aed44a HEAD experiments/2026-10-26-spark-ds41f-candidate/1107/release-tests.json
python3 -m unittest discover -s tests -p "test_*.py"
```
