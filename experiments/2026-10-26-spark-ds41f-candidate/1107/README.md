# Candidate 1107

Unpromoted release candidate combining the validated default-enabled `C40`
runtime base (`5f926df6d312`) with the reviewed and strengthened post-`1030` PRs
across the vLLM serving stack and the `C45` TileLang compiler stack.

## Source Identity

| Component | Base Revision | Patch Series | Patch Count | Patch Head | Expected Tree | Patchset SHA-256 |
|---|---|---|---:|---|---|---|
| `vllm` | `19f2c20ed4d64ae22bd009ceacadd85e2a81e354` | `experiments/2026-10-26-spark-ds41f-candidate/1107/vllm/series` (`0001..0730`) | `615` | `c47b54ae471a24b830e3515212b8c0bf5ee14604` | `13e4781d19bc4a18b0d3d4daad7921c465e3355e` | `9ef866d85a989b41354a3ee4536f32ff31740d4e2063e2298a1c656f8c0da8aa` |
| `tilelang` | `2123e412ce2217b324c50c4ec967213906456ac5` | `experiments/2026-11-05-compiler-prepared-rework/tilelang/series` (`0001..0012`) | `12` | `a470d5daf8529fcf2b661f1119c8f4bb1aecc5e2` | `80fcfc4521b06e0c9776bbf1b5419849557e1090` | `2018cea5190861fd1ebaf51a5ed95289debdcba4b2f931fd333af95794a90fff` |
| `sparknet` | `b61660f47a3f5c8f9369f21f92a75313e9b39579` | `experiments/2026-10-24-sparknet-all-to-all/sparknet/series` (`0001..0010`) | `10` | `5ce8c94218f8ce352d36c0d3995f896957e5052f` | `6602287e7f5a60de22fa1533e70e8e73eb7b4713` | `99cd8469f3b490c7361fc54ef481e0621417eb6ab3632cb6614fb55c31e36374` |

## Included PRs and Code-Review Refinements

1. **Base Stack (`C40`, `0001..0689`, 574 patch files)**:
   - Validated `C38` base (`0001..0687`) plus default `CopyDrafter` enablement and safe envelope (`0688`) and lazy top-k scratch-read guard from PR `#878` (`0689`).
2. **TileLang Runtime Closure & Single-Rank Path (`#885`, `#886`, `#887`, `#893`, `#895`, `#912`, `#913`, `#914`, `#915` / `#926`, `0690..0719`)**:
   - Replaces all remaining ATen compute kernels on the single-GPU request path with TileLang or DMA:
     - `0690`, `0703`, `0704`, `0707`, `0712` (`#885`): TileLang `copy_logits` with 4-wide aligned vectorization, column-first traversal, 1024-column FP32 tiles, and pitched 2D `cudaMemcpy2DAsync` bypass for aligned FP32 matrices `>= 8 MiB`.
     - `0691` (`#886`): TileLang 8-D strided `rank_order_strided` reduction replacing `torch.sum` fallbacks.
     - `0692`, `0705`, `0708`, `0711`, `0713`, `0715` (`#887`): Fused `gather_columns_into` and `concat_rank_shards` with magic-multiplier division (`m = (1 << 32) // d + 1`) and 1-warp (`threads=32`) compact blocks for tiny non-vectorized tails.
     - `0693`, `0700` (`#893`): Fused `gather_logits_kernel` for `logits`-mode selected token scores.
     - `0694..0699`, `0701..0702`, `0719` (`#895`): Native `int32`/`int64` strided token-ID score inputs, eliminating `.long().contiguous()` ATen `copy_`/`fill_` launches and using 1-warp shared-memory staging.
     - `0706`, `0709..0710` (`#912`): Fused `prompt_scores_and_ranks` kernel computing prompt log-softmax and token ranks in a single vocabulary pass.
     - `0714`, `0716` (`#913`): Non-blocking 1-D span `copy_` in `LogprobsTensors.to_cpu_nonblocking`, eliminating per-step GPU slice materialization when `max_num_logprobs == 0`.
     - `0717` (`#915`): Single-rank fused greedy Markov drafting in `DSparkDraftModel`.
     - `0718` (`#914`): Native `int32`/`int64` TP1 token IDs in `VocabParallelEmbedding` without `.int()` cast kernels.
3. **Serving Contract & Dead-Code Cleanup (`#894`, `0720..0722`)**:
   - Enforces the sequence-parallel MoE contract (`tp_size > 1` with `use_sequence_parallel_moe=False` rejected), removes dead `VLLM_DS41_ENGRAM_OVERLAP`, `VLLM_DS41_MARKOV_NVFP4`, and `VLLM_DS41_DRAFT_NVFP4_HEAD` fallback branches, and records capability `ds41-serving-contract`.
4. **Frontend Request & Utility Lifetime Cleanup + Worker Test Isolation (`#890` + `C46`, `0723..0724`, `0730`)**:
   - Rolls back partially registered `OutputProcessor` / `CoreClient` state on failed `generate()` submission (`0723`) and pops abandoned `call_utility_async` futures on cancellation or send failure (`0724`).
   - **Review refinement (`0730`)**: Adding `#890`'s two new test files shifted pytest worker grouping so `test_gpu_tilelang_logprob_softmax.py` ran in the same process before `test_gpu_verified_logprob_token_ids.py::test_tilelang_worker_modules_hold_no_live_triton_reference_kernels`. Patching `logprob._topk_log_softmax_kernel` via `monkeypatch.setattr` resolved `logprob.__getattr__` and restored the lazy Triton `JITFunction` into `logprob.__dict__` on teardown. `0730-isolate-logprob-triton-monkeypatch.patch` uses `patch.setitem(logprob.__dict__, "_topk_log_softmax_kernel", None)` inside `monkeypatch.context()` and asserts `"_topk_log_softmax_kernel" not in logprob.__dict__`.
5. **HTTP Disconnect & Repeated Cancellation Ownership (`#918`, `0725..0726`)**:
   - Cancels and awaits both `handler_task` and `cancellation_task` inside a shielded `CancelScope`, closes abandoned streaming generators (`aclose()`), and keeps `track_load` wrapping `with_cancellation`.
   - **Review refinement (`0726`)**: Replaces `await asyncio.gather(*tasks, return_exceptions=True)` in `finally:` with an `asyncio.wait(tasks)` loop that absorbs repeated `asyncio.CancelledError` interrupts until child cleanup finishes, then retrieves task exceptions and re-raises the outer `CancelledError`.
6. **CED Encoder Prefix Endpoint Retention (`#923`, `0727`)**:
   - Retains `(resend, request.num_prompt_tokens)` in `KVCacheCoordinator._get_retention_endpoints` when `not self.eagle_group_ids and self.prefill_replay_tokens > 1`, preventing premature truncation of completed 256-token encoder prefix pages when `(num_prompt_tokens % 256) < 128`.
7. **Native-Depth Verification CUDA Graph Shapes (`#921`, `0728..0729`)**:
   - Threads `uniform_query_len = 6` (`vllm_config.uniform_decode_query_len`) into `dense_varlen_decode_shapes`, `dspark.py` (`_context_capture_sizes`), and `b12x_prepare.py` (`_planned_decode_counts`). Restores exact `(num_reqs, 6 * num_reqs)` target, context, and warmup CUDA graph shapes (`36` rows at `c6`, `42` rows at `c7`) with `+9` target graphs on TP3 (`60 -> 69`) instead of the `shape-union` alternative (`89` on TP3 / `169` on TP4).
8. **TileLang Compiler Refresh & Prepared-Kernel Fix (`#875`, `#879..#883`, `C45`, `tilelang/0001..0012`)**:
   - Refreshes TileLang to `2123e412ce22` with retained SM120/SM121 features (`0001..0005`), correctness fixes (`0006..0010`), explicit `prepare()` compilation when `is_cache_enabled()` is false before `freeze()` (`0011`, fixing `C42`'s IKET failure), and snapshot `VERSION` namespacing (`0012`).

## Excluded Candidates and PRs

- **`C41` (`experiments/2026-11-04-copy-overlap-rework`)**: Excluded per `decision.md` after two-boot hardware measurement showed engine-grid cost regressions (`+0.206%` to `+0.409%`, worse in 7/8 views) and cold 524K prefill regressions (`-0.181%` to `-0.667%`, worse in 4/8 views) relative to `C40`.
- **Unqualified / Active Research Drafts (`#877`, `#916`, `#917`, `#919`, `#920`, `#922`, `#924`, `#925`)**: Excluded until their independent correctness and two-boot serving qualifications complete.

## Verification

```bash
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/1107/tp3.json doctor
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/1107/tp4.json doctor
python3 scripts/release_tests.py check /path/to/vllm 19f2c20ed4d64ae22bd009ceacadd85e2a81e354 HEAD experiments/2026-10-26-spark-ds41f-candidate/1107/release-tests.json
python3 -m unittest tests.test_source_locks tests.test_recipe tests.test_build tests.test_doctor
```
