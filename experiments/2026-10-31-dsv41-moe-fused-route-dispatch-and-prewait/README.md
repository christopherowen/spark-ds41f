# DeepSeek V4.1 TileLang MoE: Fused Decode `route_partials` + `moe_dispatch_v41`, Pre-Wait Metadata Hoisting & 2-Stage Decode Down GEMM (Part 2 Item 4 + Best of PRs #792 & #803)

## Motivation
Across every decode step (58 routed MoE layers + 1 DSpark draft MoE layer = 59 routed MoE layers):
1. **Fused Decode Router Top-K + MoE Dispatch (`route_and_dispatch_partials`, Part 2 Item 4)**:
   - Previously, `TileLangGateRouter` ran `partials_kernel` (`bf16_gemm_partials`), then launched `route_partials` to write `topk_ids[rows, topk]` and `topk_weights[rows, topk]` to global memory, and then `MoeDispatchV41Plan` launched a 1-CTA `moe_dispatch_v41` (`512` threads = `16` warps) to read `topk_ids` and `topk_weights` back from global memory and write the contiguous layout tensors (`order`, `layout_weights`, `route_pos`, `tile_expert`, `expert_count`, `expert_offset`, `psum`, `num_tiles`).
   - Because `moe_dispatch_v41` is already a 1-CTA kernel of 16 warps and `route_partials` assigns 1 warp per token, `route_and_dispatch_partials` fuses both kernels into a single 1-CTA launch for decode batches (`rows <= 32`): Phase 1 reduces the split-K gate partials and computes the `sqrt(softplus(x))` top-k routing into shared memory (`topk_idx_s`, `topk_weights_s`, reusing `warp_count` for unbiased scores so total shared memory stays at 29 KiB), and Phase 2 immediately runs `rank_routes`, `layout`, and scatter in the same CTA.
   - This eliminates **59 kernel launches per step**, 118 `torch.empty` allocations per step, and the global memory round-trip of `topk_ids` and `topk_weights`.
2. **Pre-Wait Dispatch Metadata Hoisting in `moe_swiglu_cast` and FC2 `grouped_fp8_fp4_gemm` (Best of PR #803)**:
   - Hoists `TileExpert`, `Order`, and `Weights` loads before `pdl.chain()` in `moe_swiglu_cast`, and hoists `TileExpert`, `Order`, `SFBase`, and `Narrow` loads before `pdl.wait()` / `pdl.chain()` in FC2 (`grouped_fp8_fp4_gemm(..., gather=False)`), hiding dispatch metadata latency behind the predecessor kernel's tail.
3. **Deferred Barrier Initialization & 2-Stage Decode Down GEMM (Best of PR #792)**:
   - Moves `loaded = T.alloc_barrier([threads] * stages)` into a child `with T.sblock("gemm"):` after `if expert < 0: T.thread_return()` so empty CTAs return without initializing barriers, issuing cluster barrier fences, or calling `__syncthreads()`, while active CTAs share a single `__syncthreads()` between `rows_shared` setup and barrier initialization.
   - Uses `down_num_stages = 2` (`2 CTAs/SM`) for decode tile height (`block_M = 32`) on the `down` (`FC2`) grouped GEMM.

## Verification
Verified on `dgx4` (`spark-ds41f:19f2c20ed4d6-next-21`):
- `pytest --noconftest -p tests.kernels.deepseek_v41_pdl_sources tests/kernels/moe/test_deepseek_v41_tilelang_router.py tests/kernels/moe/test_deepseek_v41_tilelang_moe.py tests/kernels/test_deepseek_v41_tilelang_pdl.py` -> **203 passed in 17.20s** (`PDL sources: 139 kernels launched, 118 with PDL, 0 violations`).
