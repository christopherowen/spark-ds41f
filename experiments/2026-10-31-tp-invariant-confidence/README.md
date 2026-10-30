# TP-Invariant Confidence Calculation & Elision of Per-Step Rank-0 Broadcast

Addresses `TODO.md:414-419`:
- Eliminates the per-step ungraphed `tp_group.broadcast(self._confidence_probs, src=0)` collective in `AdaptiveVerificationManager.record_confidences` when both the confidence head (`tp_invariant = True`) and the TP communicator (`rank_order_reduce_scatter = True`) are deterministic across TP ranks.
- Updates `DSparkSpeculator` (`_sample_sequential` and `_sample_sequential_topk`) to pass `out=self.draft_token_confidence_probs[:num_reqs].view(-1)` directly into `compute_confidence`, writing probabilities in-place into the contiguous slot slice without an intermediate tensor allocation + `.copy_()`.

## Architecture & Bitwise Determinism Audit

1. **Deterministic TP Collective Reductions**:
   - On `DSparkDeepseekV4ForCausalLM` with `rank_order_reduce_scatter = True` (our RoCE collective policy), all target and draft TP reductions execute in fixed rank order (`SparknetOneShotAllReduce` up to 204 rows; `all_reduce_rank_order` / `reduce_scatter_rank_order` above 204 rows).
   - Consequently, `head_hidden`, `markov_embed`, and drafted token IDs are bitwise-identical across all TP ranks at every draft step.
2. **Deterministic Per-Row Confidence Projection**:
   - `TileLangConfidenceHead` (`dspark_confidence_kernel` in `vllm/models/deepseek_v4_1/tilelang/confidence.py`) computes each row independently using a fixed per-thread 8-element accumulation chain, fixed warp butterfly reduction (`stride = 16, 8, 4, 2, 1`), and fixed 4-warp shared-memory tree in FP32, independent of batch size (unlike cuBLAS `F.linear`, which switches between dot and GEMV reduction trees across batch sizes).
   - Marking `TileLangConfidenceHead.tp_invariant = True` (while keeping `DSparkConfidenceHead.tp_invariant = False` for fallback `F.linear` heads) allows `AdaptiveVerificationManager.record_confidences` to skip `tp_group.broadcast(self._confidence_probs, src=0)` when `tp_group.device_communicator.rank_order_reduce_scatter` is active.

## Verification

Tested on `dgx4` (`spark-ds41f:1030`):
- `pytest -q -o addopts="" --noconftest tests/v1/spec_decode/test_adaptive_verification.py tests/kernels/test_deepseek_v41_tilelang_confidence.py`: **51 passed, 1 skipped**.
