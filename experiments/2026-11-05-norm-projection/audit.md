# Current source boundary audit

Candidate-38 vLLM `1f33102547e4` (`c061e5f69313`), pinned by the experiment.
Candidate-36 in this base is `6d4e49d1cdbb` (`cc0fdf582d0bd`). The four
norm/front/linear/router modules are unchanged between those candidate heads.
The shared working checkout separately names an uncommitted candidate-36 tree
`bc7c9cd5e022`; it is evidence of ongoing work, not this experiment's input.

- `tilelang/norm.py:get_norm_forward_and_per_token_cast_kernel_cuda` has a
  row/batch grid. Each row's threads reduce its sum of squares once. It emits
  BF16 values before E4M3/UE8M0 conversion. There is no output-column GEMM grid
  around the norm reduction.
- `tilelang/front.py:attention_front_kernel` launches `ceil(M/2)` Q blocks plus
  `ceil(M/8)` KV blocks. The sets own disjoint latent columns; neither repeats
  the other's norm. Q's cast is reused by Q-B and indexer Q-B in
  `attention.py:_forward` and `_index_query`.
- `tilelang/linear.py:_block32_linear` consumes `MXFP8Rows` from the producer,
  or performs a standalone cast before the GEMM. `gemm.py` reads activation
  values and scales; its projection tiles do not compute RMSNorm. Weight scales
  stay in their 32x32 block grid rather than an expanded per-output-row array.
- `tilelang/router.py:route_partials` folds gate split-K partials in fixed shard
  order, then selects and normalizes routing weights. This is a projection
  epilogue, with no repeated hidden-state RMSNorm. Padding gets no routes;
  non-finite live logits remain an error.
- `tilelang/mhc.py:project_and_finalize_streams` reduces disjoint K shards and
  lets the last block finalize in fixed split order. The small split sums are
  replicated across consuming lanes, but a whole hidden-row reduction is not
  repeated in every projection output tile. The finalizer emits one BF16
  normalized row and its reusable MXFP8 cast.
- `nvidia/dspark.py:project_context` uses `main_norm.forward_with_cast` to
  hand all context KV projections the same normalized values and cast.
- WO-A's input quantization is already in the attention output store, once
  per row; the WO-A epilogue produces WO-B's input cast. Moving quantization
  across either GEMM would change the checkpoint arithmetic and is excluded.

The screen isolates the existing attention-front boundary while retaining
plain projection arithmetic. Kernel results alone cannot qualify a serving
change. Earlier recorded full-step evidence for the attention-front fusion
is `experiments/2026-10-28-candidate-10-levers`: c1 paired decode +0.88%
(margin 0.32%), decode grid 0.44% faster (margin 0.39%), c8 level, with the
widest 12-16-request/depth-6 shapes approximately 1% slower. That older result
is motivation to test the current wide shapes, not current-candidate proof.
