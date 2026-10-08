# CED decoder routing mask (r6b candidate)

Base: `origin/main` `e905613` and r6a, the simplified DSpark series the owner
accepted on 2026-10-08 (`experiments/2026-10-07-dspark-verification`, image
`vllm-ds41f-kkref:04c30fa98e79-r6a`, vLLM tree `2c2f9d6d`). One vLLM patch is
added; nothing else changes.

The fix was found and written by an r6 user serving real gateway traffic on
three GB10s (TP3 behind a switch), contributed in
[#15](https://github.com/christopherowen/spark-ds41f/pull/15) and merged with
its experiment, [2026-10-05-ced-decoder-routing-mask](../2026-10-05-ced-decoder-routing-mask/).
This candidate applies that patch file in place, so the built vLLM commit keeps
its author. The only edit to the file is the removal of a tool co-author
trailer, which this repository does not carry.

## Problem

With the TileLang family, the engine stops at the TileKernels router's NaN/Inf
assertion at the first **mixed step**: a request whose prefill takes the CED
path (more than 128 uncached tokens) is admitted while other requests decode
with adaptive verification.

A CED decoder layer routes the step's rows gathered by CED's indices. 0035's
`_routing_mask(rows)` slices the step's padding mask to the decoder's row count
instead. That is only right when padding sits at the end of the batch and the
plan is exact. In a mixed step neither holds:

- 0010 marks dead verification rows inside the batch, not only at its end;
- the CED plan is sized from the CPU's scheduled-token upper bounds, so when
  verification shortens the decode ranges the compact batch ends in `-1`
  indices.

Padding rows get routed (their NaN logits trip the assertion) and live rows get
skipped. The B12X family does not run this code.

**It applies to the promoted recipes.** `VLLM_MOE_SKIP_PADDING` defaults to 1,
both recipes run `VLLM_DS41_KERNEL_BACKEND=tilelang` with the dead-row cut
(`SPARK3_DSPARK_DEAD_ROWS_TAU=0.2`), and the production agent workload admits
prefix-cached prompts with uncached tails while another request decodes. The
benchmark never mixes prefill and decode in one step, which is why it never
failed there.

## Intended delta

[0045](../2026-10-05-ced-decoder-routing-mask/vllm/0045-tilelang-ced-decoder-routing-mask.patch), from #15, on top of
r6a's series ([vllm/series](vllm/series)):

- `DeepseekV41ModelState.prepare_attn` gathers the step's padding mask by the
  CED indices (`-1` counts as padding; `gather_rows` zeroes invalid indices)
  and puts it on the decoder metadata, once per step;
- decoder-layer routers are tagged at construction (`attn.is_ced_decoder`) and
  take that mask; encoder and drafter routers keep the step's. Row count alone
  never selects the compact mask, because an encoder forward can have the same
  number of rows;
- the mask is a new tensor each step, so a CED step that would replay a CUDA
  graph raises instead of reading a stale mask.

Applying the series with the build's `git am` flags reproduces r6a's recorded
patch head `709a159a` and tree `2c2f9d6d`, then gives patch head `1ba3f01d`
and tree `668e3aba` with 0045. `git diff --check` is clean and every changed
file compiles.

## Review

- **Correct where it changes anything.** With no dead rows and an exact plan,
  the gathered mask equals the old slice, so pure prefill steps and decode-only
  steps route exactly as before. Only mixed steps change, and only where the
  old mask was wrong.
- **The CUDA-graph guard cannot fire in the promoted recipes.** A CED plan
  exists only when a request schedules more than 128 tokens. Graph capture
  stops at 96 tokens (TP4) and 48 (TP3), and the single-request prefill graph
  needs `VLLM_USE_BREAKABLE_CUDAGRAPH`, which both recipes set to 0. Graph
  capture stages CED with every request kept full, so no plan exists then
  either. A recipe that captured more than 128 tokens would fail loudly at the
  first CED step rather than route with a stale mask.
- **Cost.** One small gather per CED step (prefill steps only, under 1% of
  engine steps in the agent workload). Decode steps are unchanged.
- **Scope.** TileLang routers only. The guard in `prepare_attn` runs for both
  families but only matters when a graph would replay a CED step.

## Quality and safety gates

1. Build `-r6b` from this lock on an idle Spark
   (`bin/spark --cluster-config experiments/2026-10-08-ced-routing-mask/r6b-tp4.json build prepare`,
   then `build image`) and load it on all four nodes with one image ID.
2. In the image, with `--noconftest`:
   `tests/kernels/moe/test_deepseek_v41_tilelang_router.py` (the patch's two new
   GPU tests have only run as a standalone script so far) and r6a's
   `test_adaptive_verification.py` and `test_dead_rows.py`.
3. One lab window on TP4:
   - `base` (r6a): [repro_mixed_step.py](repro_mixed_step.py) for three rounds.
     The expected result is a failure at the first mixed step; if r6a passes,
     record that and continue, since the fix is still correct by review.
   - `candidate` (r6b): the same script must pass all three rounds with the
     server healthy afterwards;
   - then r6's acceptance benchmark on [r6b-tp4.json](r6b-tp4.json)
     (quality, decode 1-16 streams, prefill 32K-1M) against r6a's TP4 run.
4. TP3 needs the triangle cabling; [r6b-tp3.json](r6b-tp3.json) is ready for it.

Rollback trigger: any quality failure, a failed request, or a decode point
slower than r6a with an interval that excludes zero.

## Acceptance criteria

The mixed-step reproduction passes on r6b, and serving is level with r6a
(decode and prefill intervals include zero or favour r6b). This is a
correctness fix; it does not need to be faster.

## Status

Gate run on the TP4 ring, 2026-10-08 13:22-13:56 UTC (window
`dgx1-1791465750`):

- **Build.** 9 minutes on dgx4; one image ID on all four nodes
  (`sha256:6371971f`).
- **Tests in the image.** 73 passed, 1 skipped, 2 failed. Both failures are
  `test_padding_rows_are_not_routed` raising `AttributeError`: that fixture's
  bare forward context has no `attn_metadata`, which 0045 now reads (a real
  `ForwardContext` always has it). 0045's own tests passed. The test-only fix
  is 0046 in the r6c candidate ([2026-10-08-lil-rebase](../2026-10-08-lil-rebase/)).
- **Mixed-step reproduction.** r6a passed as well, so the first script never
  reached the failing step: its streams ended at end of sequence after 7-55
  chunks and each mixed request returned a single chunk. r6b passed three
  rounds with the server healthy. The script now keeps eight streams decoding
  with `ignore_eos`, staggers four prefix-cached requests per round, and fails
  a round in which a mixed prefill did not overlap a decoding stream; r6c's
  window runs it on r6a first.
- **Quality** 5/5.
- **Against r6a's TP4 run, same protocol.** Every decode step-time interval
  includes zero except prose-c4 (-0.3%, [-0.7, -0.0]). Prefill at 32K, 262K,
  500K and 1M tokens: -0.1, -0.2, -0.3 and -0.2%, intervals including zero.
  No failed requests; minimum MemAvailable 19.8 GiB.

Meets the acceptance criteria. Needs the owner's acceptance; the r6c candidate
carries it onto the current upstream heads.
