# CED decoder routing mask under TileLang (r6)

## Problem

With r6 (TileLang kernel family), the engine stops at the TileKernels router's
NaN/Inf assertion at the first **mixed step**: a request whose prefill takes the
CED path is admitted while other requests decode with adaptive verification.

A CED decoder layer routes the step's rows **gathered by CED's indices**, but
[0035](../../patches/vllm/0035-tilelang-routing-skips-padding.patch)'s
`_routing_mask(rows)` takes the step's padding mask and slices it to the
decoder's row count. That is only right when padding sits at the end of the
batch and the plan is exact. In a mixed step neither holds:

- adaptive verification marks dead verification rows **inside** the batch
  ([0010](../../patches/vllm/0010-dspark-dead-verification-rows.patch));
- the CED plan is sized from the CPU's scheduled-token upper bounds, so when
  verification shortens the decode ranges the compact batch ends in `-1`
  indices.

The router then routes padding rows (NaN logits: the assertion fires) and
skips live ones. The B12X family does not run this code.

## Intended delta

One vLLM patch, [0045](vllm/0045-tilelang-ced-decoder-routing-mask.patch),
on top of r6's series (it applies cleanly to tree `20c7c758`, patch head
`cf7703fb`):

- `DeepseekV41ModelState.prepare_attn` gathers the step's padding mask by the
  CED indices (`-1` counts as padding; `gather_rows` already zeroes invalid
  indices) and puts it on the decoder metadata, once per step;
- the decoder layers' routers are tagged at construction
  (`attn.is_ced_decoder`) and take that mask; encoder and drafter routers keep
  the step's. A compact mask is never chosen by row count alone, since an
  encoder forward can have the same number of rows;
- the mask is a new tensor each step, so a CED step that would replay a CUDA
  graph raises (CED needs more rows than the capture sizes, so it runs eagerly).

The series, lock and source manifest are left as they are, so CI's fingerprint
checks still hold: adding 0045 to the series and building a candidate is the
owner's call through the usual promotion flow.

## Reproduction (3 × GB10, TP3, r6, 4 KiB profile)

Five requests decoding with DSpark (5 draft tokens, adaptive verification) when
a sixth is admitted whose prompt has 3,328 prefix-cached tokens and a 331-token
uncached tail: the engine fails at the first mixed step. Reproduced with
2.0 GiB and 2.2 GiB of KV, and with prefix caching off; minimum MemAvailable
stayed above 6.9 GiB with no swap, so it is not memory.

## Results with 0045 (applied as a runtime patch to the r6 image)

- GPU regression in the r6 image on an idle node, before loading weights: the
  158-row plan for 150 live rows with three interior dead rows gathers the
  expected mask; truncation does not; encoder and decoder masks of equal size
  stay distinct; NaN logits in the masked rows give expert `-1` and weight 0
  through the real TileKernels router.
- Integration, prefix caching on, the recipe's `FULL_AND_PIECEWISE` graphs
  unchanged: two rounds of six concurrent streams, **24/24 complete
  responses**, prefix hits on every round (36,608 tokens per round), no
  preemption, NaN/Inf or swap.
- Quality after the change: 5/5 coherence checks, tool calling, and a real
  295,000-token prompt with three needles at 10/50/90 % found 3/3
  (68.0 s); minimum MemAvailable 6.85 / 7.89 / 7.81 GiB.

The pytest port of the GPU regression
(`tests/kernels/moe/test_deepseek_v41_tilelang_router.py`, inside the patch)
compiles but has not been run as pytest; the same assertions passed as a
standalone script in the r6 image.
