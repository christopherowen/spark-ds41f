# Fusion candidates for DeepSeek V4.1 on GB10: what to fuse, where, and what it can win

Owner, 2026-10-25: optimize the small kernels before fusing (a separate session audits each
kernel for unnecessary work), and look at the fused-kernel designs for ideas on what to fuse
and where. The rules for every change that follows: bound its gain from our own traces before
building it; a port is bit-identical or matches DeepSeek's reference semantics, and is at least
as fast; correctness before speed; no fallback paths.

This is a planning record. Nothing was built, and no GPU or cluster time was used. It bounds
every candidate fusion and overlap in our serving path from our own traces and ranks them. The
review of the published implementations the candidates were drawn from is kept out of this
repository, as AGENTS.md requires.

Inputs:
- vLLM fork `tilelang-ports` at 26382f5cf, the frozen tree of the release now qualifying, read
  only (`vllm/models/deepseek_v4_1/tilelang/*.py`);
- rank 0's profiled traces of
  [request-inventory-5](../2026-10-25-request-inventory/README.md) on the TP4 ring
  (`results/private/determinism/request-inventory-5/ports/{decode,prefill}-trace/dgx1/`):
  one window per request type at one stream, eight mixed requests, a 15,395-token prompt and an
  image request.

The traces were taken on the ports build of the sixth serving check (vLLM 9554a406c's
package). The frozen tree ports more of the sampler and runner glue after it, so the figures
for the head, sampling and step preparation are upper bounds for 26382f5cf; the layers did not
change.

## The short answer

- **Decode on GB10 is a bandwidth problem, and fusion does not move weight bytes.** The routed
  MoE's expert GEMMs are 41% of a one-stream greedy step and 59% of an eight-request step; the
  dense GEMMs, the two LM heads and the all-reduce waits take most of the rest. A fusion saves
  tens of KiB of activations a token a layer, against about 39 MB of attention weights a layer.
  On our hardware a fusion is worth what the kernel boundaries it removes cost: launch gaps,
  fill and drain, serialized latency-bound kernels.
- **The budget for all layer fusions together is 12.5% of a c1 greedy step, 15.8% sampled and
  7.5% at c8.** That is every kernel in the 40 target layers other than a weight GEMM, the
  attention core or an all-reduce, plus the idle gaps between them: what a perfect per-layer
  megakernel would save. No single decode fusion's ceiling is above about 4% of the step.
- **The largest decode ceilings** (c1 greedy / c1 sampled / c8):
  the mHC boundary restructure: finalize emitting MXFP8, the coefficients on a side stream, the
  post, collapse and norm in the all-reduce's reduce phase (3.1 / 4.0 / 1.4%);
  FC2 streaming its weights at FC1's per-byte rate, through one persistent FC1 and FC2 kernel
  or a retiled FC2 (2.4 / 2.3 / 3.2%: FC2 runs at about 92% of FC1's rate today);
  PDL across the decode graphs with weight loads ahead of the wait (1.9 / 2.4 / 1.1%);
  the attention front's RoPE, cache-write and q/kv norm kernels (2.1 / 2.7 / 1.1%);
  Engram's lookup projection moved ahead on a side stream (1.6 / 2.0 / 1.0%);
  the indexer's and compressor's small kernels (1.6 / 2.1 / 1.0%);
  producers writing the all-reduce's staging buffer (about 1.3 / 1.6 / 1.5%, estimated);
  the attention output chain into WO-B (1.1 / 1.4 / 0.6%);
  one sparse-index table per index source (0.8 / 1.3 / 0.4%).
  The routed MoE chain's epilogue fusions (SwiGLU into FC1, the combine into FC2, PDL between
  them) are 0.9 / 1.0 / 0.8%.
- **Prefill has the larger ceilings, in its communication.** 22.9% of the 15K-token prefill is
  NCCL time with nothing else running. Overlapping the MoE output exchange with the combine
  (6.3%), gathering the MoE input as MXFP8 with the router run on local rows (5.6%) and finer
  slicing of the attention exchange (up to 4.3%) lead, and each is bit-identical by
  construction.
- **Recommended order:** next release, the low-risk set that is bit-identical by construction
  (PDL, the attention front's horizontal fusions, shared index tables, the act-quant fold and
  `evict_first` from the other session, the prefill MXFP8 gather and the combine writing the
  exchange slots); then the mHC coefficients on a side stream, Engram ahead, the WO-A chain, a
  probe of FC2's per-tile cost and the prefill exchange overlap; last, the sparknet producer and
  consumer fusions and the persistent MoE kernel with its epilogue fusions, which touch the
  collective and the expert GEMMs.

## Status (2026-10-26, CET)

Since this record, other experiments have taken up these rows (numbers from their PRs):

| Rows | Covered by | Where it stands |
| --- | --- | --- |
| 1 PDL | #589 | Prototype bit-identical across builds and replays; the c1 layer chain is 1.81% faster warm and 1.30% cold, 24 and 48 rows level. Its same-stream boundary ceiling is 3.70 / 4.79 / 1.71%, above this record's because it counts every graph boundary of the step. The warp-specialized GEMMs keep their launch. The serving screen is queued. |
| 2 act-quant fold, 23 | #587 | Finalize writes the MXFP8 cast: -4.3 us a decode layer at c1, -0.88 ms a TP4 prefill layer; on the next candidate. |
| 7 metadata | #585 | Page tables over each row's live pages (0.70 / 0.41 / 1.19% ceiling) and each builder in one launch; one table per index source is still open. |
| 11 FC2 rate | #598 | FC1 measured at 14.2 us plus 11.41 us an expert (208-256 GB/s); the routed experts' launch fixed costs are 1.24 ms a step. The FC2 retile probe is open. |
| 13 evict-first | #587 | Dropped: ptxas miscompiles the hinted K-tail copy on sm_121a, the trace ceiling is about 0 at c1, and no L2 priority setting has a ceiling worth a window. |
| 17, 18 prefill exchange | #593 | The combine writes the exchange slots and the exchange runs under it, sliced: -1.27 ms a layer at 15K; on the next candidate. WO's slices take the same order. |

Still open: rows 3-6, 8-10, 12, 14 and 19-22. #585 found `fn` hot in L2 when `project_streams` reads it, so row 3's note that the kernel is close to DRAM-bound does not hold; row 3's ceiling stands. Outside this record's list: 64-row prefill tiles for the routed experts (#599, routed MoE -12.8 to -16%), and #598's c1 byte floor, which puts all communication at no more than +3.2% at c1.

## How the ceilings were computed

[ceilings.py](ceilings.py) reads the traces and writes [ceilings.json](ceilings.json):

```sh
python3 -I experiments/2026-10-25-fusion-survey/ceilings.py results/private/determinism/request-inventory-5/ports
```

- A decode step is one target forward (embedding, 40 layers, LM head, sampling, the drafter's
  inputs) plus the drafter forward after it, from one target `vocab_embedding_kernel` to the
  next. The first step of a window (the request's prefill), the sampled window's step 24 (a
  26 ms host stall at the request's end) and the mixed window's first two and last steps (before
  and after all eight requests share the batch) are left out: 11 greedy steps of 34.3 ms, 23
  sampled steps of 27.8 ms, 17 mixed steps of 63.6 ms.
- Every kernel on the critical path gets its **exclusive time** (the time it runs while no other
  kernel does; the L2 prefetch's side stream does not count, the shared expert's stream does)
  and its **gap** (the idle time before it starts).
- **Folding kernel X into a neighbour saves at most excl(X) + gap(X), plus the gap before the
  kernel that follows the chain.** The fused kernel is assumed to cost no more than the kernel
  it keeps. That is an upper bound: an epilogue can slow the GEMM it lands in.
- **FC2 at FC1's per-byte rate:** for the same experts FC2 streams exactly half of FC1's weight
  and scale bytes (5120 x 576 against 1152 x 5120 E2M1 values, one UE8M0 byte per 32). The
  ceiling is FC2's exclusive time less half of FC1's, summed over layers.
- Ceilings overlap. Every fusion's ceiling includes boundary gaps that PDL alone would also
  claim, and the mHC boundary includes the act-quant fold. The ranked table says which ones
  share time.
- The traces were taken with Python stacks on. That slows the host, so eager launches sit 2-5 us
  apart instead of back to back and their gaps are upper bounds; kernels and graph replays are
  not affected (inside the CUDA graphs kernels launch 0.2-0.7 us apart). The all-reduce figures
  are rank 0's view; the other three ranks wait for different stretches.
- Ceilings are percentages of the step (decode) or of the prompt's forward (prefill), at TP4 on
  the ring. TP3 has the same chain with wider per-rank shapes.

## Where a step goes

Exclusive time plus the gap before each kernel, by family, as a share of the step:

| Family | c1 greedy | c1 sampled | c8 mixed |
| --- | ---: | ---: | ---: |
| Routed MoE FC1 + FC2 (target) | 41.4% | 27.6% | 58.7% |
| Dense GEMMs (target: Q-A/KV, Q-B, WO-A, WO-B, shared expert, router, indexer, Engram) | 14.1% | 16.4% | 8.3% |
| One-shot all-reduces (target) | 12.1% | 15.0% | 9.2% |
| LM heads (target verification + drafter) | 6.0% | 7.4% | 3.6% |
| Attention: sparse MLA, RoPE, cache writes, metadata | 5.3% | 7.6% | 4.2% |
| mHC entry and post | 5.3% | 6.8% | 3.3% |
| Norm and activation casts | 1.8% | 2.0% | 0.9% |
| Indexer and compressor | 1.2% | 1.4% | 0.8% |
| MoE small kernels (dispatch, SwiGLU cast, combine, shared SwiGLU) | 0.8% | 0.9% | 0.8% |
| Engram | 0.2% | 0.4% | 0.1% |
| All-gathers (sampling, drafter inputs, Engram) | 1.2% | 1.4% | 0.8% |
| Sampling, verification, draft glue | 0.4% | 0.4% | 1.8% |
| Step preparation and copies | 1.1% | 1.4% | 0.8% |
| Drafter layers (3) | 4.4% | 5.2% | 4.5% |
| Overlapped (two kernels at once) | 4.5% | 6.0% | 2.2% |

The sampled window generates different text from the greedy one; its six rows a step touch
fewer distinct experts (FC1 + FC2 27.6% against 41.4%), so every fixed cost weighs more.

Two weight reads that no fusion removes: the mHC mix projection reads `fn` in FP32 (1.97 MB a
boundary, 157 MB a step) and the router its BF16 gate (3.9 MB a layer, 157 MB a step), together
about 1.15 ms of DRAM time a step at 273 GB/s.

The 15,395-token prefill (one sequence-parallel forward, 2,140 ms on rank 0): routed MoE GEMMs
26.4%, NCCL exchanges and gathers with nothing else running 22.9%, sparse MLA 14.1%, dense GEMMs
12.0%, the MoE combine 5.2%, mHC 6.1%, casts 2.8%, copies 1.5%, metadata 1.2%. Layers 0-20 run
on every row (about 90 ms each); the CED decoder layers after them run on the compacted rows.

## Our chain today (26382f5cf)

One decode layer at one stream (six rows), with typical exclusive times from the greedy window:

- **Attention entry:** `mhc_project_streams` (the previous sublayer's post, the mix
  projection's partials, the collapse with the carried pre-mix; 7-13 us), `mhc_finalize`
  (partials reduced, RMS, mix split, Sinkhorn, the collapse's RMSNorm; 4-6 us), `per_token_cast`
  (1 us).
- **Front:** fused Q-A/KV GEMM (10-48 us, depending on how much of its weight the L2 prefetch has
  loaded), two `norm_forward_and_per_token_cast` (q and kv latents, 1-2 us each), a
  `per_token_cast` (1 us), Q-B (16-60 us), two `rope` (1-5 us), `cache_writer` (2-4 us),
  `metadata_decode` or `metadata_chunk` (2-6 us; per layer).
- **Indexer layers** add a compressor chain (a split-K BF16 GEMM, gather, compress, partials,
  reduce, norm, two RoPEs, MXFP4 quantize, cache write) and an indexer chain (cast, indexer Q-B,
  RoPE, MXFP4 quantize, head-weight partials and reduce, page metadata, score, top-k segments
  and merges): about 150 us on the index sources, 50 us on the reuse layers.
- **Core and output:** `sparse_mla` (11-14 us), its split-KV combine `sparse_mla_1` (2-5 us),
  WO-A split-K partials (13-30 us), `gemm_splitk_reduce_cast` (1-3 us), WO-B (8-37 us), the
  one-shot all-reduce (45-105 us; at this size the 2026-10-07 anatomy measured staging 4.7 us,
  then the wait, then the reduce 8.4 us).
- **MoE:** the FFN entry (`mhc_project_streams`, `mhc_finalize`), then two streams. The routed
  stream runs the router (`gemm_bf16_partials`, `router_partials`), `moe_dispatch`, the routed
  input cast, FC1 (120-370 us), `moe_swiglu_cast` (3 us), FC2 (65-205 us). The shared stream
  runs its own input cast, gate/up, SwiGLU, cast and down (about 25 us, hidden behind FC1).
  `moe_combine` sums the six routes in slot order in FP32 and adds the shared output before one
  rounding; then the all-reduce.

What is already fused: the mHC post with the next entry's projection and collapse
(`project_streams`), the coefficient chain (`finalize`), norm with cast, the route weight with
the SwiGLU clamp and cast, the shared add into the combine, the attention act-quant into the
sparse-MLA store (r6u), and under sequence parallelism the unreduced parts summed inside the
mHC entry (r6d).

Three facts the plan depends on:
- **The mHC pre-mix is carried from the previous sublayer**, as in DeepSeek's reference: the
  entry collapses with the incoming pre-mix and computes the next one (`mhc.py`'s docstring).
  So when an all-reduce's result lands, everything the post, the next collapse and the norm
  need is already known, and the next coefficients can be computed off the critical path.
- **No kernel on the DS4.1 serving path requests PDL at 26382f5cf.** TileKernels' `set_pdl` is
  never called (its default is off). The tree's `model_executor/kernels/mhc/tilelang_kernels.py`
  does use `T.pdl_sync`, but the model builds `TileKernelsMHC` from
  `deepseek_v4_1/tilelang/mhc.py` (`kernels.mhc`), which has none. The pinned TileLang provides
  `T.pdl_sync()`/`T.pdl_trigger()` (`tilelang/cuda/language/pdl.py`), and its TVM CUDA runtime
  launches a kernel that uses them with programmatic stream serialization
  (`3rdparty/tvm/src/runtime/cuda/cuda_module.cc`). No kernel uses an L2 cache-policy hint
  either; the L2 prefetch's side stream fills the FFN and NEXT windows during the all-reduces.
- **The FFN input is cast to MXFP8 twice** with identical arguments, once by the shared expert's
  `_block32_linear` (`tilelang/linear.py`) and once by `MoeExpertsV41Plan.run`
  (`tilelang/moe.py`). Decode hides the second on the parallel stream; prefill pays it on every
  row (1.1% of the 15K prefill). Passed to the audit session.

## Candidate fusions, ranked

Ceilings are upper bounds in percent of the step (decode: c1 greedy / c1 sampled / c8 mixed)
or of the 15K prefill's forward, from [ceilings.json](ceilings.json) unless marked as an
estimate. Rank weighs the ceiling against risk and effort.

Bit-exactness risk: **none** means no arithmetic changes (launch order, layout, integer work,
the same per-tile math scheduled differently); **low**, the same operations in the same order,
moved between kernels; **medium**, a reduction order or rounding point has to be reproduced
exactly. Nothing here changes DeepSeek's reference arithmetic. Effort: **S** up to two days
with its kernel bundle, **M** about a week, **L** more, or a sparknet wire-format change.

### Decode

| # | Boundary removed | Technique | Ceiling c1 / sampled / c8 | Prefill | Risk | Effort |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Every kernel boundary in the 40 layers' graphs | PDL: `T.pdl_trigger()` early, weight and scale loads before `T.pdl_sync()`; check that the launch attribute is really set and that the captured graph keeps the programmatic edges | 1.94 / 2.41 / 1.14 (gaps alone 1.75 / 1.93 / 0.96) | ~0 (eager) | none | S a family |
| 2 | `mhc_finalize` → `per_token_cast` → Q-A/KV, shared and routed inputs | The act-quant fold: finalize writes E4M3 + UE8M0 in each consumer's scale layout from the BF16 values it rounds | 0.72 / 0.92 / 0.37 | 0.22 (+1.07 for the duplicate cast) | none | S (other session) |
| 3 | The mix projection and Sinkhorn on the critical path | Coefficients on a side stream: with the carried pre-mix they do not depend on the sublayer; an event joins them before the next entry | ≤ 2.01 / 2.59 / 0.91 | ≤ 0.92 | low | M |
| 4 | All-reduce reduce phase → mHC entry | The collective's consumer does the post, the collapse, RMSNorm and the cast, loading streams, coefficients and γ before it polls | 0.40 / 0.55 / 0.16 measured gap, plus an estimated ≤ 0.5 of the reduce's write and tail | - | medium | L |
| 5 | WO-B and the combine → the all-reduce's staging copy | Producers publish: the WO-B epilogue and the combine store BF16 straight into the registered send buffer; the last CTA rings the doorbell | about 1.3 / 1.6 / 1.5 (estimate: staging 4.7 us an op at 60 KB, 2026-10-07; c8 extrapolated) | - | none | L |
| 6 | Two RoPEs and the cache write; the q/kv norms and their casts | Horizontal fusion of per-row work: one kernel for q RoPE, k RoPE and the cache write, one for both norms and their casts | 1.45 / 2.08 / 0.74, plus 0.65 / 0.60 / 0.32 | about 0.4 | none to low | S-M |
| 7 | Per-layer `metadata_decode`, `metadata_chunk`, `metadata_pages` | One sparse-index table per index source a step, reused by its consumer layers | 0.83 / 1.27 / 0.42 | 0.89 | none | S-M (with the audit session) |
| 8 | Engram rows → gather → cast → `wkv` GEMM → gather, on layers 1 and 14 | It depends only on token ids: run it on a side stream from the step's start | 1.64 / 1.95 / 0.97 | small | none | M |
| 9 | 17-23 small kernels of each index-source layer's compressor and indexer | Compression with norm, RoPE, quantization and cache write in one kernel; the indexer query's RoPE and MXFP4 quantization in its Q-B epilogue; fewer top-k merge passes | 1.64 / 2.08 / 1.00 | 0.7 | low to medium (top-k ties and merge order) | M-L |
| 10 | Split-KV combine → WO-A → split-K reduce + cast → WO-B | The combine in WO-A's prologue, the reduce and cast in WO-B's prologue, WO-A's weights loaded before the PDL wait | 1.12 / 1.43 / 0.58 | ~0 | low to medium (split-K order, z's BF16 round) | M |
| 11 | FC1 → FC2, and FC2's short K (576, 4.5 steps of 128) | One persistent FC1 and FC2 kernel: a weight-producer warp streams continuously across tiles and phases, FC2 tiles wait on per-tile FC1 completion counters, FC1 tiles are claimed first so the 48 SMs cannot all wait on unscheduled FC1 work; or first, FC2 retiled so it does not pay fill and drain per short tile | 2.41 / 2.26 / 3.20 (+ drafter 0.14 / 0.16 / 0.17) | - | none (same per-tile math); liveness needs a deadline trap | L (retile probe S-M) |
| 12 | FC1 → SwiGLU cast → FC2 → combine | SwiGLU and its cast in FC1's epilogue (gate and up rows interleaved at load), the combine done by a token's last FC2 tile in slot order, PDL between them | 0.85 / 0.96 / 0.80 | SwiGLU 1.05 | medium | L (with row 11) |
| 13 | (L2 policy, no boundary) | `evict_first` on the streamed expert weights so they do not evict the prefetched windows | ≤ about 2 at c1 (earlier estimate) | - | none | S (other session) |
| 14 | Head → argmax → gather → sampler chain → draft inputs; the drafter's five Markov steps; step preparation | Argmax partials in the head's epilogue, the greedy sampler chain in one kernel, the post-forward glue captured in a graph, the Markov chain in one vocabulary-sharded kernel | target tail 0.62 / 0.70 / 1.48, drafter tail 0.94 / 1.21 / 0.59, eager gaps 0.61 / 0.81 / 0.45 | - | none | M |
| 15 | Router partials → top-k → dispatch | One router kernel | 0.09 / 0.09 / 0.07 | (row 19) | low | - |
| 16 | The shared expert's own GEMMs | The shared expert as extra tiles of the routed GEMMs | 0.27 / 0.31 / 0.16 | - | medium | L |

Rows 2-4 together are the mHC boundary restructure: 3.13 / 4.04 / 1.43% for the measured
parts. Rows 1 and 2-12 share their gap terms, so the ceilings do not add. The set inside the
layers other than row 11 is bounded by the layers' small-kernel budget (12.5 / 15.8 / 7.5%).

Notes on single rows:
- Row 3: the projection reads `fn` in FP32 (0.58 ms a step at 273 GB/s if it came from DRAM; the
  kernel audit later found it hot in L2, see Status). A DRAM-bound kernel on a side stream pays
  only where the main stream is latency-bound (the all-reduce waits). A side-stream
  kernel that shares the SMs with the MoE GEMMs can also run several times slower than alone:
  keep its pipeline shallow and time it beside them.
- Row 8: the same applies to the 180 us `wkv` GEMM.
- Row 11: it assumes FC1's rate is the attainable one; FC1 itself may sit below the DRAM peak,
  which is the kernel audit's question. Retiling FC2 is the cheap probe of how much of the gap
  is per-tile fill and drain.
- Row 12: the earlier estimate (at most 0.7% at c1 and 0.3% at c8) came from another window;
  here the mixed window's SwiGLU and combine take 6 and 5 us a layer. The FC1 epilogue must keep
  `moe_swiglu_cast`'s arithmetic, including its certificate and fallback, over the same
  32-column groups, and the route weight where it is today (before the cast). No epilogue may
  change its arithmetic with the tile size chosen by the row count, or the draft and verify
  passes stop agreeing.
- Long context, which these short prompts do not show: a stream's six verify rows could share
  one pass over the index keys with bit-equal logits. Check whether `indexer_score` reads each
  request's keys once or once a row before planning it.

### Prefill (15,395 tokens, one sequence-parallel forward)

| # | Boundary removed | Technique | Ceiling | Risk | Effort |
| --- | --- | --- | --- | --- | --- |
| 17 | Combine → four copies into the exchange slots → NCCL exchange | The combine writes each destination rank's rows into its slot and posts it when complete, so the exchange runs under the rest of the combine | 6.32 (the exchange hidden under the combine and the copies) | none | M-L |
| 18 | The copies alone | The combine writes the exchange slots | 1.31 | none | S |
| 19 | BF16 all-gather of the MoE input → two identical casts on every row → the router on every row | Cast and route on local rows; gather E4M3 + UE8M0 (52% of the BF16 bytes) with the top-k ids and weights | 5.63 | none if the router GEMM is row-local and batch-invariant, as ours is designed to be | M |
| 20 | The MoE all-gather after the mHC entry | Gather row slices while the entry finishes the next slice | ≤ 3.09 (shares row 19's time) | none | M |
| 21 | The attention exchange left after the WO slices | More WO slices, or the exchange's tail under the next entry | ≤ 4.28 | none | S-M |
| 22 | The all-gather of the q/kv latents (attention layers without a compressor) | Cast the q latent on local rows and gather MXFP8; RoPE and quantize the kv latent on local rows and gather the 528 B cache records | 0.92 | none | M |
| 23 | `mhc_finalize` | Row 3's split, or finalize folded into the entry | 0.92 | low | M |

Rows 7 (0.89%), 12 (SwiGLU, 1.05%) and the duplicate cast (1.07%, inside row 19) apply to
prefill too. The other exposed gathers are unexplained and worth a look before any plan: three
of 7-9 ms after the index layers' top-k (1.13% together), the Engram row gathers (0.64%), and
the forward's last two (5.4 and 20.2 ms, 1.53% with the embedding's), which bring the rows
back to every rank for the head and the drafter's auxiliary states (`sp_prefill.py`).

## Techniques on GB10

| Technique | On GB10 (SM121, 48 SMs, 24 MiB L2, 273 GB/s, RoCE between nodes) |
| --- | --- |
| PDL, with weights loaded before the wait | Available in the pinned TileLang; unused by our kernels. The cheapest broad lever (row 1). |
| Producers emitting FP8 and several scale layouts | Unchanged integer UE8M0 arithmetic (rows 2, 19, 22). |
| Epilogue fusion (SwiGLU in FC1, a token's last tile does the combine, the cast in WO-A) | Works with `mma.sync`; the BF16 rounding points and reduction orders are the work (rows 10, 12). |
| A persistent GEMM pair with a continuous weight stream | Possible on 48 SMs with counter-gated phases and FC1 claimed first; needs a deadline trap for liveness (row 11). |
| A persistent per-layer megakernel with warp roles | 99 KiB of shared memory a block and no TMEM limit it, and the whole layer budget is 12.5% at c1. Not recommended. |
| TMA multicast, clusters and DSMEM | Unverified on SM121; do not plan on them. |
| Communication overlap through peer memory | No NVLink. What transfers is the structure: producers write the send buffer, the consumer works in the reduce, coefficients leave the critical path (rows 3-5); in prefill, the NCCL exchanges overlap compute (rows 17, 20, 21). |
| L2 policies (evict-first on streams, evict-last on re-read state) | Available; measure against the L2 prefetch (row 13). |
| Side streams for independent work | Pays only where the main stream is latency-bound (the all-reduce waits); DRAM-bound side work competes for the same bandwidth (rows 3, 8). |

## Recommended order for the next releases

Each step is one change on its own branch, with its ceiling re-measured on the newest build
first, a kernel bundle that checks it bit for bit against the kernels it replaces, then the
serving screen; keep or drop each on its own measurement.

**Next release (low risk, bit-identical by construction):**
1. The act-quant fold and `evict_first` (rows 2 and 13), already being prototyped.
2. PDL (row 1), one kernel family at a time: the MoE chain, the attention chain, the mHC
   entry. First a one-kernel probe that the TileLang launch sets programmatic stream
   serialization and that the captured graph keeps the programmatic edge: PDL instructions do
   nothing without the launch attribute.
3. The attention front's horizontal fusions (row 6).
4. One index table per index source (row 7), with the audit session.
5. Prefill: the combine writing the exchange slots (row 18) and the MXFP8 gather with the
   router on local rows (row 19), which also removes the duplicate cast.

**The one after:**
6. The mHC coefficients on a side stream (row 3).
7. Engram's projection ahead of its layer (row 8), measured against the L2 prefetch it will
   compete with for DRAM.
8. The WO-A chain (row 10).
9. A retiled FC2 (row 11's probe), with the kernel audit.
10. Prefill: the MoE exchange under the combine (row 17), the attention exchange and the q/kv
    gather (rows 21, 22).

**Then (sparknet and the expert GEMMs):**
11. The producer and consumer fusions around the one-shot all-reduce (rows 4 and 5). sparknet's
    parked `fused-halves` branch (bd0dd23) already has the device-side send and receive
    halves; it was parked for want of a concrete producer and consumer with a measurable
    ceiling, and these are they (together about 2% at c1).
12. The persistent FC1 and FC2 kernel with its epilogue fusions (rows 11 and 12).
13. The indexer and compressor chain (row 9) and the sampler and draft tails (row 14).

Expectations: the next release's set has ceilings that sum to about 6% of a c1 step and 3% at
c8 (part of it counted twice), and about 7% of a long prefill. Expect less: each ceiling
assumes the fused kernel costs no more than the one it keeps.

## Not recommended, and why

- **A persistent per-layer megakernel.** 99 KiB of shared memory, 48 SMs and no TMEM, against a
  whole-layer budget of 12.5% at c1; epilogue-level fusion plus PDL reaches most of it.
- **Two-shot or token-parallel all-reduce at our row counts.** Each adds a dependent network trip
  (6-11 us on the ring). The break-even may be as low as 6-13 rows, so it deserves a probe at two
  to four streams, not a plan.
- **A replicated embedding** to drop the embedding all-reduce: 0.12% at c1 for about 1 GB of KV
  cache a rank.
- **A single router kernel and the shared expert inside the MoE kernels** (rows 15, 16): under
  0.3%.
- **Anything that changes DeepSeek's reference arithmetic**: skipping the BF16 round of the
  attention output, a TF32 mix projection, an approximate Sinkhorn, fast-math SiLU or Sinkhorn,
  a reordered combine, re-encoded expert weights (NVFP4, MXFP8 or W4A16), quantized or
  BF16-accumulated combine and all-reduce wires. Each would be an owner-approved option, not a
  port.
- **NVLink-only mechanisms**: multicast stores, in-switch reductions, symmetric-memory expert
  parallelism, mailboxes that rely on 16-byte store atomicity across the fabric.
- **Swap-AB decode attention** is a kernel-efficiency question, not a fusion: sparse MLA is
  2.6 / 3.4 / 2.9% of the step. It belongs with the kernel audit.

## Sketches

**A. The mHC boundary in three changes.** Today an entry is `project_streams` (post, mix
partials, collapse with the carried pre-mix), `finalize` (reduce, RMS, mix split, Sinkhorn,
the collapse's RMSNorm) and a cast.
1. `finalize` also writes the cast's E4M3 values and UE8M0 scales, computed from the same BF16
   values the cast reads now, in the layout each consumer reads.
2. The coefficient work leaves the main stream. A side-stream kernel reads the updated streams
   (40 KiB a row, L2-resident at decode), forms the mix partials and runs the Sinkhorn for the
   next entry; the main stream keeps the post, the collapse and its norm. An event joins them
   before the next entry. The collapse's sum of squares must reduce in today's order.
3. The post, collapse, norm and cast move into the one-shot all-reduce's reduce phase. Block 0
   alone polls the flags (the wide-read lesson); the other blocks load the streams,
   coefficients and γ before the data lands; peers are summed in FP32 in rank order, then the
   post uses `project_streams`' exact FMA order. Gate: bit for bit against the three kernels on
   captured operator inputs. The norm feeds the router, so a last-bit difference can change
   expert selection.

**B. Producers write the send buffer.** WO-B's epilogue stores each output tile's BF16 into the
registered pinned send slot instead of its output tensor, and a per-launch counter lets the last
CTA write the doorbell the proxy watches, which is the staging phase's job today. The combine
does the same with the rows it already rounds once. On unified memory the send buffer is the
same DRAM, so the saving is the copy and its launch, not the bytes.

**C. The routed experts as one persistent pair.** A fixed grid of 48 CTAs. A producer warp in
each CTA streams expert weight tiles through a shared-memory ring without stopping between
tiles or between FC1 and FC2; the MMA warps consume them. FC1 tiles are claimed first from a
monotonic counter; each finished FC1 tile bumps a per-(expert, row tile) counter, and an FC2
tile starts once the FC1 tiles that produce its K range are done. The SwiGLU and cast run in
FC1's epilogue with today's arithmetic; the last FC2 tile to finish a token sums its six routes
and the shared output in slot order and rounds once. Every wait has a deadline that traps.

**D. Prefill's MoE input and output.** Today: the entry on local rows, a BF16 all-gather of all
rows, a cast for the shared expert, the router on all rows, a second identical cast for the
routed experts, the experts, the combine, four copies into the exchange slots, the exchange.
Proposed: the router's GEMM and top-k and one cast on local rows; one all-gather of E4M3 rows,
UE8M0 scales and the top-k ids and weights; the experts read the gathered rows; the combine
writes each destination's rows into its slot and posts that slot as soon as its rows are
done. Every op on the way is row-local, so each gathered byte is the byte its owner computes
today.

## Coordination

- The `evict_first` and act-quant fold prototype (rows 2 and 13) and the small-kernel audit run
  in their own sessions; nothing here duplicates them. The audit-relevant findings (the
  duplicate FFN input cast, per-layer metadata, the prefill copies, FC2's per-byte rate, no PDL
  anywhere) were passed to that session.
- [docs/inspiration.md](../../docs/inspiration.md) stays a watchlist; this record adds nothing to
  it.

## Files

- [ceilings.py](ceilings.py): the critical-path accounting and every ceiling above.
- [ceilings.json](ceilings.json): its output for the request-inventory-5 traces.
