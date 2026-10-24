# Finishing the TileLang migration, and the road to 100 tokens a second

Owner, 2026-10-24: "I think we should finish the TileLang migration, because it will bring
interesting fusion opportunities with it. I don't like how we rushed through the attention sink
change to triton", "I would like us to gear up for getting 100tps decode without compromising on
quality", and "we can also start preparing the tilelang kernel migrations".

Quality is fixed while speed moves: the checkpoint's native-precision weights, DeepSeek's
reference arithmetic, temperature-0 outputs bit-identical to the previous release, lossless
speculative decoding, and each port at least as fast as the kernel it replaces (the migration
rule). A kernel that never launches is not ported; it leaves with its engine.

## Where a decode step goes (image -next-2, ring, 2026-10-24)

From the torch-profiled decode run `next-budget-decode-profile` (every rank; one stream, 26
steps; sixteen streams on code, 109 steps): [decode-kernels-next-2.json](decode-kernels-next-2.json)
holds every kernel launched (171 distinct), by count and milliseconds summed over the four ranks.
By family, milliseconds a step on one rank (profiled; the TileLang total includes the L2 prefetch's
side stream, about 10 ms at one stream):

| Family | Kernels | One stream | Sixteen streams | Then |
| --- | --- | --- | --- | --- |
| TileLang (ours and TileKernels) | 48 | 46.8 | 78.3 | done |
| Triton (vLLM's runner, sampler, speculative decoding, sink detector) | 20 | 0.071 | 0.177 | port |
| TileKernels CUDA (`mhc_post_fwd`, `mhc_pre_apply_mix_fwd`) | 2 | 0.062 | 0.121 | port |
| NCCL (sequence-parallel prefill's exchanges and gathers; none in a decode forward) | 2 | 0 | 0.821 | see below |
| cuBLAS (gemv, dot) | 3 | 0.004 | 0.005 | port |
| vLLM CUDA (vocabulary embedding) | 1 | 0.005 | 0.009 | port |
| ATen (copies, fills, top-k, sorts, scans, index ops) | 93 | 0.40 | 0.50 | fold into their neighbours or port |

The Triton kernels: `_apply_write_kernel`, `_combine_sampled_and_draft_tokens_kernel`,
`_compute_slot_mappings_kernel`, `_expand_idx_mapping_kernel`, `_finish_argmax_pair`,
`_gather_block_tables_kernel`, `_gather_lookback_kernel`, `_get_num_sampled_and_rejected_kernel`,
`_greedy_rejection_kernel`, `_gumbel_sample_kernel`, `_partial_argmax`, `_partial_argmax_logits`,
`_post_update_kernel`, `_prepare_dflash_inputs_kernel`, `_prepare_pos_seq_lens_kernel`,
`_prepare_prefill_inputs_kernel`, `_select_first_argmax`, `_sink_candidates_kernel`,
`_sink_policy_kernel`, `_zero_kv_blocks_kernel`.

Prefill, the vision tower and the checkpoint loader (B12X's last pieces) need their own profiled
inventory: one window, every request type the deployment serves.

## The families, in order

1. **Sampling and verification** (owner's priority: the sink detector). `_gumbel_sample_kernel`,
   the greedy argmax kernels, `_greedy_rejection_kernel`, the sink candidates and policy, the
   sampled/rejected bookkeeping. One unit: the sink detector's alternates reuse the sampler's
   Gumbel noise, which comes from Triton's Philox generator (`tl.rand`); the TileLang port must
   reproduce that stream bit for bit, or seeded sampled outputs move. Gate: identical outputs at
   temperature 0 and at seeded temperature 1, the sink contract tests (NaN, determinism, batch
   independence, counters), and per call at least Triton's.
2. **The runner's input preparation** (integer kernels): slot mappings, block tables, index
   expansion, lookback, positions and sequence lengths, prefill and DFlash inputs, KV block
   zeroing, the draft write. Bit-identical by construction.
3. **mHC post and pre-mix** in TileLang, from TileKernels' CUDA kernels; bit-identical against
   them, and the first fusion candidate (the post with the next layer's pre, the sink detector's
   column with the residual it reads).
4. **The ring's collectives** (corrected 2026-10-24): a decode forward makes no NCCL call. The
   sixteen-stream profile's NCCL kernels belong to the prefill of its sixteen fresh prompts (one
   sequence-parallel forward of about 1,000 tokens): 83 exchanges a rank (the attention and MoE
   reduce-scatters, the embedding's, the Engram row all-to-all) and 52 all-gathers above the
   one-shot 2 MiB limit. A one-shot all-to-all for ring4 (sparknet branch `ring-alltoall` 4932e13,
   vLLM `ring-alltoall-sparknet` 3a785ee1e; offline only: `make check`, the simulator, the vLLM
   policy tests) estimates about 18 ms less a forward at 1,000 tokens but 10-25% slower than NCCL
   for 20 MiB blocks, because staging does not overlap the wire. Parked until a third slot overlaps
   them (wire ABI 12); then the fabric probe, the GPU suite on ring and triangle, and a
   prefill-weighted serving A/B.
5. **The ATen glue**: each launch traced to its source and folded into the kernel beside it.
6. **Prefill and the vision tower** after their inventory; then B12X retires with its loader.

## Toward 100 tokens a second

One stream on the ring runs at 73.3 tok/s: 32.4 steps a second x 2.26 tokens a step. 100 needs
1.36x. Shares of the main stream (profiled): routed MoE 49%, dense MXFP8 GEMMs 14%, one-shot
all-reduces 11%, small kernels (mHC, quantize, RoPE, elementwise, metadata, cache writes) about 11%,
vocabulary heads 5%. Neither lever reaches 1.36x alone:

- **A faster step** (the migration, then fusion): the MoE expert pipeline at six rows, the
  all-reduce into its producer and consumer, the small kernels into their neighbours, the greedy
  argmax into the head's epilogue. A plausible 15-25% gives 86-97 tok/s.
- **More tokens a step** (lossless): suffix drafting on agent traffic (+7-17% at one stream,
  experiments/2026-10-23-next-release, copy probe) and draft trees (ceiling to measure).

First, measure before building (one window): the prefill/vision inventory; each kernel's achieved
bandwidth against GB10's roofline at one and sixteen streams (where the MoE GEMMs stand matters
most); the draft-tree acceptance ceiling from logged drafter distributions. The target is the
release suite's single stream on the ring at temperature 0, tracked at temperature 1 and on an
agent-traffic suite as well.

## The sampling family: what bit-exact means

The sink detector's alternates are the sampler's own draw in descending order, so the sink
kernels, the Gumbel sampler and the greedy argmax kernels move together. Their noise must stay
the stream Triton produces today, or every seeded sampled output changes. From the PTX Triton
compiled for `_gumbel_sample_kernel` in the image's cache (`.target sm_121a`):

- **Philox-4x32, ten rounds**, Triton's `randint4x`: multipliers 0xD2511F53 and 0xCD9E8D57, key
  increments 0x9E3779B9 and 0xBB67AE85, the 64-bit seed split into the key, the offset as the
  first counter word. The per-row seed is `tl.randint(seed, pos)` (with the draft salt 2^30
  added to `pos` while drafting), the draw `tl.rand(row_seed, token_id)`.
- **uint32 to float**: `x ^ (x >> 31)` (Triton's `uint_to_uniform_float`), `cvt.rn.f32.s32`, a
  multiply by 0f2FFFFFFF (4.6566127342e-10), and the clamp to that same minimum.
- **The noise**: `-log(-log1p(-u))` through libdevice's `__nv_log1pf` and `__nv_logf`, inlined
  with flush-to-zero FMAs (`fma.rn.ftz.f32`, `add.rz.ftz.f32`).
- **Temperature**: `logits / temp` compiles to `div.full.f32`, the approximate divide (CUDA's
  `__fdividef`), not IEEE `div.rn`.

The TileLang kernels share one CUDA header with these device functions, compiled with the same
flush-to-zero and fast-divide semantics, so the noise is a function of (seed, position, token) and
nothing else. The gate: the TileLang sampler's per-block maxima and argmaxes equal Triton's bit for
bit across seeds, positions, temperatures (0, 1, 0.6, 1.3), FP32 and FP64 draws, drafting and
target streams, and logits with ties, infinities and NaN; then the full-model determinism gate at
temperature 0 and a seeded temperature-1 run identical to the previous release.

## The sampling family in TileLang (vLLM branch `tilelang-sampling`, 2026-10-24)

Two commits on release-next's 174a746f4 (0265): a module commit with the kernels, their tests
and benchmark (`vllm/v1/worker/gpu/sample/tilelang_sampling.py`), and a switch commit that
points the call sites at them. The Triton kernels stay as the tests' reference.

| TileLang kernel | Replaces | Contract |
| --- | --- | --- |
| `gumbel_sample_kernel` | `gumbel._gumbel_sample_kernel` | target, drafting and logits-cache variants; per-block maxima and indices bit for bit |
| `temperature_kernel` | `gumbel._temperature_kernel` | `div.full.f32`, in place |
| `sink_candidates_kernel` | `sink._sink_candidates_kernel` | four argmax passes per block over the same noise |
| `sink_policy_kernel` | `sink._sink_policy_kernel` | every output and every state buffer it touches |
| `partial_argmax_kernel` | `greedy._partial_argmax`, `_partial_argmax_logits` | 16-byte reads where the rows allow them, as Triton's aligned specialization |
| `finish_argmax_kernel` | `greedy._finish_argmax_pair`, `_finish_argmax` | |
| `select_first_argmax_kernel` | `greedy._select_first_argmax` | |
| `greedy_rejection_kernel` | `greedy_verification._greedy_rejection_kernel` | |

- **The random stream.** Philox-4x32 with ten rounds and Triton 3.5's seed and counter
  layout; the FP32 noise as Triton compiles it, with libdevice's `log1pf` and `logf` from
  Triton's bundled libdevice (flush-to-zero FMAs) carried as inline PTX in one shared header.
  CUDA 13's own libdevice is another implementation, so calling `log1pf` from CUDA C would not
  give Triton's bits. Instruction for instruction the header matches the PTX Triton 3.5 emits
  for sm_121a; the test compares the noise on all 2**32 Philox words.
- **The reduction order.** A comparison with NaN is false, so which index a block with NaN
  reports depends on the order. The sampler and the candidates follow Triton's order at four
  warps: elements `t + 128 j` in sequence within a thread, then XOR shuffles 16 to 1, lane 0
  of each warp, then threads 0-3 at distances 2 and 1. The greedy reductions use DSpark's
  total-order combine (NaN first, then the maximum, then the lowest index), so any order gives
  the same pair.
- **Declarations.** Each owner declares the calls it launches where it registered the Triton
  warmups. That includes probabilistic drafting and the full-vocabulary greedy draft, which
  compiled at first launch under Triton. After the kernel cache freezes, a missing
  declaration raises instead of compiling.
- **Not ported: the FP64 draw.** `use_fp64_gumbel` is off in this deployment, and its path
  would add libdevice's double-precision `log` as another hundred lines of PTX. The runner now
  rejects it at startup.
- **Still Triton in the sampled-request path** (their own inventory, a temperature-1 profile):
  the block-verification and resampling kernels in `rejection_sampler_utils.py` (which inline
  the Triton Gumbel functions), vocabulary-parallel drafting, logprobs, penalties, top-k/top-p,
  min-p, the thinking budget and `_get_num_sampled_and_rejected_kernel`.

The bundle `bundles/sampling` (in spec `ports-kernels.json` with the other ports' bundles, one node each) runs the bit-equality
tests, the existing sink, Gumbel and greedy contract tests through the switched call sites,
and the paired per-call benchmark at DeepSeek V4.1's serving shapes (`--gate`: no shape slower
than Triton by more than 1%). The full-model gate follows in a serving arm: temperature-0
outputs and a seeded temperature-1 run identical to the previous release.

## The port bundles' first GPU run (candidate 4's window, 2026-10-24 12:50 CET)

- **Benchmarks never ran.** Every port benchmark failed with `No module named
  vllm.models.deepseek_v4_1`. The bundles' `PYTHONPATH=/b` replaced the image's
  `/opt/spark-ds41f/candidate/vllm:/opt/spark-ds41f/candidate/b12x`. Pytest still worked, because
  `-m pytest` puts the working directory on the path; the scripts did not. The bundles now keep
  the image's path after `/b`.
- **Embedding and confidence head:** 85 passed.
- **Runner inputs:** 48 failed, all from the tests (`tilelang-runner-inputs` 9cb3eb1b9 and 6bda32b34):
  - the slot-mapping test ran a prefill chunk past the 16384-token positions buffer when the
    step started past row 0;
  - the staged-write test staged overlapping writes with different values, which Triton itself
    applies in no defined order (the same seed gave different winners per dtype);
  - the boundary-restore test's stand-in model state predates release-next's
    `get_recurrent_checkpoint_fresh` (efcd03d6a, 2026-09-23), so it fails on release-next too.
  - The TileLang staged write, lowered, makes exactly Triton's guarded element stores.
- **mHC:** 18 failed. About 1 in 70,000 values differed from TileKernels in its bits.
  - TileKernels writes `post * x + comb[0] * r[0] + ...`. Of the first addition's two products,
    nvcc fuses the left one, so TileKernels rounds stream 0's product, not `post * x`. In PTX that
    is `mul.f32 comb[0,n]*r0`, then `fma post[n],x`, then streams 1-3 by `fma`, then
    `cvt.rn.bf16`.
  - The port rounded `post * x` instead. It now spells TileKernels' sequence out with explicit
    FMAs (`tilelang-mhc-post` 08c6ce4e7), so nvcc's choice no longer decides it.
  - A dataflow trace matches every output's chain to TileKernels', operand for operand, and an
    exact emulation reproduces the GPU's 0x3400 vs 0x3401. The collapse already matched.
  - The order-of-operations test's expected values had assumed the wrong order; they are
    corrected, with a case that tells the two apart (83b073cec also sorts the declared-calls
    test by factory).

## The ports' kernel window (2026-10-24 14:10 CET, run `tilelang-ports-kernels`)

One spec ([ports-kernels.json](ports-kernels.json)), one node a bundle, image -next-4. Production
was restored at 14:12.

- **Embedding and confidence head: pass.** 85 tests and the benchmark gate.
- **mHC: all 60 tests pass, bit-identical to TileKernels.**
  - The post is faster at every shape: one stream -37%, sixteen -6%, 8K and 16K prefill -4.6%.
  - The collapse is faster at decode (-4.6%, -10%) but 1.1-1.2% slower at 8K and 16K
    prefill, so the gate fails; under repair.
- **Runner inputs: all 507 tests pass.** Its benchmark hit an illegal memory access in its first
  case before printing a row; under repair.
- **Sampling: 351 passed, 43 failed.**
  - The image's Triton is 3.7.1, not 3.5.0. The noise on all 2**32 words and the Philox streams
    still match: 3.7.1 inlines the same libdevice sequences.
  - **3.7.1 lays the block out by the load's specialization.** When the logits' base and row
    stride are divisible by 16, a thread holds 16-byte runs: four FP32 values, runs 512
    apart, or eight 16-bit values. Otherwise it holds one element, 128 apart (the order the
    port first replicated). The order decides the index a block with NaN reports, which is
    every failing Gumbel and candidates case: the "special" logits at a stride of 129,344.
    The cases at strides 1,064 and 2,113 passed. Production's Triton cache shows the same:
    FP32 runs of four and bf16 runs of eight, at every cached variant.
  - The port now takes the run from the same rule and loads each whole run as one vector, as
    Triton does (vLLM `tilelang-sampling` e5f883a9c). The tests cover both strides at every
    vocabulary, partial tail runs included.
  - The policy kernel had one length for its flags, values and drafts, but the drafts may be
    longer, and TileLang's packed ABI checked them equal. Every buffer now has its own length.
  - Greedy verification also takes int64 drafts.
  - The sampler-flags test hooked every Triton compile and caught the top-p kernels, which an
    earlier test happened to have compiled in the release tests. It now checks that the
    TileLang Gumbel kernels compile at warmup only (0cfca05d8).
  - **Speed** (before the layout change, scalar loads):
    - the Gumbel sampler was 2-18% slower with bf16 logits at 1-6 tokens and level or faster
      with FP32;
    - the temperature kernel was 6-50% slower;
    - the sink candidates were 4-46% slower.

    Triton's 16-byte loads account for most of it; the next run measures the vector port.

**Round 2's fixes** (the same spec, `run` tilelang-ports-kernels again):
- **mHC collapse** (`tilelang-mhc-post` d781f2a04). Both kernels already used 16-byte loads
  and stores on the same grid; TileKernels runs 64 threads a block, each 16 columns in two chunks
  512 apart. The collapse now reads and writes in TileKernels' order and block shape, without its
  shared-memory stores that are never read back. It has the same FMA chain, and 232 SASS
  instructions a thread in both.
- **Runner benchmark** (`tilelang-runner-inputs` 1bc9f13f6). Three cases passed tables of raw
  addresses without keeping the tensors alive. The graph capture's `empty_cache` returned them to
  the driver, and the Triton warm-up read through freed addresses. The cases now keep the
  tensors. Under Triton's CPU interpreter every case's launches stay in bounds.

## Round 2 (2026-10-24 14:55 CET): every test passes; speed is the work left

- **Bits:** every test passes in all four bundles.
  - Sampling: 639 passed, the NaN blocks at both strides included; Triton 3.7.1's layout rule
    holds.
  - mHC: 60 passed. Runner inputs: 507 passed. Embedding and confidence: 85 passed, and its
    gate passes.
- **Speed: three gates fail.** Ratios are TileLang to Triton (or to TileKernels) per call;
  every one is outside its 95% interval.
  - **Sampling.**

    | Kernel | Faster or level | Slower |
    | --- | --- | --- |
    | Sink policy | -28 to -30% | |
    | Greedy verification | level | |
    | Gumbel sampler, FP32 | level | |
    | Gumbel sampler, bf16 | | +3.6% and +3.0% at 16 and 96 tokens |
    | Temperature | | +70% at one token, +12% at 16 |
    | Sink candidates | | +13% at one request (greedy), +9% and +7% at 16 and 64 (sampled) |
    | Greedy shard pairs | | +17 to +42% |
    | Full-vocabulary greedy draft | | +20 to +37% |
    | Select | | +25 to +32% (0.3 us on 1.3 us) |

  - **mHC.** The post is faster at every shape (-37% to -4.7%). The collapse is faster at
    decode, but TileKernels' block shape made long prefill worse: +2.4% and +2.6% at 8K and
    16K (round 1: +1.2% and +1.1%).
  - **Runner inputs.**
    - Faster, by large margins: block tables -22 to -59%, slot mappings -24 to -86%, staged
      writes -7 to -57%, post update at decode, prefill inputs at 16K.
    - Slower, the tiny kernels (1-3 us): dead-row draft combine +27 to +34%, index expansion
      +14 to +16%, DFlash inputs at decode +18 to +21%, prefill inputs at decode +5 to +16%,
      lookback +4%, the KV zeroing at one stream +2%.
- **Next:** measure before tuning again; the offline reasoning about speed missed twice.
  - A profiler mode gives each kernel's own GPU time, since some calls launch two kernels.
  - Variant sweeps measure the shapes that do not change the bits: threads and rows a block,
    and a sequential finish against the warp's for the tiny reductions. Layouts that decide a
    NaN block's index stay Triton's.

## Round 3: variant sweeps ([variants-kernels.json](variants-kernels.json))

Each sweep checks every variant bit for bit against the kernel it replaces, then times it
against that kernel per call (eight fresh inputs, paired CUDA graphs, order alternating).

- **[bundles/sampling-variants](bundles/sampling-variants)** (dgx4):
  - the greedy partial argmax at 64, 128 and 256 threads a block, with a warp or a thread per
    row in the finish;
  - the select kernel likewise;
  - temperature at 64, 128 and 256 threads;
  - each production kernel compiled as built and with TileLang's replay of bound loads off
    (`tl.if_stmt_binding_inline_replayable_binds`), which the runner agent found re-issues
    loads after every store and branch;
  - each kernel's own device time (CUPTI).

  The Gumbel sampler and the candidates keep Triton's layout, which decides the bits.
- **[bundles/mhc-variants](bundles/mhc-variants)** (dgx3): 31 layouts of the mHC collapse with
  TileKernels' FMA chain:
  - threads a block, rows a block and columns a thread;
  - load order and vector width;
  - blocks a SM capped with unused shared memory, and persistent grids;
  - round 1's and TileKernels' shapes as references.

  The port's collapse is back to round 1's (`tilelang-mhc-post` c68e5b70f).
- **[bundles/runner-variants](bundles/runner-variants)** (dgx1): about 68 variants of the runner's
  tiny kernels:
  - loads hoisted, binds off;
  - Triton's lane layout, a block per request;
  - int32 indices, static strides;
  - threads and work a thread.

### Round 3 results (2026-10-24 16:41 CET; all variants bit-exact)

Ratios are per call against the kernel replaced (paired CUDA graphs); "device" is the kernel's
own time (CUPTI).

- **Sampling** (dgx4):
  - Bind replay off changes nothing in this family.
  - Temperature is best at 256 threads: +18% at one token (+70% at 128), +10% at 16 (not
    significant), +17-18% at 96 at any thread count.
  - Greedy shard pairs are best with a warp per row in the finish: +27-31% at 1-6 rows (256
    threads), +22-25% at 16 (128), +15-22% at 96 (64-128). A thread per row is always
    slower.
  - Select is +20-33% (about 0.3 µs) in every mode.
  - Device time: Gumbel bf16 +3.9% at 16 tokens and +5.8% at 96, f32 +0.9%; candidates +16%
    at one request and +8% at 16; greedy Markov pair at one row: partial 2.24 vs 1.29 µs,
    finish 1.25 vs 0.85 µs.
- **mHC collapse** (dgx3), against TileKernels at decode c1 / c16 / 8K / 16K:
  - Round 1's: -5.5% / -10.7% / +1.6% / +1.5%.
  - Capped at 16 KiB of reserved shared memory a block: -4.9% / -7.6% / -1.7% / -2.2%,
    faster at every shape.
  - Capped at 24 KiB: -0.8% / -6.0% / -2.4% / -2.6%.
  - Persistent grids, column-major grids and more rows a block are slower.
- **Runner** (dgx1):
  - Pass:
    - post_update with hoisted scalars at 32 threads: -9 to -16% at every shape.
    - combine at 128 threads: -9 to -12% at decode, +5% (0.08 µs) at the 16K prefill.
  - Bind replay off: device time -8 to -13% for combine and post_update, nothing elsewhere.
  - int32 indices and static strides gain nothing anywhere.
  - Still slower in every variant:
    - expand_idx_mapping: +10-13% (device +0.25 µs on 0.81).
    - gather_lookback: +1-7%.
    - prepare_dflash_inputs at decode: +11-17%.
    - prepare_prefill_inputs at decode c16: +14%.
    - combine with dead rows at the 16K prefill: +4%.

### Why the small kernels lose (SASS for sm_121a, compiled locally)

Triton writes a block's loads as one masked tile, issued before anything uses them, and its
masked stores are predicated instructions. The ports are written a thread at a time with
branches, and two things follow:

- **expand_idx_mapping:** the TileLang kernel loads the request's two starts, branches on
  them, and only then loads its state index: two round trips to memory where Triton has one.
  ptxas moves that load below the branch even when the source reads it first (round 3's
  hoisted variant), and even as a volatile asm load. With the guarded stores written as
  predicated PTX there is no branch to sink into, and the three loads issue together, as
  Triton's do.
- **Temperature:** each run's load, division and store come before the next run's load, and
  the logits are read and written through one pointer, so every load waits for the store
  before it: a round trip per run, 16 per thread. Triton loads the block, divides, then stores.
- **Select:** a loop of dynamic length, unrolled by four with a remainder (252
  instructions). Triton fixes the number of ranks. In tile form the kernel is 66
  instructions: one load per lane, then the shuffles.

## Round 4a: Triton's tile form ([tileform-kernels.json](tileform-kernels.json))

- [bundles/sampling-variants](bundles/sampling-variants) `4a` (dgx4):
  - temperature at 1, 16 and 96 tokens and 128 or 256 threads, production against
    [sampling_tile.py](bundles/sampling-variants/sampling_tile.py) (every run loaded, then
    divided, then stored);
  - select at 1, 16 and 96 rows, production against tile form (ranks fixed when built);
  - device times.
- [bundles/runner-variants](bundles/runner-variants) (dgx1): expand_idx_mapping in production,
  hoisted and tile form (`expand_tile_kernel`, predicated stores), at 32 and 128 threads.
- [bundles/mhc-variants](bundles/mhc-variants) (dgx3): round 1's collapse and the 8, 16 and
  24 KiB caps, three passes each, to see whether the prefill gain repeats.

If tile form closes the gaps, it becomes the port convention:
- predicated store and masked load helpers in the shared header;
- loads first, then compute, then stores;
- trip counts fixed in the kernel key.

Every port that still loses at a serving shape gets rewritten that way. The compiler pass
that would do this automatically (turning guarded stores into predicated ones) is TileLang
work that comes after.

### Round 4a results (2026-10-24 17:04 CET; every variant bit-exact)

- **mHC collapse**, three passes against TileKernels at decode c1 / c16 / 8K / 16K:
  - The 24 KiB cap: -4.3 to -5.8% / -6.0 to -8.2% / -2.2 to -2.4% / -3.0 to -3.1%.
  - The 16 KiB cap: -5.1 to -5.7% / -7.2 to -9.4% / -0.9 to -1.0% / -2.1 to -2.3%.
  - Round 1's collapse: +1.0 to +1.7% at prefill.

  The production collapse takes the 24 KiB cap.
- **Tile form** against Triton, per call:
  - expand_idx_mapping at 32 threads: +1 to +3%, within noise (production +13 to +19%).
  - Select: +6 to +8% (production +23 to +33%).
  - Temperature at 256 threads: +7% at one token (production +17%), level at 16. At 96
    tokens, +16% in both forms.
- **Read-only loads:** the remaining tenth of a microsecond is the read-only cache path.
  TileLang marks inputs `const __restrict__`, so nvcc emits `ld.global.nc`
  (`LDG.E.CONSTANT`); Triton's are ordinary loads. `T.annotate_restrict_buffers` on the
  inputs gives Triton's `LDG.E` (seen in SASS).

## Every port together (vLLM branch `tilelang-ports` 1a967a2ec, [integration-kernels.json](integration-kernels.json))

One vLLM branch off candidate 4's source (release-next 174a746f4) carries every TileLang
port. Codex's trailers are dropped from the picked commits.

The ports:
- runner inputs (`tilelang-runner-inputs`);
- token counts (`tilelang-token-counts`, #506);
- the sampling family;
- the resampled token (`tilelang-resampled-token`, #509);
- logprob ranks (`tilelang-logprob-ranks`);
- mHC post and collapse;
- the embedding and DSpark's confidence head.

The measured changes on top:
- the mHC collapse's 24 KiB cap;
- temperature and select in tile form;
- expand_idx_mapping in tile form with ordinary loads.

[make_integration_bundles.py](make_integration_bundles.py) writes four bundles. Each
overlays all 46 changed files on image -next-4, so every family runs with every other port
in place. Each runs one share of the ports' GPU tests and gated benchmarks:
- runner and token counts (dgx1);
- embedding, confidence, ranks and resampled token (dgx2), with the vocabulary-parallel
  drafting tests in their own step: the unchanged image fails eight of them;
- mHC (dgx3);
- sampling (dgx4).

## Round 4b: ordinary loads ([plain-loads-kernels.json](plain-loads-kernels.json))

The sampling and runner variant bundles now overlay the integration branch, so "prod" is
the branch's kernels. [plain_loads.py](plain_loads.py) recompiles a production kernel with
every buffer it never stores to listed in `tl.non_restrict_params`, which is what
`T.annotate_restrict_buffers` sets. Those buffers' loads become ordinary loads, as
Triton's are.

Both forms of every production kernel are timed against Triton per call:
- sampling `4b` (dgx4): Gumbel, candidates, temperature (1, 16 and 96 tokens), greedy
  pairs, select;
- runner `--variants prod` (dgx1): every runner kernel as built, with bind replay off, and
  with ordinary loads.

Triton's temperature is also timed against itself on the two 96-token buffers, to test
whether that gap belongs to the buffers.

### The first integrated run (2026-10-24 17:33 CET, run `tilelang-integration-kernels`)

- **Tests:**
  - Runner and token counts: 539 passed.
  - Sampling: 639 passed.
  - mHC: 60 passed.
  - Embedding, confidence, ranks and verification: 203 passed, 8 failed. The 8 are all
    `test_gumbel_drafted_rejection_sample_is_unbiased`. Its proposals are drawn by
    `gumbel_sample` with int32 positions, which Triton's kernel takes (`tl.randint` uses an
    int32 offset as its only word and salts a draft's in 32 bits). The TileLang sampler
    and sink candidates required int64. Fixed on the branch: the positions' dtype is in
    both kernels' keys (serving's are int64, as declared), and int32 positions have their
    own tests against Triton.
  - The vocabulary-parallel drafting tests fail 8, as on the unchanged image.
- **Gates:**
  - mHC: passes.
  - Token counts: -9 to -12%. Embedding and confidence, logprob ranks and the resampled
    token pass their own gates.
  - Runner, still slower:
    - combine with dead rows: +0.4 to +0.7 µs;
    - prepare_dflash_inputs at decode: +0.5 µs;
    - prepare_prefill_inputs at c16: +0.26 µs;
    - post_update and combine at the 16K prefill: about +0.12 µs;
    - gather_lookback: +0.06 to +0.10 µs;
    - zero_kv_blocks at c1: +0.09 µs, and +0.6% at 16K.

    expand_idx_mapping is +0.01 to +0.04 µs.
  - Sampling:
    - Temperature is level at every size, 96 tokens included (389.0 against 389.4 µs).
      Round 4a's +16% was its own benchmark's.
    - Still slower: the Gumbel sampler in BF16 (+2 to +3.5%), sink candidates (+2 to
      +12%), select (+7%, under 0.1 µs), and the greedy shard pairs and Markov sampling
      (+18 to +42%, 0.8 to 1.7 µs).

The second integrated run repeats every bundle on the fixed branch.

### Round 4b results (2026-10-24 17:43 CET)

- **Ordinary loads gain nothing.** Every production kernel, sampling and runner, takes the
  same time with its read-only buffers off the read-only cache path, within noise. The
  read-only path is not what slows the small kernels. expand_idx_mapping's last 0.13 to
  0.15 µs of device time remains with ordinary loads and Triton's instruction shape.
- **The 96-token temperature gap is the buffers.** Triton's own kernel takes 458.8 µs on
  the first of the two buffers and 387.8 µs on its clone (+18%), the same gap as round 4a's
  "+16%".

  Two same-size buffers allocated one after the other stream at different speeds on GB10.
  If that holds for serving's weights, KV cache or logits, it is a decode-speed lever
  that leaves the bits unchanged. The placement bundle looks for the cause.
- The TileLang launch is an ordinary `cuLaunchKernel` (TVM's runtime adds programmatic
  dependent launch only for kernels with a grid sync), so the launch is not the fixed
  cost either.

## Round 4c and placement ([greedy-placement-kernels.json](greedy-placement-kernels.json))

- **Greedy reductions** (sampling-variants `4c`, dgx4). The greedy shard pairs at 1, 6, 16
  and 96 rows are timed four ways: production, the partial argmax in tile form (every
  run loaded before any is combined), the finish in tile form (parts fixed when built,
  each lane's parts loaded up front), and both. Each is checked bit for bit and timed
  against Triton, then the kernels' device times.
- **Placement** ([bundles/placement](bundles/placement), dgx2). The same in-place
  streaming pass runs over 49.6 MB buffers allocated in different ways:
  - round 4b's pair (randn, then a clone), and the reverse order;
  - eight buffers in a row;
  - sizes rounded to whole 2 MiB pages;
  - slices of one allocation;
  - a block reused from the cache.

  Each buffer's time is printed with its address.

### The second integrated run (2026-10-24 17:48 CET, run `tilelang-integration-kernels-2`)

Every port's tests pass with every other port in place:
- runner and token counts: 539;
- embedding, confidence, ranks and the rejection sampler: 211, the int32-position cases
  included;
- mHC: 60;
- sampling: 643.

The eight vocabulary-parallel drafting failures are a bug in the tree, not in the ports.
The backport of vllm-project/vllm#61161 (`ad6c55362`, 10-19) puts the draft step in the
seed's high word in the full-vocabulary draft sampler, DFlash2 and the rejection sampler.
The vocabulary-parallel draft kernel kept the bare seed, so its drafts after step 0
differed from the full-vocabulary path. DSpark samples its drafts over the full vocabulary
and is unaffected. Fixed on the branch (`f7cf50549`), to go into the release series too.

### Round 4c and placement results (2026-10-24 18:00 CET)

- **Greedy reductions:** tile form helps little.
  - Partial argmax: 2.28 to 2.07 µs device at one row (Triton 1.3).
  - Finish: 1.24 to 1.13 µs (Triton 0.85).
  - Per call, the pairs stay +18 to +40%.

  Both kernels are chains of dependent combines (16 a thread, then the shuffles). The
  production combine takes the other pair under an `if`; Triton's is selects.
- **Placement, round 1:** every scenario had exactly one fast buffer (about 265 GB/s,
  the rest about 222 GB/s), always the one whose graph was built last. That held even for
  the last slice of one allocation, and the same address was fast in one scenario and
  slow in the next. Not settled yet: it may be an ordering effect, not placement.

## Round 4d and placement round 2 ([branchless-placement-kernels.json](branchless-placement-kernels.json))

- **Branchless combines** (sampling-variants `4d`, dgx4): every sampling kernel traced with
  its combines as selects (`sampling_tile.branchless`). The greedy partial argmax also
  reduces each thread's sixteen values as a tree. The first maximum is a total order, so
  the result is the same and the dependent chain is four combines deep instead of
  fifteen.
- **Placement, round 2** ([placement2.py](bundles/placement/placement2.py), dgx2):
  - the graphs built in reverse order, to see whether the fast buffer moves with them;
  - the same buffers timed without graphs;
  - a speed map of twenty 48 MiB slices of one 1 GiB allocation.

### Round 4d and placement round 2 results (2026-10-24 18:12 CET)

- **Placement round 2:**
  - Four separate 50 MB buffers, timed in shuffled order with or without graphs, all
    stream at the same 222-223 GB/s. Round 1's fast buffer was mostly its alternating
    timing order.
  - In a 1 GiB allocation mapped in twenty slices, only the last slice is faster (253
    GB/s), in shuffled order as well. That one remains unexplained.
  - The boot-spread experiment (2026-10-14) already measured fragmentation at serving
    boots: 80-93 GiB free in blocks of 2 MiB or more once serving stops, and compaction
    made no measurable difference to step time.

  So 222 GB/s is GB10's in-place streaming rate. Placement is not a lever for serving,
  and it is parked.
- **Branchless combines and the tree:**
  - Small gains: select from +7-8% to +4-7%; the greedy pairs' best form (tree partial,
    tile finish) at +0.6 to +0.8 µs (from +0.8 to +1.3).
  - Gumbel and candidates unchanged.
- **The cause, in SASS** (disassembled on dgx4 from the bench's compile cache; TileLang
  embeds the cubin in `executable.so`): the production greedy partial argmax has
  **107 branches**, Triton's has 3 (with 76 selects). The header's combine helpers,
  `vllm_first_argmax_keep` and `vllm_argmax_keep`, used `&&` and `||`, which nvcc
  compiles to a branch for each comparison. Round 4d's select macros kept those helpers,
  so they were still branchy inside.

  The branch now uses bitwise operators (same values, no short-circuit) and select
  macros (`1449db182`). Every argmax-style kernel uses these helpers: Gumbel, candidates,
  greedy, select, finish.

## Third integrated run ([integration-kernels.json](integration-kernels.json), run `tilelang-integration-kernels-3`)

The branch with the branch-free combines and the vocabulary-parallel seed fix (47 files),
every bundle again: tests, then the gated benchmarks.

### What the instruction counts showed next (SASS from the nodes' compile caches)

TileLang's kernels against Triton's, counted on dgx1 and dgx4 (`nvdisasm` on the cubin
TileLang embeds in `executable.so`):
- **combine with dead rows:** 636 instructions and 47 loads against Triton's 159 and 9.
  TileLang's IfStmtBinding replays bound loads under each guarded statement. The runner
  module now turns replay off (`0441ba1be`). Round 3 measured -8 to -13% device time on
  combine and post_update, with the same bytes everywhere.
- **The Gumbel sampler:** 1,288 instructions against Triton's 1,163, of which +72 are
  integer adds. Every element's Philox draw sat inside its own bound and temperature
  checks, so nvcc recomputed the key schedule (the same for every element of a token)
  for each one.

  The branch now tests the temperature once around the block's elements, as Triton
  does, and draws each word before the element's bound check (`ea3492d9f`). The sink
  candidates kernel is restructured the same way.
- **prepare_dflash_inputs:** 44 to 47 branches against Triton's 24. Not yet addressed.

## Fourth integrated run (run `tilelang-integration-kernels-4`)

The branch with the replay-free runner kernels and the restructured Gumbel and candidates
kernels (`ea3492d9f`).

### Third and fourth integrated runs (2026-10-24 18:20 and 18:33 CET)

Every test passes in both runs, the vocabulary-parallel drafting tests included (12 of 12),
which confirms the seed fix.

- **Branch-free combines (run 3):**
  - greedy Markov pairs: from +18-42% to +9% (1 row), +8% (6), level at 16 and 96;
  - greedy shard pairs: +5% at 6 rows, level at 36 and 384, -6% at 96;
  - sample_greedy_markov: +4% at one row, level at 16;
  - select: +1-2%;
  - sink candidates at t=0: -3% at 16 and 64 requests.
- **Gumbel and candidates draws outside the bound checks (run 4):**
  - the Gumbel sampler is now 3-5% faster than Triton in BF16 and 2-10% in FP32, at every
    size;
  - sink candidates are level or faster everywhere. One request at t=0 reads +12%, but
    its interval includes zero.
- **Runner without bind replay (run 4):**
  - combine with dead rows: from +0.69 to +0.44 µs;
  - post_update: faster than Triton at every shape;
  - combine: level.
- **Still slower:**
  - temperature at one token (+0.2 µs);
  - greedy pairs at 1-6 rows (+0.2-0.3 µs);
  - combine with dead rows (+0.3-0.4 µs);
  - prepare_dflash_inputs at decode (+0.5 µs);
  - prepare_prefill_inputs at c16 (+0.3 µs);
  - gather_lookback (+0.07 µs);
  - zero_kv_blocks (+0.1 µs at c1, +0.7% at 16K);
  - the embedding's 8K cold prefill (+3 to +4 µs of 488, +0.6-0.8%, the same in every
    run; first significant in run 4).

## Fifth integrated run (run `tilelang-integration-kernels-5`)

prepare_dflash_inputs pads the query slots from eight extra blocks (`baf8ce86c`). Before,
the last request's first block wrote all 16,384 of them alone, the kernel's longest path at
decode.

## Sixth integrated run (run `tilelang-integration-kernels-6`)

On top of the fifth (vLLM `tilelang-ports` 649dde962):
- **combine** reads each request's scalars, draft row and confidence row up front.
  With dead rows each request takes a block of its own (round 3's fastest form). The
  key carries the drafts' width.
- **The greedy finish** loads `ceil(parts / 32)` parts a lane before combining (Triton's
  masked values past the end), keyed by that count (1 or 2, both declared), instead of a
  loop of runtime length.
- **Temperature** divides 2048 columns a block instead of Triton's 8192. It is
  element-wise, so the split is free, and one token gets 64 blocks instead of 16.

### Fifth integrated run (2026-10-24 18:41 CET)

- Every test passes.
- prepare_dflash_inputs is unchanged at +0.45 to +0.52 µs, so spreading the padding was
  not the cure. In SASS, Triton's kernel issues its nine loads together at the top and
  stores under masks. Ours issues them in rounds behind branches: the request index, then
  the positions, then the block table, each in its own conditional block.
- The resampled token's one-request mixed-temperature case read +8% (+0.06 µs), the case
  its experiment left inconclusive. Every other size is 1-6% faster than Triton.

## Seventh integrated run (run `tilelang-integration-kernels-7`)

prepare_dflash_inputs in Triton's tile form (vLLM `e4729693c`): every load up front at an
index that exists whatever the masks say, masks as selects, stores predicated.

### Sixth and seventh integrated runs (2026-10-24 18:47 and 18:58 CET)

Every test passes in both.
- **combine** with dead rows, every scalar up front and a block a request: -0.20 to -0.38
  µs against Triton at decode (it was +0.44), +0.07 µs at the 16K prefill.
- **prepare_dflash_inputs** in tile form: faster than Triton at every shape (-0.18 to
  -0.38 µs; it was +0.45 to +0.53).
- **The greedy finish** with its parts loaded first: the pairs are level from 6 rows on
  and +7% (0.2 µs) at one row.
- **Temperature at 2048 columns a block:** -15% at one token (it was +7%) but +5% at 16
  tokens, where 8192 was -13 to -19%.
- **Still slower:**
  - sink candidates at one request (+13%);
  - prepare_prefill_inputs at decode (+0.2 to +0.35 µs; the early exit is the same chain
    of loads as Triton's, and round 3's thread and hoisting sweeps did not close it);
  - zero_kv_blocks (+0.1 µs at c1, +0.4% at 16K);
  - gather_lookback (+0.07 µs);
  - the embedding's 8K cold prefill (+0.7%).

## Eighth integrated run (run `tilelang-integration-kernels-8`)

vLLM `tilelang-ports` 5c819fba5:
- the sink candidates' rounds unrolled;
- temperature's block width by the step's token count (2048 below eight tokens, Triton's
  8192 from eight), both declared and both tested;
- zero_kv_blocks reads its segment's scalars before the page test.

### Eighth integrated run (2026-10-24 19:05 CET)

Every test passes. Sampling has 661 tests now, with temperature at both block widths.
- **Temperature** is faster than Triton at every size: -16% at one token, -14% at
  sixteen, -1% at 96.
- **Sink candidates**, with their rounds unrolled: faster everywhere but one request at
  t=0, which went from +13% to +8%.
- **Unchanged:**
  - zero_kv_blocks, +0.1 µs at c1 and +0.4% at 16K. Reading the scalars before the page
    test did not move it.
  - prepare_prefill_inputs at decode, +0.1 to +0.3 µs.
  - gather_lookback, +0.06 to +0.09 µs.
  - the greedy pair at one row, +6%.

## Ninth integrated run (run `tilelang-integration-kernels-9`)

vLLM `tilelang-ports` 0c0c0bb35:
- **the embedding** takes a row a block from 512 tokens (each thread's five pieces
  loaded before any is stored); below 512 a row stays spread over five blocks. Both
  layouts are declared.
- **the greedy partial argmax** loads every run before combining.

## The serving check ([serving-ports.json](serving-ports.json))

Every port of the branch, served. [ports-tp4.json](ports-tp4.json) mounts the branch's 29
package files (bundles/serving/overlay, written by `make_integration_bundles.py
--serving`) over candidate 4 at tp4-ring. It is measured beside candidate 4 itself
(rc4/rc-tp4.json), ports first. Each arm runs:
- the lean bench;
- `token_determinism.py --wide`: temperature 0, alone, eight at once, staggered, mixed with
  a long prefill, fifteen at once, cached, again;
- [sampled_digest.py](sampled_digest.py): seeded temperature 1, alone, eight at once,
  again, with speculative decoding's sampled drafts, verification and resampling;
- distinct prompts at one and eight streams.

[tables_arms.py](tables_arms.py) compares the arms' reference digests. The ports pass when
both arms are identical within themselves, their temperature-0 and seeded temperature-1
digests are equal, and the ports boot and serve with no undeclared kernel
(`KernelNotPreparedError` would fail the requests).

### Ninth integrated run (2026-10-24 19:20 CET)

- Every test passes.
- The embedding's 8K cold prefill is 1.7% faster than vLLM's CUDA kernel (it was +0.7%).
- The greedy reductions are level or faster at every size (-13% at 96 rows).
- Still slower:
  - sink candidates at one request and t=0 (+8%, 0.23 µs);
  - zero_kv_blocks;
  - combine at the 16K prefill;
  - expand_idx_mapping at c1;
  - gather_lookback at c16.

### The serving check (2026-10-24 19:51 CET, run `tilelang-ports-serving`)

The ports booted and served with no undeclared kernel.
- **Outputs, bit-identical to candidate 4:**
  - Temperature 0: identical in every scenario within each arm, and the ports' reference
    digests equal candidate 4's.
  - Seeded temperature 1: the ports' reference digest equals candidate 4's.
  - In both arms the same seeded requests come out differently eight at once than alone,
    so the seeded draws depend on the company. That is the candidate's own behaviour, not
    the ports'.
- **Speed:**
  - One stream: level. Prose step 29.68 against 29.74 ms (±0.2%), JSON 33.60 against
    33.65 ms; c1 distinct 93.5 against 93.2 tok/s.
  - **Eight distinct streams: the ports are 1.7% slower**: 189.4 tok/s (±0.05%) against
    192.65 (±0.2%), on outputs that are identical (equal c8 digests). The ports ran
    first, so heat soak favoured them. The kernel benchmarks show no such cost, so a
    torch-profiled pair finds the kernels: [profile/profile-ports.json](profile/profile-ports.json)
    profiles ports, candidate 4 and ports again on the c8 JSON case and single-stream
    decode.
  - Prefill at 1K and 16K: within its intervals (±20-40% at two repeats).

## Tenth integrated run (run `tilelang-integration-kernels-10`)

vLLM `tilelang-ports` 1ac412cc7:
- **Sink candidates** issue every load once its index is known, the logits before the
  temperature's test and the rejection by selects (`9c696482b`).
- **The generation controls' TileLang candidates** (#524,
  experiments/2026-10-24-sampling-controls): its fourteen vLLM commits (patches 0261-0274),
  rebased onto this branch with their trailers dropped. The candidates are penalties,
  histogram accumulation, min-p, logit bias, forbidden sequences, the thinking marker cache
  and budget, prompt token ids, custom logprob ids and NaN counting. They are prepared
  and declared, but the existing kernels stay dispatched.
  - The sampling bundle now also runs their tests (`test_gpu_sampling_controls.py`,
    `test_gpu_thinking_budget.py`, `test_gpu_logit_bias.py`) and
    `benchmark_sampling_controls.py --gate`.
  - Open, from #524:
    - min-p is unqualified (568 against 539 µs at 96 rows), and its threshold equality
      with the reference logarithm must be kept;
    - seven candidates have no timing yet;
    - histogram callers keep their selected-row zeroing.

  The reference-logprob investigation stays separate from these behavior-preserving
  translations.

The first profile run stopped at its first workload: a profiled arm must serve with
`--profiler-config`, and the serving arms do not. [prof-ports.json](profile/prof-ports.json)
and [prof-candidate4.json](profile/prof-candidate4.json) are the two arms with candidate
4's resolved serve arguments and the torch profiler added. The run is repeated as
`tilelang-ports-profile-2`.

### Tenth integrated run (2026-10-24 20:10 CET)

- **Tests:** every one passes, the generation controls' 141 included.
- **The sampling family passes its whole gate:** at every shape each kernel is level with
  Triton or faster. The sink candidates' loads at one request closed the last gap.
- **Generation controls** (gated timings: penalties, histogram accumulation, min-p):
  - Penalties: -30% at one row, -7% at 16, -5% at 96.
  - Histogram accumulation: -4 to -44% at every width and row count.
  - Min-p: faster at 1 and 16 rows, **+8% at 96 rows** (572 against 530 µs). Both kernels
    take a row a block, a max pass then a threshold pass, 1024 columns at a time. At 96
    rows that is two blocks an SM, and ours issues one block's loads at a time behind a
    per-element bound check.
- **Runner, still flagged:** zero_kv_blocks at c1, combine with dead rows at the 16K
  prefill, gather_lookback.

### The profile (2026-10-24 20:24 CET, run `tilelang-ports-profile-2`)

Torch-profiled, with three arms: ports, candidate 4, then ports again.
- **Eight JSON streams:**
  - Step span, in ms a step on dgx1:
    - ports 73.13;
    - candidate 4 73.25;
    - ports again 73.11.
  - The other ranks are the same to within 0.1 ms.
  - The window's rate, in tok/s:
    - ports 263.3;
    - candidate 4 262.9;
    - ports again 263.4.
  - Drafts and acceptance are identical in all three arms (292 draft events, 687 accepted).
- **One stream:** step span 39.53, 39.33 and 39.43 ms.
- **By kernel family:** the two ports arms differ from each other by as much as either
  differs from candidate 4.
  - The one-shot all-reduce and the L2 prefetch move by 0.3-1.1 ms a step between arms.
  - The families the ports replace (mHC, sampling, the runner's "other") move by under
    0.06 ms.

The profile finds no cost of the ports at eight streams, so the serving check's -1.7% at
eight distinct prompts looks like boot-to-boot variation. A bracketed re-measure without
the profiler settles it:
- [serving-c8-bracket.json](serving-c8-bracket.json) measures ports, candidate 4, then
  ports again;
- the workloads are eight distinct prompts (three samples) and one distinct stream; there
  is no bench.

## Min-p in tile form (run `tilelang-controls-variants`)

In SASS, #524's min-p takes its 1024-column tiles in small groups of scalar loads. Each
group sits between branches on the bound check, so few bytes are in flight. Triton keeps
two 16-byte vectors a thread in flight and stores them as vectors.

vLLM `tilelang-ports` 08c1383c3 rewrites the kernel in tile form:
- every load of a tile is issued before any value is used;
- when the rows are 16-byte aligned (`run` 4, the launch checks), loads and stores are
  vectors in Triton's layout;
- only the last, partial tile clamps its columns to the row and predicates its stores.

The threshold is still `maximum + vllm_control_log(min_p)`, the exact reference logarithm.

[bundles/controls-variants](bundles/controls-variants) sweeps the run (vectors or scalars),
the threads a block (128, 256) and the tile width (1024, 2048, 4096):
- each variant is first compared with Triton bit for bit (threshold seeds 48291-48298 at 96
  rows, and non-finite rows);
- then it is timed against Triton at 1, 16 and 96 rows.

### The eight-stream bracket (2026-10-24 20:42 CET, run `tilelang-ports-c8-bracket`)

Eight distinct prompts, three samples an arm, no bench:

| Arm | tok/s | Samples |
| --- | --- | --- |
| ports | 189.47 | 189.35, 189.72, 189.34 |
| candidate 4 | 190.79 | 192.96, 189.43, 189.98 |
| ports again | 189.16 | 189.21, 188.64, 189.63 |

- The gap is 0.7%, down from the serving check's 1.7%.
- Two of candidate 4's three samples fall inside the ports' range. Its first is the outlier.
- One stream is level: 93.57, 93.18 and 93.35 tok/s.

[serving-c8-abab.json](serving-c8-abab.json) settles it: ports, candidate 4, ports, candidate
4, eight samples each.

## Runner round 5: gather_lookback and zero_kv_blocks in tile form (run `tilelang-runner-round5`)

- **gather_lookback**, in SASS:
  - Ours computes the element's row with two 64-bit divisions by three (TileLang widens
    the index arithmetic) and branches before its first load.
  - Triton's first load depends only on the block index, and every load is predicated.
- **zero_kv_blocks:** Triton loads only the page size before its early exit, and stores
  int32s with predicates.

[runner_variants.py](bundles/runner-variants/runner_variants.py) adds Triton's form of both:
- **Lookback:** the row and column in int32. The request index, computed count and token
  are predicated loads (`@p ld`, masked lanes keep 0 or -1), and the store is predicated.
  - Threads a block: 32 or 128.
  - A block covers an element range, or a block takes a row.
- **zero_kv_blocks:** only the page size before the exit, and every store predicated. The
  stores are either int32s in Triton's layout or 16-byte stores when the segment's address,
  stride and page allow.
- **Combine** (vLLM `tilelang-ports` 742eb9bf5, in the overlay):
  - Triton loads a request's drafts and confidences only when it has drafts.
  - Ours hoisted them into the first round of loads, so a 16K prefill chunk's request
    waited on rows it never used (+0.08 µs).
  - They are now predicated loads (`_LOAD_IF`) on the draft count, which arrives in that
    same first round. Decode keeps its single round, and the prefill chunk issues none.
  - The production kernel is timed with and without dead rows.

### Min-p tile form, round 1 (2026-10-24 20:46 CET)

Every variant is bit-exact against Triton: the threshold-boundary seeds and the non-finite
rows.

| Rows | Triton (µs) | Ours, best shape (µs) | Change |
| --- | --- | --- | --- |
| 1 | 67 | 18-20 | -70 to -74% |
| 16 | 112 | 59-60 | -47% |
| 96 | 517-523 | 523-527 | +0.2 to +1.5% |

- At 1 and 16 rows, Triton's maximum pass reduces across the whole block on every tile, two
  barriers each time; ours reduces once.
- At 96 rows the vector tiles are level at 4096 columns: +0.21% at 256 threads, +0.32% at
  128. Every other vector shape is +0.8 to +1.5%.
- Scalars (`run` 1) are +2 to +8% at 96 rows.
- Both kernels read the 50 MB batch twice. That is more than the L2 holds, so the threshold
  pass reads it from DRAM again.

**Round 2** (vLLM 80a577a81) takes the threshold pass last tile first. It starts on the
tiles the maximum's pass read last, the likeliest still in L2. The order of an elementwise
pass does not change its bits. Shapes: vectors, 128 or 256 threads, 2048 or 4096 columns,
forward or reversed.

## Eleventh integrated run (run `tilelang-integration-kernels-11`)

vLLM `tilelang-ports` c7ee05a5f:
- min-p in tile form (above);
- combine loads a request's drafts only when it has them (742eb9bf5);
- `benchmark_sampling_auxiliary.py` times the other seven generation-controls candidates
  against Triton, each first compared bit for bit: logit bias, forbidden sequences, the
  reasoning-marker cache, the thinking budget, prompt-logprob token ids, custom logprob
  token ids and the NaN count. The sampling bundle runs it with `--gate`.

[controls_digest.py](controls_digest.py) is the serving workload for switching the
controls. Each scenario sets one control the default requests never use: penalties at
temperature 0 and seeded 1, min-p, logit bias, bad words, a thinking budget, prompt logprobs
and custom logprob ids. It takes digests of the eight prompts alone and eight at once. A
build with the controls switched must print candidate 4's digests.

### The alternating eight-stream re-measure (2026-10-24 21:06 CET, run `tilelang-ports-c8-abab`)

Eight distinct prompts, eight samples an arm, in the order ports, candidate 4, ports,
candidate 4:

| Arm | tok/s | ±95% |
| --- | --- | --- |
| ports | 191.40 | 0.73% |
| candidate 4 | 190.70 | 0.57% |
| ports again | 191.65 | 0.58% |
| candidate 4 again | 190.16 | 0.38% |

**The ports are not slower at eight streams.** They are level, or +0.6% on the means.

Every arm's samples fall in two modes, about 189.5 and 192.8 tok/s, and both builds land in
both. The serving check's -1.7% and the bracket's -0.7% came from which mode their two or
three samples hit. This agrees with the profile, which shows equal step times.

Round 5 stopped at the bench's guard that the variants compile with production's pass
configs. Production has turned TileLang's bind replay off since vLLM 0441ba1be, and
runner_variants.py had not. With the configs aligned, it runs again as
`tilelang-runner-round5b`.

### Eleventh integrated run (2026-10-24 21:18 CET)

**Tests:** every test passes in every bundle, the controls' 141 included. The verdicts read
FAIL only because two gates flagged kernels (below), and because the min-p sweep's bundle
uses the "exit" verdict for a script without a pytest summary.

**Min-p, round 2.** Every variant is bit-exact against Triton. Taking the threshold pass last
tile first:

| Shape | 96 rows (µs) | Triton (µs) | Change |
| --- | --- | --- | --- |
| 256 threads, 2048 columns | 452.4 | 517.4 | -13.5% |
| 128 threads, 4096 columns | 454.6 | 516.7 | -12.2% |
| 256 threads, 4096 columns | 460.4 | 521.7 | -12.0% |

- Forward-order variants of the same shapes are -0.2 to +1.3%.
- Even scalars, reversed, are -9.4%.
- At 1 and 16 rows it stays -46 to -74%.
- Production takes 256 threads and 2048 columns, always reversed (vLLM c7840d48b). The
  controls gate times it at -74% / -44% / -15% at 1 / 16 / 96 rows.

**The other seven generation controls** (`benchmark_sampling_auxiliary.py`, each case
first compared with Triton bit for bit):
- **Level or faster everywhere:**
  - forbidden sequences: -2 to -77%, the most at 128 words and 96 rows;
  - the reasoning-marker cache: up to -49% when it resumes a 16K reasoning span;
  - the thinking budget: -35 to -39% with a 16-token end phrase;
  - custom logprob ids: level;
  - the NaN count: -8 to -47%, level for FP32 at 96 rows.
- **Slower:**
  - Prompt-logprob ids at a 16K chunk: 22.5 against 10.3 µs. Each 1024-token block waited
    on its own branchy loads.
  - Logit bias at 96 rows, with 300 biases and with min_tokens under structured output:
    +2.8%.

Both slow kernels are now in tile form (vLLM 5845b392f):
- Each list's ids and values are loaded, predicated on its length, before the gathered
  logits, and those before any store.
- The prompt ids take 4096 tokens a pass.
- **Bias NaN fix:** the bias kernel's all-masked check now ignores NaN, as Triton's
  maximum does. #524's kernel counted a NaN logit as alive, so a row whose only unmasked
  logit is NaN did not get its stop tokens back. A new test pins this against Triton.

**Dispatch.** The generation controls now dispatch to TileLang:
- the penalties and the histogram (987ef4337; the callers still clear the selected rows);
- everything else (3bb713c4f).

The Triton kernels stay as the tests' references. The thinking budget and the NaN count
are now declared at startup; the NaN count only when NaN counting is on.

## Twelfth integrated run (run `tilelang-integration-kernels-12`)

vLLM `tilelang-ports` 3fe4099d1: the controls dispatched, and zero_kv_blocks and the lookback in Triton's form (runner round 5b: level at every shape where production was flagged). The sampling bundle also
runs the bad-words and rejection-sampler tests, which exercise the switched states. If it
passes, [serving-ports-2.json](serving-ports-2.json) serves the branch beside candidate 4,
in alternating arms with eight eight-stream samples each. It checks:
- temperature 0 and seeded temperature 1;
- the controls' digests ([controls_digest.py](controls_digest.py),
  [tables_arms.py](tables_arms.py)).

**The next fusion target.** At eight streams the step still launches about 220 PyTorch
kernels: copies, top-k, sorts and scans, 0.4-0.45 ms a step. There is no Triton kernel
left on the temperature-0 path.

### Twelfth integrated run (2026-10-24 21:33 CET)

**Tests:** every test passes:
- runner 539;
- misc 211, plus vocabulary-parallel 12;
- mHC 60;
- sampling 661;
- controls and switched states 158.

**Gates:**
- **Pass:** mHC, misc, token counts, sampling, the controls (min-p, penalties,
  histogram).
- **The runner** flags prepare_prefill at one stream, combine with dead rows at the 16K
  prefill and expand at one stream. Each is under 0.1 µs, and the set changes from run to
  run. zero_kv and the lookback are no longer flagged.

**The other controls**, after the tile-form rewrites (5845b392f):

| Kernel | Change against Triton |
| --- | --- |
| Logit bias | -12 to -36% at most shapes |
| Prompt ids, one 16K chunk | -38% |
| Forbidden sequences, thinking, NaN count | level or faster |

Still slower:
- logit bias with min_tokens under structured output, 96 rows: +1.4%. Its row scan read
  scalars where Triton reads 16-byte vectors.
- prompt ids at 16 rows × 1024 tokens: +0.15 µs. Each pass issued 32 predicated loads a
  thread even for a 1024-token chunk.

**Fixes** (vLLM 253cb4719):
- the bias kernel fills and scans whole row tiles as vectors when the rows are aligned (a
  `run` key, as min-p's);
- the prompt-id gather skips the 1024-token groups a chunk does not reach, on its length
  alone.

They go into the thirteenth run (`tilelang-integration-kernels-13`), then the second serving
check (`tilelang-ports-serving-2`) serves the branch at 253cb4719.

### Thirteenth integrated run (2026-10-24 21:41 CET)

**Tests:** every test passes in every bundle.

**Gates:**
- **The whole sampling bundle passes**, every generation control included:
  - logit bias is now level or faster at every shape;
  - prompt ids too.
- **Runner:** still flags three small gaps that recur in every run:
  - prepare_prefill_inputs at decode: +0.1 µs at one stream, +0.3 at sixteen;
  - combine with dead rows at the 16K prefill: +0.05 µs.
- **Resampled token**, one request, mixed temperatures: +0.045 µs in all four runs 10-13.
  It is flagged when the interval's lower bound reaches zero.

## Fourteenth integrated run (run `tilelang-integration-kernels-14`)

vLLM `tilelang-ports` 12d6d2e23:
- **prepare_prefill_inputs:**
  - At decode every block exits after three loads, as Triton's does, but ours had eight
    warps a block where Triton has four. The exit paths match in SASS.
  - Round 3's thread sweep never tested this alone: its variants also moved the query
    starts ahead of the exit.
  - It now takes Triton's shape: four warps, eight tokens a thread, loads and stores
    predicated.
- **Combine:** its index and input-id stores are predicated instead of branching.
- **The resampled token:**
  - It took five dependent rounds of loads: count, state, temperature, block maxima, and
    the winner's id.
  - The maxima and ids depend only on the row, so they now load with it.
  - The warp reduction carries the winner's id, which leaves three rounds.
  - Ties and NaN behave as before: the lowest index wins a tie, and NaN counts as -inf.

**Min-p and logit bias are never served in production.** The second serving check's
controls digest stopped at its min-p scenario with HTTP 400. vLLM rejects `min_p` and
`logit_bias` under speculative decoding: "The min_p and logit_bias sampling parameters
are not yet supported with speculative decoding". Production serves DSpark, so those two
kernels run only in configurations without speculation. Their kernel gates and Triton
comparisons stand.

[controls_digest.py](controls_digest.py) now records a rejected scenario, with the server's
message as its digest, and continues. [serving-controls.json](serving-controls.json) runs
the controls digests alone: ports, candidate 4, then ports again.

### The second serving check (2026-10-24 22:17 CET, run `tilelang-ports-serving-2`)

vLLM `tilelang-ports` 253cb4719 (every port, every generation control dispatched) beside
candidate 4, in four alternating arms.

**Outputs:**
- Temperature 0 is identical in every scenario within every arm: eight at once,
  staggered, mixed with a long prefill, fifteen at once, cached, again.
- The digests are equal across all four arms.
- The seeded temperature-1 reference digest is equal across all four arms.
- The penalties scenarios are equal across all four arms:
  - at temperature 0, alone and eight at once;
  - seeded, alone.
- Seeded penalties eight at once differ between two runs of the same build (ports against
  ports again), so those depend on how the requests batch, not on the build.

**Speed:** level.
- One stream: 93.50 and 93.57 tok/s for the ports, against 93.02 and 93.21.
- Eight streams, eight samples an arm:
  - ports 191.14 and 190.76 tok/s;
  - candidate 4 190.95 and 191.56 tok/s;
  - the samples fall in the same two modes as before.

The remaining controls scenarios (bad words, the thinking budget, prompt logprobs and
custom logprob ids, plus the two rejections) run in `tilelang-ports-controls`.

### Fourteenth integrated run (2026-10-24 22:24 CET)

Every test passes.

**Now level or faster:**
- zero_kv_blocks, the lookback and combine (both forms; dead rows at the 16K prefill
  +0.002 µs);
- the resampled token's mixed case: +0.019 µs, level. It was +0.045.

**Still flagged:** prepare_prefill_inputs at sixteen requests, +0.27 µs. Four warps did
not change it, and neither did round 4b's ordinary loads.

The benchmark's "decode" requests are in fact all prefilling, six-token chunks: its
prefill lengths exceed every computed count. So the kernel copies, and every block runs
four dependent rounds of loads, the last from the pinned host history. Our predicated
stores order memory, and the lookahead token's load came after the chunk's stores. That
is one more round than the four, at L2 latency because the lookahead sits in the chunk's
line.

vLLM 04e097902 loads the lookahead token with the first pass, and a pass takes four 1024-token
groups (those the chunk reaches) at eight warps. It runs as
`tilelang-integration-kernels-15`.

### The controls-only serving run (2026-10-24 22:38 CET, run `tilelang-ports-controls`)

Ports, candidate 4, then ports again.

**Equal in all three arms:**

| Scenario | Alone | Eight at once |
| --- | --- | --- |
| Penalties, temperature 0 | equal | equal |
| Penalties, seeded | equal | (batching) |
| Forbidden sequences | equal | equal |
| Thinking budget | equal | equal |

- Seeded penalties eight at once differ run to run, as before.
- Min-p and logit bias are rejected (400) with the same message in every arm.

**Prompt logprobs kill the engine in both builds.** The request returns HTTP 500, and
EngineCore dies at `scheduler.py`'s `assert not prompt_logprobs_tensors` ("EngineCore
returns no partial prefill outputs"). It is candidate 4's bug, not the ports'. Production
probably has it too, but was not tested, since one request would take it down.
- **Collateral:** the last arm's dead engine kept its containers up on dgx2-4, and the
  lab's restore refused to start production over them. Production was restored by hand
  (22:41 CET, doctor --live OK): the arm stopped with its own profile, then the lab's
  window close.
- The fix, and the lab's handling of a dead arm, are their own work.
- The scenario is out of [controls_digest.py](controls_digest.py) until then.
- Custom logprob ids, which came after it, are still to be served.

### Fifteenth integrated run (2026-10-24 22:46 CET)

Every test passes. The mHC, misc and sampling gates pass, including all of the generation
controls.

prepare_prefill_inputs:
- On the 16K chunk it is now **twice as fast as Triton**: 4.36 against 8.98 µs.
- At decode it is slower: +0.27 µs at one stream and +0.57 at sixteen (an interval that
  crosses zero).

Its decode gap has moved between runs as the shape changed:

| Run | Shape | One stream | Sixteen |
| --- | --- | --- | --- |
| 13 | 256 threads, branchy | +0.10 µs | +0.30 µs |
| 14 | 128 threads, tile form | +0.10 µs | +0.27 µs |
| 15 | 256 threads, four groups a pass, lookahead hoisted | +0.27 µs | +0.57 µs |

expand_idx_mapping is +0.03 µs at one stream.

**Runner round 6** (`runner-round6-kernels.json`) times the shapes head to head, paired
against Triton:
- threads 128 or 256;
- 4 or 8 tokens a thread;
- 1, 2 or 4 groups a pass;
- the lookahead token loaded with the first pass or after the stores.
