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
