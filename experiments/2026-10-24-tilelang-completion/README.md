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
