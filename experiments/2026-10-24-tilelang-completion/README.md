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
| NCCL (the ring's row all-to-all, all-gather) | 2 | 0 | 0.821 | move into sparknet's TileLang collectives |
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
4. **The ring's collectives**: the row all-to-all and the remaining NCCL all-gather as sparknet
   TileLang collectives, so NCCL leaves the decode step.
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
