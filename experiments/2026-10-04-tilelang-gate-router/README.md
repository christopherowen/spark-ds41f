# TileLang gate router: the gate's reduction fused into routing

Base deployment commit: the [TileLang router](../2026-10-04-tilelang-router/README.md)
experiment (vLLM patch 0032, TileKernels' top-k gate).

At decode the TileLang family computed each MoE layer's gate in three
launches:

1. the BF16 gate projection's split-K partials;
2. `splitk_reduce`, writing FP32 logits;
3. the router reading them back.

Two vLLM patches shorten that, and a third keeps both routers off padding rows.

**[0033](vllm/0033-tilelang-splitk-row-tiles.patch): row tiles sized to the
rows.** The split-K partials kernel always used 64-row tiles. At one to six
decode rows that takes twice as long as a 16-row tile: 9.2 µs against 4.3 µs
for the 384-expert gate. Every tile height gives the same partials bit for
bit, so each call takes the smallest of 16, 32 and 64 rows that fits. This
applies to every TileLang BF16 projection that splits K, not only the gates.

**[0034](vllm/0034-tilelang-fused-gate-router.patch): the gate's reduction
fused into the router.** `TileLangGateRouter` owns the gate. The MoE layer
hands it to the router instead of the runner, which then passes the hidden
states. At decode, `route_partials` adds the partials in shard order, which
reproduces `splitk_reduce`'s logits bit for bit, and routes them in the same
launch with TileKernels' arithmetic:

- each round's warp-wide winner takes two integer reductions instead of ten
  shuffles: the score as an integer that orders like the float, then the
  lowest expert id holding it;
- non-finite logits are refused once per lane.

The logits never reach memory, and each MoE layer runs one launch fewer.
Above 64 rows the gate runs its GEMM and TileKernels routes the logits.

**[0035](vllm/0035-tilelang-routing-skips-padding.patch): padding rows left
unrouted.** A CUDA graph runs its captured batch size, so a step with one
decode row (one token plus its drafts) can carry padding rows. vLLM's own
routers give those rows id −1 and weight 0 when `VLLM_MOE_SKIP_PADDING` is on
(the default), and the MoE dispatch then skips them. Neither TileLang router
did, so every padding row dispatched six experts of work. Both routers now
take the step's padding mask. It is computed once per step and cached on the
forward context, like the image mask, and masked rows write −1 and 0 without
reading their logits.

[fused.json](fused.json) is the candidate, launchable, with the router arm's
cost directory.

## Kernel results

The [tests](bundles/tests/candidate.json) pass 58 of 58, overlaid on the
step-one image:

- the fused gate router's ids and weights are bit-identical to the gate
  followed by TileKernels' router, at 1–64 rows, for both routers and for
  image tokens;
- above 64 rows it matches the gate GEMM followed by the router;
- every row-tile height matches the full batch bit for bit.

[Gate plus routing per MoE layer](bundles/gate-route-bench/candidate.json), under CUDA graphs:

| Rows | Before 0032: gate (64-row tiles), Triton | Gate (row tiles), Triton | Gate, TileKernels (0032) | Fused (0034) |
| ---: | ---: | ---: | ---: | ---: |
| 1, 384 experts | 12.09 µs | 7.52 µs | 9.60 µs | 8.00 µs |
| 6 | 11.89 µs | 7.43 µs | 9.37 µs | 7.99 µs |
| 16 | 8.80 µs | 6.84 µs | 8.92 µs | 7.23 µs |
| 64 | 9.70 µs | 9.45 µs | 11.44 µs | 9.57 µs |
| 1, 128 experts (drafter) | — | — | 4.92 µs | 3.80 µs |

At single-stream decode, each target MoE layer's gate and routing take
8.0 µs, down from 12.0 µs before 0032.

## Serving results

Window 1 (`-tilelang-gate-router-v1`, before 0035) and a second window with a
rebuilt control (`-tilelang-packedhead-v2`, [packed-v2.json](../2026-10-04-tilelang-router/packed-v2.json)):

| Single-stream step time | prose c1 | code c1 |
| --- | ---: | ---: |
| Control | 33.70 ms | 36.87 ms |
| Router (0032) | 37.75 ms (+12.0%) | 39.20 ms (+6.3%) |
| Fused (0033–0034) | 37.60 ms (+12.0%) | 39.05 ms (+6.3%) |

The router arm's eight-stream throughput was unchanged, and both router arms
carried the same fixed loss at one stream. In the single-stream profiles,
routing itself costs only 45 µs per step more than the Triton kernel. The rest,
1.3 ms per step, is TileLang kernels launched as often as in the control but
running longer: the padding rows of each captured batch were routed and
dispatched as real tokens. 0035 fixes that, and its tests check both routers on
a padded step.

## Procedure

Build `-tilelang-gate-router-v2` and run both bundles. Then one window boots
the control (`packed-v2.json`) and this arm in turn. Each gets the
single-stream decode bench against the control and the single-stream decode
profile; the fused arm then gets the eight-stream bench and the prefill bench.
