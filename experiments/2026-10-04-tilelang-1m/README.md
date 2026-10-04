# TileLang family on the TP4 1M recipe

Base deployment commit: the [TP4 memory tuning](../2026-10-04-tp4-memory-tuning/README.md)
recipe, on [streamed embeddings](../2026-10-04-streamed-embeddings/README.md).

The TileLang kernel family was evaluated on the earlier TP4 recipe: 512K
context, 8 sequences, 3.5 GiB of KV per rank, 4,096 batched tokens. Since
then, the B12X candidate moved to the 1M recipe:

- 1,048,576-token context and 16 sequences;
- 8,192 batched tokens, with CUDA graphs captured up to 96 rows;
- 10.5 GiB of KV per rank, about 8 full 1M windows;
- prefix-cache retention every 8,192 tokens;
- vocabulary weights streamed straight into the GB10 display carve-out.

End-to-end testing should run the configuration that would ship, so this
experiment rebases the TileLang family onto that base.

**vLLM series.** The streamed-embeddings series (0001–0028, 0030) is
followed by the TileLang patches, rebased onto it:

| Patch | What it adds |
| --- | --- |
| [0029](vllm/0029-deepseek-v41-tilelang-kernels.patch) | kernel backend |
| [0032](vllm/0032-deepseek-v41-tilekernels-router.patch)–[0035](vllm/0035-tilelang-routing-skips-padding.patch) | TileKernels routing and the fused gate router |
| [0036](vllm/0036-tilelang-vocab-heads.patch) | vocabulary heads |
| [0037](vllm/0037-tilelang-tilekernels-mhc.patch) | TileKernels mHC |
| [0038](vllm/0038-tilelang-sparknet-collectives.patch) | sparknet collectives |
| [0039](vllm/0039-dummy-context-disjoint-blocks.patch) | fabricated profiling context with its own blocks per group and request |

Every TileLang commit applied without conflict. The packed head is packed
in place in the carve-out, and the TileLang projection reads it there. The
B12X tree is the recipe's own (`7f666380`), whose packing works in place.

**Recipe.** [candidate.json](candidate.json) is the 1M recipe with the
TileLang family's settings:

- `kernel_backend: tilelang` and TileLang attention, linear and MoE
  backends, the drafter's attention included;
- sparknet's `oneshot-ring4` transport with its `SPARKNET_ROCE_*` settings,
  at the B12X recipe's values;
- its own DSpark cost and profiler directories (`ring4-tilelang-ctx1m-20261004`).

The KV pool stays at the recipe's 10.5 GiB per rank. It qualifies only if
every node keeps at least 8 GiB available at its worst moment. The TileLang
arms kept 25–28 GiB at 3.5 GiB of KV.

**mHC.** In the [TileKernels mHC](../2026-10-04-tilelang-mhc/README.md)
window, the bench measured single-stream step time 1.6–2.3% slower than B12X's
mHC. The profile disagrees:

- the mHC kernels themselves were at parity (2.14 ms against 2.10 ms per
  scheduler step);
- the profiled run was not slower per step (39.91 ms against 40.10 ms wall);
- many unrelated kernels ran 1–3% slower in that boot.

[b12x-mhc/candidate.json](b12x-mhc/candidate.json) is the same recipe without
0037 (B12X's mHC). An A/B/A window measures the difference against the boot
noise.

## Startup failure in the cost profile (fixed by 0039)

The v1 images did not start. In DSpark's startup cost profile, TileKernels'
gate refused non-finite logits on five rows, in every arm, B12X's mHC
included. The [diagnostics](diagnostics/) narrowed it down:

| Arm | Change from the candidate | Start |
| --- | --- | --- |
| shapes8 | 8 sequences, 4,096 batched tokens, graphs to 48 rows | passes, quality 5/5 |
| seqs16-4k | 4,096 batched tokens | fails, same rows |
| ctx0 | profile without fabricated context | passes; quality 5/5, decode at 1, 8 and 16 streams finite |
| seqs12 | 12 sequences | passes |

The router-debug build (debug patch, non-finite counts per check and the
first sequence number at which each saw one) located the origin. All 40
target gates were finite. The rows were one dummy request's DSpark query
block (five rows of an 80-row drafter batch), in one profiled shape. The
first non-finite value was the drafter's first attention output, while its
query, KV and input were finite: the attention read non-finite cache bytes.

The profile fabricates 8,192 tokens of context per dummy request, and
`set_dummy_context` gave every KV cache group the same block ids, request
after request from block 0. The 17 groups draw from one pool and overlay
the same memory; the allocator gives a block to one group and request at
a time, and block 0 is the null block. Fabricated spans broke both rules,
so the drafter's sliding window read bytes another group had written in
its own format. Serving never shares a block. B12X's router routes
non-finite logits without complaint, so the B12X recipe starts with the
same reads in its profile.

0039 gives each request of each group its own whole blocks after the null
block (spans wrap only when the pool runs out). The
[poisoned-scratch](diagnostics/bundles/poisoned-scratch/candidate.json)
bundle also showed the TileLang projections and routed experts ignore
their scratch at TP4 shapes on both sides of the 64-row decode tile.

## Procedure

Build `-tilelang-1m-v2` and `-tilelang-1m-b12xmhc-v2` (0039 added; the v1
images do not start), and run the [tests](bundles/tests/candidate.json)
bundle. One window then boots the
TileKernels arm, the B12X-mHC arm and the TileKernels arm again: decode at one
and eight streams, prefill, and the memory log on every node for each.

## Results (window 2026-10-04 21:45–22:08 UTC)

The v2 images built, the tests bundle passed (99 tests), and every arm
started the full 1M recipe with the profile's fabricated context. All
three arms passed quality 5/5. The arms share one pinned DSpark cost
directory, as the first arm profiled it.

| Arm | prose c1 tok/s | code c1 | prose c8 | code c8 | prose c1 step ms | code c1 step ms |
| --- | --- | --- | --- | --- | --- | --- |
| TileKernels mHC | 63.2 | 78.8 | 219.3 | 254.8 | 33.36 | 36.66 |
| B12X mHC | 62.9 | 75.5 | 208.9 | 241.0 | 32.97 | 36.10 |
| TileKernels mHC again | 63.2 | 78.8 | 219.4 | 254.5 | 33.35 | 36.63 |

The two TileKernels boots agree within 0.1% on every point, so boot noise
is out of the comparison. B12X's mHC is 1.2–1.5% faster per single-stream
step, as the microbenchmarks predicted at small row counts. TileKernels'
mHC gets more tokens per step, though: 2.11 against 2.07 on prose and 2.89
against 2.73 on code. In aggregate it is ahead by 0.6% (prose c1, within
noise) to 5.4% (code c8). Prefill is level within noise.

At 16 streams the TileKernels arm decodes 316.5 tok/s of prose and 342.9 of
code. Against the TileLang packed-head arm on the 512K recipe, single-stream
step time is the same, eight streams are 2.7–3.0% faster, and prefill is
15–17% faster from 32K to 262K tokens, with 8,192-token batches.

MemAvailable never fell below 19.8 GiB on any node (dgx1; 20.0–21.3 GiB on
the others), so the 10.5 GiB KV pool qualifies against the 8 GiB floor.

**mHC recommendation:** keep TileKernels' mHC (0037). It follows DeepSeek's
reference arithmetic, and its acceptance more than pays for B12X's
per-step lead.

## mHC register fold (0040)

The decode profiles put TileKernels' mHC entry at about 20 µs per call in
serving, in its projection kernel. With fn cold in L2, as in serving (40
distinct fn and residual sets cycled inside one CUDA graph), timing the
kernel with parts removed showed where the time went. The post fold was
half of it: per stream, scalar shared loads of the staged streams, a staged
BF16 tile, a TMA store that waited for the write, and a reload into the GEMM
tile, with barriers between. The collapse's sum of squares then took sixteen
block-wide reductions. The GEMMs themselves were about 1 µs. Staging fn
asynchronously did not help: the projection already reads fn near the
memory bandwidth, and the earlier "fn prefetch had no effect" benchmark had
fn warm in L2.

[0040](vllm/0040-tilelang-mhc-register-fold.patch) gives each thread one
token's 16 contiguous columns. It folds the previous output into the
streams in registers (same expression and order), stores the updated
stream directly and writes the GEMM tile; both sums of squares reduce
across the token's eight lanes. The updated streams, GEMM partials and
collapse are unchanged bit for bit; the two sums of squares add in a
different fixed order, so the mixes move by about 1e-7 and y by at most one
BF16 ulp. The tests bundle passes (99).

Sublayer entry (projection and finalize), µs per call, fn cold:

| Tokens | 1 | 16 | 32 | 64 | 96 |
| --- | --- | --- | --- | --- | --- |
| 0037 | 18.5 | 20.9 | 31.1 | 49.8 | 67.3 |
| 0040 | 14.4 | 17.4 | 23.3 | 35.7 | 44.4 |

[register-fold/candidate.json](register-fold/candidate.json) is the
candidate with 0040 (image `-tilelang-1m-v3`), with the same pinned DSpark
cost curves, for an A/B against the v2 candidate in one window.
