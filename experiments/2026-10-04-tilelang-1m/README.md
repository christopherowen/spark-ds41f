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

## Startup failure in the cost profile (0039)

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
