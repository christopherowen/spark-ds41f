# TileLang vocabulary heads

Base deployment commit: the [TileLang gate router](../2026-10-04-tilelang-gate-router/README.md)
experiment (vLLM patches 0032–0035).

The TileLang family still projected its vocabulary heads with B12X's Triton
kernels. The packed BF16 target head was the largest Triton kernel in its
decode profile, 1.1 ms per target step. The DSpark Markov head, five calls per
step at about 84 µs, used a Triton row kernel for one row and cuBLAS above.

[0036](vllm/0036-tilelang-vocab-heads.patch) projects both with TileLang
when the linear backend is `tilelang`:

- **Packed BF16 head.** Each pipeline step copies two 128-column groups of
  packed bytes and codes into shared memory, decodes them to BF16 there, and
  runs a TileLang GEMM. Each thread decodes 16 columns of a group's low half
  and the same 16 of its high half, which share 16 code bytes, with 16-byte
  loads and stores. The out-of-window values are added after the dot
  product, in list order, to their own row only.
- **The copies run ahead of decoding.** Decoding and the GEMM share the last
  pipeline stage, so the decoded weights need one shared buffer instead of
  one per stage, which leaves room for four stages.
- **BF16 heads** (the 256-wide Markov projection) run TileLang's BF16 GEMM.
- **Token tiles** of 16, 32 and 64 rows each have their own configuration.
  Every head compiles its three kernels when it is prepared: a packed head
  when it is packed, a BF16 head when the logits processor registers it.

Each logit is its row's groups in order, then its exceptions. Neither the
token count nor the tile changes that order, so a token's logits are the same
bits in any batch. B12X's vocabulary projections now serve only the B12X
backend. Packing itself still runs B12X's Triton pass at load.

[candidate.json](candidate.json) is the gate router candidate on the
`-tilelang-vocab-heads-v1` image, launchable with the same cost directory.

## Kernel results

The [head bench](bundles/head-bench/candidate.json) through the prepared heads,
overlaid on the gate router image, under CUDA graphs, against B12X's Triton
projection with its default configuration:

| Rows | Packed head, TP4 shard (32,320 rows) | Packed head, TP3 shard (43,092 rows) |
| ---: | --- | --- |
| 1 | 1,123 → 1,096 µs (−2.4%) | 1,484 → 1,451 µs (−2.2%) |
| 6 | 1,132 → 1,097 µs (−3.1%) | 1,491 → 1,447 µs (−2.9%) |
| 16 | 1,145 → 1,107 µs (−3.3%) | 1,509 → 1,464 µs (−3.0%) |
| 32 | 1,139 → 1,137 µs (−0.2%) | 1,490 → 1,488 µs (−0.1%) |
| 48 | 1,187 → 1,246 µs (+5.0%) | 1,568 → 1,633 µs (+4.2%) |
| 64 | 1,187 → 1,204 µs (+1.5%) | 1,553 → 1,605 µs (+3.4%) |

Up to 16 rows the TileLang head reads its packed bytes at 224–229 GB/s. In
the 64-row tile (33–64 rows) every vocabulary tile reads the whole activation
block, and it stays a few percent slower than Triton's.

The Markov head, with weights cold in L2:

| Rows | Triton row kernel | cuBLAS | TileLang |
| ---: | ---: | ---: | ---: |
| 1, TP4 shard | 99.7 µs | 74.5 µs | 71.1 µs |
| 8 | — | 74.8 µs | 71.3 µs |
| 48 | — | 88.9 µs | 77.0 µs |
| 1, TP3 shard | 127.6 µs | 107.6 µs | 92.4 µs |

The [tests](bundles/tests/candidate.json) pass 23 of 23 for the heads.

## Procedure

Build `-tilelang-vocab-heads-v1` and run both bundles:
[tests](bundles/tests/candidate.json) and the
[head bench](bundles/head-bench/candidate.json), which measures the prepared
heads. Then one window boots the gate router candidate (the control) and this
arm in turn: decode at one and eight streams, prefill, and the single-stream
decode profile for each.
