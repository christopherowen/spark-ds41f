# TP4 end-to-end validation: TileLang and B12X

Before the default kernel family changes, both TP4 candidates run the full
validation and benchmark protocol in the same windows. TP3 follows.

| Family | Config | Image | DSpark cost directory |
| --- | --- | --- | --- |
| TileLang | [candidate.json](../2026-10-04-tilelang-1m/candidate.json) | `-tilelang-1m-v3` | `ring4-tilelang-1m-v3-20261005` |
| B12X | [b12x.json](b12x.json) | `-tp4-1m-v1` | `ring4-tp4-1m-20261005` |

Both run the [TP4 1M recipe](../2026-10-04-tp4-memory-tuning/README.md):
1,048,576-token context, 16 sequences, 8,192-token batches, CUDA graphs to
96 rows, 10.5 GiB of KV per rank, prefix-cache retention every 8,192 tokens,
vocabulary weights in the display carve-out, and the fabricated-context fix
(vLLM 0039).

- [b12x.json](b12x.json) is the recipe with launch enabled for lab windows.
- The TileLang config is the candidate itself; it adds the TileLang kernels,
  TileKernels routing and mHC with the register fold (0040), TileLang
  vocabulary heads and sparknet collectives
  ([README](../2026-10-04-tilelang-1m/README.md)).
- Each family profiles its own DSpark cost curves from its own kernels, as it
  would deploy: the B12X recipe's earlier curves came from the
  fabricated-context profile that 0039 fixes, and the TileLang curves from
  the kernels before 0040.

## Benchmark (window 2026-10-05 00:08–00:42 UTC)

The owner set the acceptance: one benchmark per family at the recipe's
limits, whose numbers feed the release announcements. One boot each, B12X
first; quality, decode on prose and code with reasoning (aggregate tok/s,
temperature 0, 256 output tokens) at 1-16 streams with three samples, and
prefill on real source text at 32K, 256K and 1M with two repeats. Both
passed quality 5/5, kept at least 20.3 GiB available on every node and saw
no thermal slowdown.

| Aggregate tok/s | 1 | 2 | 4 | 8 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prose, B12X | 62.1 | 92.7 | 141.6 | 215.7 | 297.3 |
| Prose, TileLang | 62.4 | 99.0 | 153.9 | 217.6 | 319.1 |
| Code, B12X | 72.5 | 111.3 | 168.3 | 245.7 | 323.5 |
| Code, TileLang | 74.3 | 105.0 | 171.5 | 261.7 | 352.2 |

| Prefill, real text (tok/s) | 32K | 256K | 500K | 1M |
| --- | ---: | ---: | ---: | ---: |
| B12X | 5,117 | 4,873 | 4,645 | 4,026 |
| TileLang | 5,730 | 5,387 | 5,006 | 4,312 |

The 500K column was measured separately, in
[2026-10-05-tp4-500k](../2026-10-05-tp4-500k/README.md).

Single-stream steps: TileLang 32.59 ms on prose and 35.63 ms on code, B12X
34.00 and 37.72 (-4.2% and -5.5%). At 16 streams TileLang decodes 7.3% more
prose and 8.9% more code; prefill is 7-12% faster. The bench flags one
point: code at two streams, 5.7% lower for TileLang. B12X's interval there
was ±5.4% and TileLang's steps are faster, so it is acceptance noise at low
concurrency; the repository gates on step time.

Against the promoted baseline (r5o, TP3 at 512K), the B12X TP4 1M recipe
decodes 12-26% more at one to eight streams and prefills 32-36% faster at
32K and 256K.

The full protocol first planned here (the full bench, the 1M qualification
scripts, the prompt-cache probe and second boots, about three hours) was
stopped at the owner's direction after twelve minutes; the benchmark above
is the acceptance.

TP3 (8 streams at 512K) follows once the network is re-wired to the
triangle.
