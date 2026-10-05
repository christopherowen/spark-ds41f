# r6: TileLang as the promoted kernel family

r6 is the TileLang kernel family (TileLang, DeepSeek's TileKernels and
sparknet's one-shot collectives) as the promoted backend, on the TP3 recipe
(eight sequences at 512K) and the TP4 1M recipe, built as one image with no
files mounted over it.

## Sources

The [lock](upstreams.lock.json) and [source manifest](source.json) are the
TileLang 1M build's ([2026-10-04-tilelang-1m](../2026-10-04-tilelang-1m/README.md))
with the vLLM [series](vllm/series) extended by:

- 0041 (fabricated profiling context: its own blocks per KV group) and 0042
  (the DSpark drafter marks its padding rows for the MoE routers), both from
  the TileLang 1M directory, served until now as mounted files;
- [0043](vllm/0043-tilelang-decode-tiles.patch): block-32 FP8 decode rows on
  race-free 16-, 32- and 64-row tiles swept at the TP4 serving shapes, and
  swizzled prefill for weights larger than L2
  ([decode-kernel experiment](../2026-10-05-tilelang-decode-kernels/README.md));
- [0044](vllm/0044-tilelang-prefetch-scale-words.patch): the TileLang scale
  words join each projection's weight in the L2 prefetch.

The series reproduces vLLM tree `20c7c758` (patch head `cf7703fb`). The image
is `vllm-ds41f-kkref:04c30fa98e79-r6`.

## Configurations

[make_configs.py](make_configs.py) writes the benchmark configurations from
the 2026-10-05 TileLang recipes
([TP4](../2026-10-05-tp4-500k/tilelang.json), [TP3](../2026-10-05-tp3-benchmark/tilelang.json)):
[tp4-profile.json](tp4-profile.json) and [tp3-profile.json](tp3-profile.json),
the r6 image with nothing mounted over it, their own DSpark cost directories
(`ring4-r6-20261005`, `tp3-r6-20261005`) and the torch profiler endpoints for
each boot's decode profile.

The promoted profiles are the same recipes without the profiler, on the root
lock: [config/cluster.json](../../config/cluster.json) (TP3, with
`cluster-64k` and `cluster-4k`) and
[config/cluster-tp4.json](../../config/cluster-tp4.json) (TP4).

## Benchmark

[run_benchmark.sh](run_benchmark.sh): one boot per recipe; quality, the
single-stream decode profile, decode on prose and code with reasoning (three
samples) and real-text prefill (two repeats) at the recipe's limits: TP4 at
1-16 streams and 32K-1M, TP3 at 1-8 streams and 32K-500K.

Each value is the mean ± its 95% interval (three decode samples, two prefill
repeats); **bold** marks a point whose interval does not overlap B12X r5p's
(its [promotion benchmark](../2026-10-05-r5p-promotion/README.md), and
[2026-10-05-tp4-500k](../2026-10-05-tp4-500k/README.md) for B12X's TP4 500K).

**TP4**, the four-node ring, 2026-10-05 11:48-12:10 UTC
([runs-tp4](runs-tp4)): quality 5/5, every node at least 19.35 GiB
available, no thermal slowdown. The 1M prefill point ran on the same boot
afterwards with the benchmark's size 1,000,000 (1,048,576 is skipped because
it exceeds the context once output is added).

| | 1 | 2 | 4 | 8 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prose, B12X r5p | 62.1 ± 3.8% | 92.7 ± 8.0% | 141.6 ± 7.9% | 215.7 ± 7.5% | 297.3 ± 1.2% |
| Prose, r6 | 62.4 ± 1.2% | 100.2 ± 2.7% | 147.1 ± 10.2% | 222.1 ± 0.3% | **318.4 ± 1.6%** |
| Code, B12X r5p | 72.5 ± 7.5% | 111.3 ± 5.4% | 168.3 ± 9.6% | 245.7 ± 1.3% | 323.5 ± 1.5% |
| Code, r6 | 76.0 ± 0.6% | 116.8 ± 18.9% | 174.9 ± 0.6% | **264.3 ± 0.6%** | **349.8 ± 2.5%** |

| Prefill, real text (tok/s) | 32K | 256K | 500K | 1M |
| --- | ---: | ---: | ---: | ---: |
| B12X r5p | 5,117 ± 1.7% | 4,873 ± 2.5% | 4,645 ± 1.7% | 4,026 ± 0.9% |
| r6 | **5,836 ± 4.2%** | **5,471 ± 3.1%** | **5,072 ± 1.3%** | **4,370 ± 0.6%** |

Single-stream steps: **31.79** (prose) and **34.75 ms** (code) against
34.00 and 37.72.

**TP3**, the dgx1-dgx3 triangle after recabling, 2026-10-05 12:16-12:29 UTC
([runs-tp3](runs-tp3)): quality 5/5, dgx1 at least 5.99 GiB available, no
thermal slowdown.

| | 1 | 2 | 4 | 8 |
| --- | ---: | ---: | ---: | ---: |
| Prose, B12X r5p | 50.4 ± 12.2% | 75.9 ± 10.6% | 118.8 ± 12.9% | 173.3 ± 8.4% |
| Prose, r6 | 52.2 ± 0.7% | 84.4 ± 0.6% | 125.7 ± 0.5% | **198.6 ± 0.4%** |
| Code, B12X r5p | 60.1 ± 2.0% | 90.8 ± 5.0% | 136.3 ± 6.3% | 192.3 ± 3.2% |
| Code, r6 | **63.0 ± 0.5%** | 97.3 ± 8.5% | 159.2 ± 44.2% | **240.3 ± 1.0%** |

| Prefill, real text (tok/s) | 32K | 256K | 500K |
| --- | ---: | ---: | ---: |
| B12X r5p | 3,778 ± 14.9% | 3,632 ± 2.2% | 3,432 ± 0.6% |
| r6 | 4,306 ± 20.9% | **4,034 ± 0.3%** | **3,743 ± 0.8%** |

Single-stream steps: **38.21** (prose) and **42.43 ms** (code) against 41.92
and 46.44.

r6 is behind on no point. B12X's intervals are wider because its text changes
between samples; r6 generates the same text each sample, and its wide points
(two- and four-stream code) come from scheduling that interleaves streams
differently between samples.

The TP4 boot's rank-0 decode profile (`ring4-r6-20261005`) shows the kernel
set of the decode-kernel experiment's window 3 (the same patches mounted
over the earlier image), with a 40.5 ms six-row step against B12X's 44.9;
its routed-expert calls read slower than window 3's (297 against 250 µs)
while the benchmark's step times were not.

## Promotion

r6 becomes the promoted baseline
([manifests/baselines/2026-10-05-karmic-kraken-r6.json](../../manifests/baselines/2026-10-05-karmic-kraken-r6.json)):
the TileLang patches join `patches/vllm` (series-r5p keeps r5p's series),
the root lock and source manifest become r6's, and `config/cluster*.json`
become the TP3 recipe with `kernel_backend: tilelang`, `config/cluster-tp4.json`
is the TP4 1M profile, and [r5p/](r5p) keeps
r5p's lock and TP3 configuration, which the B12X tuning profiles and the
topology tests use.
