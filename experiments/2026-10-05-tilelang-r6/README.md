# r6: TileLang as the TP4 kernel family

r6 is the TileLang kernel family (TileLang, DeepSeek's TileKernels and
sparknet's one-shot collectives) on the TP4 1M recipe, built as one image with
no files mounted over it.

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

[make_configs.py](make_configs.py) derives both from the TP4 TileLang recipe
([2026-10-05-tp4-500k/tilelang.json](../2026-10-05-tp4-500k/tilelang.json)):

- [tp4.json](tp4.json): the serving configuration, no profiler;
- [tp4-profile.json](tp4-profile.json): the same with the torch profiler
  endpoints, for the benchmark boot's decode profile.

Each uses its own DSpark cost directory (`ring4-r6-20261005`).

## Benchmark

[run_benchmark.sh](run_benchmark.sh): one boot; quality, the single-stream
decode profile, decode on prose and code with reasoning at 1-16 streams
(three samples) and real-text prefill at 32K, 256K, 500K and 1M (two
repeats), the TP4 acceptance benchmark's points.

Results: pending.
