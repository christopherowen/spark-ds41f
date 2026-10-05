# r5p: native drafter heads, packed head and the TP4 1M recipe

Base: main `c3b0595` with the TP4 1M recipe branch (`tp4-memory-tuning`,
`64f30e5`) merged.

r5p promotes the B12X work since r5o. The TileLang kernel family, merged
after r5p as an alternative kernel backend, was benchmarked against these same
configurations in [TP4](../2026-10-05-tp4-validation/README.md) and
[TP3](../2026-10-05-tp3-benchmark/README.md).

## What changes

One image serves both topologies: `vllm-ds41f-kkref:04c30fa98e79-r5p`
(`sha256:4995e0d3`, built on 2026-10-04 at 23:46 UTC as
`-r5o-roce-contract-tp4-1m-v1` from deployment commit `0f665a9`). Its source
manifest is [2026-10-05-r5p-source.json](../../manifests/sources/2026-10-05-r5p-source.json):
vLLM tree `eaa33543`, B12X tree `7f666380`, NCCL tree `6af10aa7`. The promoted
series apply the same patch bytes in the same order as the benchmarked
experiment series.

| Change | Patches and settings | From |
| --- | --- | --- |
| The DSpark draft head and Markov projection keep the checkpoint's BF16 tensors; r5o re-quantized both to NVFP4 at load | `VLLM_DS41_DRAFT_NVFP4_HEAD=0`, `VLLM_DS41_MARKOV_NVFP4=0` | [native drafter heads](../2026-10-03-native-drafter-heads/README.md) |
| The target vocabulary head, which the draft head shares, is served from an exact 12-bit packed form of its BF16 weights | B12X 0012, vLLM 0028, `VLLM_DS41_PACKED_BF16_LM_HEAD=1` | [packed BF16 head](../2026-10-03-packed-bf16-head/README.md) |
| The vocabulary weights load straight into the display carve-out, without a copy in ordinary memory | vLLM 0030 | [streamed embeddings](../2026-10-04-streamed-embeddings/README.md) |
| The requested RoCE collective policy is enforced instead of falling back silently | vLLM 0027 | [collective contract](../2026-10-03-collective-contract/README.md) |
| The fabricated profiling context uses its own blocks for each KV cache group and request | vLLM 0039 | [TP4 memory tuning](../2026-10-04-tp4-memory-tuning/README.md) |
| Four-node transports: the RoCEnante ring relay, NIC-forwarded mesh, four-path mesh, bidirectional relay and dispatch capacity; NCCL bidirectional switchless rings and balanced channels. They are inactive on TP3's direct transport | B12X 0006-0011, NCCL 0002-0004 | [balanced policy](../2026-10-03-balanced-policy/README.md) |

**TP3** (`config/cluster.json`, `config/cluster-64k.json`): r5o's shape with
the r5p image and its own DSpark cost curves (`tp3-b12x-512k-20261005`). The
shape is unchanged: 524,288 tokens per request, eight sequences, 3.5 GiB of
KV per rank (2,845,543 tokens), 4,096-token batches and CUDA graphs to 48
rows. The configuration equals the benchmarked one
([runs/tp3-b12x.json](runs/tp3-b12x.json)) apart from four things:

- the production deployment path and branch;
- the site node map;
- the readiness timeout;
- no torch profiler endpoint, which is a lab facility.

`config/cluster-4k.json` takes the same image and settings at its own
limits. It was not re-measured.

**TP4** (the `tp4` tuning profile): the
[1M recipe](../2026-10-04-tp4-memory-tuning/candidate.json):

- 1,048,576 tokens and 16 sequences;
- 8,192-token batches and CUDA graphs to 96 rows;
- 10.5 GiB of KV per rank (8,580,566 tokens, 8.18 full windows);
- prefix-cache checkpoints every 8,192 tokens;
- the balanced ring transport.

The benchmarked configuration is the recipe with launch enabled
([runs/tp4-b12x.json](runs/tp4-b12x.json)). `bin/spark3 tuning create tp4`
materializes it for a site's ring.

## Evidence

The owner's acceptance is one benchmark per configuration at its limits:

- **Suites:** quality; decode on prose and code with reasoning at 1-8 streams
  (TP3) or 1-16 (TP4), three samples, temperature 0, 256 output tokens;
  prefill on real source text, two repeats.
- **Windows:** both ran on 2026-10-05 from `bin/spark3 bench`:
  - TP4 on the ring, 00:10-00:26 UTC: [report](runs/tp4-bench.json), [summary](runs/tp4-bench.txt);
  - TP3 on the triangle, 01:10-01:22 UTC: [report](runs/tp3-bench.json), [summary](runs/tp3-bench.txt).
- **Report paths:** the reports name the configurations by their paths
  (`experiments/2026-10-05-tp3-benchmark/b12x.json`,
  `experiments/2026-10-05-tp4-validation/b12x.json`). The copies in `runs/`
  have the same SHA-256 as those files.

Both configurations passed quality 5/5 and saw no thermal slowdown.

**TP3 against r5o** (the 2026-10-02 64 KiB reference):

| Aggregate tok/s | 1 | 2 | 4 | 8 |
| --- | ---: | ---: | ---: | ---: |
| Prose, r5o | 52.8 | 79.4 | 117.9 | 171.2 |
| Prose, r5p | 50.4 | 75.9 | 118.8 | 173.3 |
| Code, r5o | 64.8 | 90.7 | 135.6 | 195.8 |
| Code, r5p | 60.1 | 90.8 | 136.3 | 192.3 |

| | r5o | r5p |
| --- | ---: | ---: |
| Single-stream step, prose / code | 40.84 / 45.36 ms | 41.92 / 46.44 ms |
| Prefill, real text, 32K / 256K | 3,768 / 3,684 tok/s | 3,778 / 3,632 tok/s |
| Prefill, real text, 500K | | 3,432 tok/s |
| Lowest MemAvailable, dgx1 / dgx2 / dgx3 | 5.83 / 7.59 / 7.60 GiB | 5.93 / 7.11 / 7.19 GiB |

The bench rates every point the same as r5o within its interval. The one
consistent direction is single-stream steps, 1.1 ms (2.6% and 2.4%) longer.
That is the price of the native heads: the drafter now reads BF16 heads where
r5f's NVFP4 re-quantization saved about 2.2 ms per step, and the packed head
recovers part of that. Serving weights stay as the checkpoint stores them, as
the owner requires.

**TP4 recipe:**

| Aggregate tok/s | 1 | 2 | 4 | 8 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prose | 62.1 | 92.7 | 141.6 | 215.7 | 297.3 |
| Code | 72.5 | 111.3 | 168.3 | 245.7 | 323.5 |

| | |
| --- | --- |
| Single-stream step, prose / code | 34.00 / 37.72 ms |
| Prefill, real text, 32K / 256K / 1M | 5,117 / 4,873 / 4,026 tok/s |
| Lowest MemAvailable, dgx1-dgx4 | 20.4 / 22.2 / 22.2 / 21.3 GiB |

Against r5o's TP3 reference, the recipe decodes 12-26% more at one to eight
streams and prefills 32-36% faster at 32K and 256K.

## Not in r5p

- **vLLM 0041 and 0042**, found while starting TileLang at TP3: dummy layouts
  give each KV cache group its own blocks, and the drafter marks its padding
  rows for the routers. B12X's router routes the rows they protect without
  complaint, so B12X serves without them. Adding them needs an image build
  and a benchmark. They are in the TileLang experiment's series
  (`experiments/2026-10-04-tilelang-1m/vllm/`).
- **The TileLang kernel family** (TileLang, TileKernels, sparknet and the
  kernel backend policy) is an alternative backend, not the r5p default.
- **The bench precision set** (PR #6) has not run against a live service yet.

## Deployment

Every node tags the image as `vllm-ds41f-kkref:04c30fa98e79-r5p`. The tag adds
a name and leaves the existing tag in place. The site `config/nodes.json` must
describe the triangle's current CX7 addresses (10.12, 10.13 and 10.23). The
DSpark cost curves are on dgx1 under `cache/kkref/dspark-costs/`. Production
starts from `main` through `bin/spark3 cluster sync --apply` and
`cluster start --apply`.
