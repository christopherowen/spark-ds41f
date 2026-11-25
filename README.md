# spark-ds41f

**DeepSeek V4.1 Flash on DGX Sparks. No switch required.\***

A reproducible deployment of DeepSeek-V4.1-Flash-1017 across one to eight DGX Sparks:
the Docker/vLLM build, the tuning, and every benchmark behind it. The site,
[switchle.ss](https://switchle.ss), is rendered from this repository's manifests.

- **Native weights.** DeepSeek's checkpoint, unchanged: the published FP8/FP4/BF16 weights
  and tokenizer, loaded directly. Nothing is re-quantized.
- **Correct first.** DeepSeek's reference arithmetic term for term, temperature-0 outputs
  that do not depend on the batch, and a determinism gate and release tests that every
  promotion passes ([docs/testing.md](docs/testing.md)).
- **Then fast.** Every speed-up lands as a measured experiment against the release before
  it, and the unfavourable runs are kept ([docs/methodology.md](docs/methodology.md)).

| | |
| --- | --- |
| Context | 1,048,576 tokens on 3–8 Sparks (`262,144` on 1–2 Sparks and 4 KiB TP3) |
| Sessions | 32 at once on 4–8 Sparks (`192`-row CUDA graphs), 8 on 3 Sparks, 4 on 1–2 Sparks |
| Prompt cache | up to 27 full 1M contexts (`25.5 GiB`/rank on TP8) or 17 full 1M contexts on TP4 (`16.5 GiB`/rank; 5 on TP3; 34 full 256K contexts on TP1/TP2); every response reports its cache hits |
| Fabrics | single-node (`tp1`), switchless direct (`tp2-direct`), triangle (`tp3-triangle`, `-4k`), ring (`tp4-ring`), and switched\* (`tp2-switched` to `tp8-switched`) |
| Speed | DSpark and CopyDrafter speculative decoding, CED row skip, prefill yield, CUDA `13.5.37`, TileLang kernels, one-shot RoCE collectives, zero-copy `io_uring` expert slot cache (`TP1`/`TP2`); current numbers in [Performance](#performance) |
| Steering | [weightless](https://weightle.ss): a second served name, `deepseek-v4.1-flash-weightless`, steered per request by a GLP vector, with the stock name bit-identical beside it |
| API | OpenAI-compatible, with the DeepSeek API's effort aliases and `thinking` switch, plus OpenResponses / `CustomTool` support |

\* We do support switches: `tp2-switched` to `tp8-switched`, all qualified in
`spark-ds41f-1120` ([docs/switched-fabric.md](docs/switched-fabric.md)). We just never needed one.

Production is **spark-ds41f-1120**, serving DeepSeek-V4.1-Flash-1017 across all
twelve hardware shapes ([release record](experiments/2026-10-26-spark-ds41f-candidate/README.md#spark-ds41f-1120-candidate-62),
[release notes](experiments/2026-10-26-spark-ds41f-candidate/release-notes.md), frozen in
[releases/spark-ds41f-1120](releases/spark-ds41f-1120/)). It advances onto CUDA `13.5.37`, retires
B12X completely (`0802`), adds `tp1`, `tp2-direct`, `tp2-switched`, and `tp3-triangle-4k`, and
raises `TP4..TP8` concurrency from 16 to 32 streams with expanded KV pools.
[Release history](#release-history) has what each release changed and measured.

The promoted kernel family is **TileLang** (the only kernel family in the image;
B12X was retired in `vLLM 0802` / #1096): TileLang kernels for SM121 (`CUDA 13.5.37`),
with DeepSeek's TileKernels and the one-shot RoCE collectives of
[sparknet](https://github.com/christopherowen/dgx-spark-networking). Checkpoints load
directly (`--load-format direct`, `VLLM_PLUGINS` empty). The
[r6 promotion record](experiments/2026-10-05-tilelang-r6/README.md) has the initial promotion, and the
[decode-kernel experiment](experiments/2026-10-05-tilelang-decode-kernels/README.md)
compares the retired and promoted families kernel by kernel.
[docs/fundamentals.md](docs/fundamentals.md) explains the model itself: GEMM
shapes, number formats, one decoder layer, attention, mHC, Engram, DSpark, and
how TP3 and TP4 split the layers. The site, [switchle.ss](https://switchle.ss), is
[docs/index.html](docs/index.html): the page in [site/index.html](site/index.html)
with the manifests' numbers embedded by `scripts/render_docs.py`.

One recipe, [config/cluster.json](config/cluster.json), serves all twelve shapes
on one image (`spark-ds41f:1120`): single-node (`tp1`), two nodes direct or switched
(`tp2-direct`, `tp2-switched`), three nodes in a [switchless triangle](docs/switchless-topology.md)
(`tp3-triangle`, `tp3-triangle-4k`) or switched (`tp3-switched`), four in a
ring (`tp4-ring`, 1,048,576-token context with 32 sequences, [Performance](#performance)),
and four to eight through a switch (`tp4-switched` to `tp8-switched`,
[docs/switched-fabric.md](docs/switched-fabric.md)). The shapes differ only in
their cabling (node map, transport, NCCL settings) and their memory sizing.

## Quick start

You need one DGX Spark, two cabled to each other, three cabled as a triangle, four
cabled as a ring, or two to eight cabled into a switch, on DGX OS 26.09.2 or later.
Each node needs about 480 GiB of free disk, Docker
with the NVIDIA runtime, and passwordless SSH and `sudo` from the head node.
One or two Sparks read routed experts from disk on demand, so they serve a shorter
context and decode far slower than three or more
([docs/setup.md](docs/setup.md#requirements-and-tested-configurations)).
[docs/setup.md](docs/setup.md) is the full
guide: what each stage changes on your machines, how to check it, and how to
undo it.

| Sparks | Shape of [config/cluster.json](config/cluster.json) | Context | Kernel | [Memory saver](https://github.com/christopherowen/dgx-spark-memory-saver) |
| --- | --- | ---: | --- | --- |
| 1 | `tp1` | 262,144 | 64 KiB | required |
| 2 | `tp2-direct` | 262,144 | 64 KiB | required |
| 3 | `tp3-triangle-4k` | 262,144 | stock 4 KiB | not needed |
| 3 | `tp3-triangle` | 1,048,576 | 64 KiB | required |
| 4 | `tp4-ring` | 1,048,576 | 64 KiB | required |
| 2, switched | `tp2-switched` ([docs/switched-fabric.md](docs/switched-fabric.md)) | 262,144 | 64 KiB | required |
| 3 to 8, switched | `tp3-switched` to `tp8-switched` ([docs/switched-fabric.md](docs/switched-fabric.md)) | 1,048,576 | 64 KiB | required |

Name your cabling once with `bin/spark site shape NAME --apply`; every command
then resolves the recipe at that shape (`--shape NAME` overrides it). A site that
names none gets the recipe's default, `tp4-ring`. The 4 KiB
shape needs no kernel change, so it is the shortest path to a running service.
A shape starts only once a release benchmark has qualified the recipe's image
on it (`benchmark_reference`); `cluster start --unqualified-shape` overrides that.
Before writing your site's node maps, `./scripts/example_site.py` initializes
git-ignored example maps on a fresh clone for offline `bin/spark doctor` and test runs.

| Stage | Do | Check |
| --- | --- | --- |
| 1 | [Host baseline](docs/setup.md#1-host-baseline): `scripts/host-recovery apply`, headless boot, `nvidia-drm` mode setting, reboot | `scripts/host-recovery check` |
| 2 | [64 KiB kernel and memory saver](docs/setup.md#2-kernel-64-kib-profiles) (64 KiB profiles only) | `getconf PAGESIZE` prints 65536 |
| 3 | [Fabric](docs/setup.md#3-fabric-cabling-and-addressing): cable, then address with `sparknet topology discover` | 9000-byte pings on every cable path |
| 4 | [Site configuration](docs/setup.md#4-site-configuration): node map, head address, interfaces, `bin/spark site shape NAME --apply` | `bin/spark doctor` |
| 5 | [Model](docs/setup.md#5-model): the pinned checkpoint on every node; the [GLP steering file](docs/setup.md#glp-steering-file) is optional | `bin/spark doctor --live` warns where it is missing |
| 6 | [Image](docs/setup.md#6-image): `bin/spark build prepare`, `bin/spark --cluster-config config/cluster.json build image --apply`, copy to every node | one image ID everywhere |
| 7 | [First start](docs/setup.md#7-first-start): `bin/spark --cluster-config config/cluster.json cluster start --apply` | `bin/spark doctor --live` |
| 8 | [Verify](docs/setup.md#8-verify): `bin/spark bench` | quality 5/5, no regressions |

Read-only commands default to `config/cluster.json` at the site's shape; every command that takes
`--apply` refuses to act until the profile is named with `--cluster-config <profile>`
before the subcommand, so no node operation runs on a profile by accident. The
repository never changes a host by itself:
`bin/spark doctor` only reports, and prints each fix as a command for you to
run ([docs/doctor.md](docs/doctor.md)). `cluster start` refuses to launch
until the profile's kernel, fabric, cabling and image checks pass, and refuses
to change the running engine's shape (nodes, tensor parallelism, context, KV)
unless told to ([launch preflight](docs/doctor.md#launch-preflight)).

## Current baseline

The baseline is
[manifests/baselines/2026-11-20-spark-ds41f-1120.json](manifests/baselines/2026-11-20-spark-ds41f-1120.json)
([release record](experiments/2026-10-26-spark-ds41f-candidate/README.md#spark-ds41f-1120-candidate-62),
[release notes](experiments/2026-10-26-spark-ds41f-candidate/release-notes.md)), qualified
across all twelve topologies (`tp1`, `tp2-direct`, `tp2-switched`..`tp8-switched`, `tp3-triangle`,
`tp3-triangle-4k`, `tp4-ring`). See [docs/current-state.md](docs/current-state.md) and
[switchle.ss](https://switchle.ss) ([docs/index.html](docs/index.html), regenerated and
verified by `scripts/render_docs.py`).

<!-- BEGIN GENERATED: current-baseline-summary -->
| Shape | Ranks & transport | Context | Sequences | Batch budget | KV pool / rank | Min free RAM (head / workers) | Full contexts held (startup line) | Qualified reference |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `tp4-ring` **(default)** | TP4 `oneshot-ring4` (64 KiB) | 1,048,576 | 32 (up to 579,840 tok each) | 16,384 | 16.5 GiB | 16.09 / 18.43 GiB | 17 (17.86x; startup `18,711,942` / `17.85x`) | [`spark-ds41f-1120`](manifests/suites/2026-11-20-spark-ds41f-1120-tp4/) |
| `tp4-switched` | TP4 `oneshot-switched` (64 KiB) | 1,048,576 | 32 (up to 579,840 tok each) | 16,384 | 16.5 GiB | 14.34 / 16.32 GiB | 17 (17.86x; startup `18,711,942` / `17.85x`) | [`spark-ds41f-1120`](manifests/benchmarks/2026-11-20-spark-ds41f-1120-switched.json) |
| `tp5-switched` | TP5 `oneshot-switched` (64 KiB) | 1,048,576 | 32 (up to 831,744 tok each) | 32,768 | 24.0 GiB | 9.31 / 11.61 GiB | 25 (25.46x; startup `26,687,154` / `25.45x`) | [`spark-ds41f-1120`](manifests/benchmarks/2026-11-20-spark-ds41f-1120-switched5.json) |
| `tp6-switched` | TP6 `oneshot-switched` (64 KiB) | 1,048,576 | 32 (up to 432,384 tok each) | 24,576 | 13.0 GiB | 5.38 / 7.58 GiB | 13 (13.41x; startup `14,050,600` / `13.40x`) | [`spark-ds41f-1120`](manifests/benchmarks/2026-11-20-spark-ds41f-1120-switched6.json) |
| `tp7-switched` | TP7 `oneshot-switched` (64 KiB) | 1,048,576 | 32 (up to 530,176 tok each) | 32,768 | 16.0 GiB | 5.63 / 7.78 GiB | 16 (16.36x; startup `17,143,479` / `16.35x`) | [`spark-ds41f-1120`](manifests/benchmarks/2026-11-20-spark-ds41f-1120-switched7.json) |
| `tp8-switched` | TP8 `oneshot-switched` (64 KiB) | 1,048,576 | 32 (up to 888,320 tok each) | 32,768 | 25.5 GiB | 8.99 / 11.07 GiB | 27 (27.17x; startup `28,476,609` / `27.16x`) | [`spark-ds41f-1120`](manifests/benchmarks/2026-11-20-spark-ds41f-1120-switched8.json) |
| `tp3-triangle` | TP3 `oneshot-direct` (64 KiB) | 1,048,576 | 8 (up to 682,240 tok each) | 6,144 | 4.9062 GiB | 6.38 / 8.60 GiB | 5 (5.24x; startup `5,488,353` / `5.23x`) | [`spark-ds41f-1120`](manifests/suites/2026-11-20-spark-ds41f-1120-tp3/) |
| `tp3-switched` | TP3 `oneshot-switched` (64 KiB) | 1,048,576 | 8 (up to 682,240 tok each) | 6,144 | 4.9062 GiB | 6.34 / 7.92 GiB | 5 (5.24x; startup `5,488,353` / `5.23x`) | [`spark-ds41f-1120`](manifests/benchmarks/2026-11-20-spark-ds41f-1120-switched3.json) |
| `tp3-triangle-4k` | TP3 `oneshot-direct` (4 KiB) | 262,144 | 8 (up to 262,144 tok each) | 4,096 | 2.2 GiB | 11.28 / 13.17 GiB | 8 (8.80x; startup `2,304,107` / `8.79x`) | [`spark-ds41f-1120`](manifests/benchmarks/2026-11-20-spark-ds41f-1120-tp34k.json) |
| `tp2-direct` | TP2 `oneshot-direct` (64 KiB) | 262,144 | 4 (up to 262,144 tok each) | 4,096 | 8.0 GiB | 9.03 / 11.23 GiB | 34 (34.35x; startup `8,999,170` / `34.33x`) | [`spark-ds41f-1120`](manifests/benchmarks/2026-11-20-spark-ds41f-1120-direct2.json) |
| `tp2-switched` | TP2 `oneshot-switched` (64 KiB) | 262,144 | 4 (up to 262,144 tok each) | 4,096 | 8.0 GiB | 6.61 / 8.27 GiB | 34 (34.35x; startup `8,999,170` / `34.33x`) | [`spark-ds41f-1120`](manifests/benchmarks/2026-11-20-spark-ds41f-1120-switched2.json) |
| `tp1` | TP1 `local` (64 KiB) | 262,144 | 4 (up to 262,144 tok each) | 4,096 | 8.0 GiB | 18.76 / 18.76 GiB | 34 (34.35x; startup `8,999,170` / `34.33x`) | [`spark-ds41f-1120`](manifests/benchmarks/2026-11-20-spark-ds41f-1120-tp1.json) |
<!-- END GENERATED: current-baseline-summary -->

The promoted deployment (`tp4-ring` on four nodes by default; all twelve shapes share one
image, `spark-ds41f:1120`, differing only in cabling, transport, carve-out/expert-slot policy,
and memory sizing):

- twelve qualified shapes across `1..8` DGX Sparks: `tp4-ring` (`1M`, `32` seqs, `16.5 GiB` KV,
  `--cpuset-cpus=5-9,15-19`), `tp4..8-switched` (`1M`, `32` seqs, `13.0–25.5 GiB` KV),
  `tp3-triangle` and `tp3-switched` (`1M`, `8` seqs, `4.90625 GiB` KV), `tp3-triangle-4k` (`4 KiB`,
  `256K`, `8` seqs, `2.20 GiB` KV without Memory Saver), `tp2-direct` and `tp2-switched` (`256K`, `4`
  seqs, `8.00 GiB` KV), and single-node `tp1` (`local` `uni` executor, `256K`, `4` seqs, `8.00 GiB`
  KV), all (except `tp3-triangle-4k`) on DGX Spark 26.09.2 with kernel `7.0.0-1019-nvidia-64k`
  (`kho=off`), signed memory-saver DKMS 0.4.1 with UVM leaf-table packing enabled, no desktop,
  `vm.watermark_boost_factor=0`, and glibc malloc pinned (`MALLOC_ARENA_MAX=2`,
  `MALLOC_MMAP_THRESHOLD_=1048576`); vLLM's usage reporting off (`VLLM_NO_USAGE_STATS=1`,
  `DO_NOT_TRACK=1`, in the recipe and the image);
- dual ConnectX-7 200 GbE RoCE paths between every adjacent pair on switchless (`oneshot-ring4`,
  `oneshot-direct`) or through a switch (`oneshot-switched`), or local `uni` on `tp1`;
- Local Inference Lab's `integration/karmic-kraken-beta` vLLM (`19f2c20e`) with `candidate-62`'s
  vLLM series (`0001-0829`, `702` active patches, patch head `9b251a18`, tree `8fc99ca7`),
  CUDA `13.5.37` (hash-locked `.deb` layer over base `13.0.2`), and complete B12X retirement
  (`0802`):
  - TileLang `0.1.15+snapshot.35f15411cf3af0fc06a1cb0097215b1372a40464` (`14` patches, tree `741806c7`)
    and a device-compiler cache key (`0014`), preparing and freezing every declared kernel before memory profiling;
  - DS4.1 TileLang kernels end to end: sparse MLA front/indexer/quantize (`0777`, `0808`, `0813`, `0817`),
    64K globaltimer-paced SP prefill L2 prefetch (`0781`), sliced SP Engram row exchange (`0779`),
    Clock-Pro shielded direct native Engram reader (`0815`), grid-40 persistent vocab refine (`0819`),
    MoE down PDL preload (`0822`), UMA buddy sponge across post-load (`0800-0801`), zero-copy
    `io_uring` expert slot cache on `TP1`/`TP2` (`0822`), RM `nvidiactl` display carve-out on `TP2`/`TP3`
    (`0821`), and TileLang vision tower with multi-image packing and fast MM prefix cache (`0812`);
  - DeepSeek's TileKernels 2.0.0 (`cb2fa41c`, `1` patch) and sparknet 0.5.0 (`e06f6a06`, `3` patches:
    big-core RoCE proxy, pipelined multi-stage RoCE collectives, PDL collective chaining) with
    fail-stop `SPARKNET_ROCE_TIMEOUT_MS=10000`;
  - direct O_DIRECT checkpoint loading (`--load-format direct`, `VLLM_PLUGINS` empty) and rebuilt
    NCCL `2.30.7` (`6af10aa7`, `4` patches, 6-channel bidirectional ring on `tp3-triangle`);
- DeepSeek-V4.1-Flash-1017 (`1bae0e10`) native FP8/FP4/BF16 weights, with exact 12-bit packed BF16
  LM head (`VLLM_DS41_PACKED_BF16_LM_HEAD=1`, `VLLM_DS41_HEAD_DTYPE=float32`) and both
  `deepseek-v4.1-flash` and GLP-steered `deepseek-v4.1-flash-weightless`; opt-in FP8 group-32 LSA
  available via `VLLM_DS41_LSA=1` (`0828`, off by default);
- DSpark 5-token speculative decoding (`FULL_AND_PIECEWISE` CUDA graphs `1-192` on `TP4..TP8`, `1-48`
  on `TP3`, `PIECEWISE` `1-24` on `TP1`/`TP2`), margin confidence head + grid-40 Markov refinement
  (`0814`, `0819-0820`), calibrated piecewise curve on `TP3`, and CopyDrafter + late admission +
  prefill yield;
- hardened DSML token-provenance streaming/holdback (`0824`, `0829`), structured-output bitmask
  sizing (`0823`), SSE keep-alive callback binding (`0825`), OpenResponses / `CustomTool` (`0826`),
  and checkpoint `ignored_layers` normalization (`0827`).

One content-addressed image (`spark-ds41f:1120`, `sha256:5c12472565a7`) runs identically on all 12
nodes (`dgx1..dgx4`, `sp1..sp8`) and passes the release test suite (`12,023` passed, `356` skipped,
`0` failed across `169` files); see [Performance](#performance).

The machine-readable desired configuration is the recipe
[config/cluster.json](config/cluster.json): everything the shapes share once,
and a `shapes` table with each shape's node map, transport, NCCL settings,
memory sizing and qualifying benchmark
([scripts/recipe.py](scripts/recipe.py) resolves it; the
[memory profiles](docs/memory-profiles.md) explain the page-size variants). The
per-shape files it replaced are archived as release records in
[experiments/2026-10-14-one-recipe](experiments/2026-10-14-one-recipe/README.md).
To reproduce the deployment on your own Sparks, follow
[docs/setup.md](docs/setup.md).

## Performance

<!-- BEGIN GENERATED: performance-summary -->
Current qualified reference runs by shape (from `config/cluster.json` and `manifests/suites/`):

| Metric | `tp4-ring` ([`spark-ds41f-1120`](manifests/suites/2026-11-20-spark-ds41f-1120-tp4/)) | `tp3-triangle` ([`spark-ds41f-1120`](manifests/suites/2026-11-20-spark-ds41f-1120-tp3/)) |
| --- | ---: | ---: |
| Decode step geometric mean (20 TP4 / 15 TP3 shapes) | 19.36 ms | 25.52 ms |
| Decode, 16 streams (`c16` output tok/s) | 280.4 | — |
| Decode, 8 streams (`c8` output tok/s) | 213.6 | 170.7 |
| Decode, 4 streams (`c4` output tok/s) | — | 131.6 |
| One conversation at a time (`single-stream` `c1` output tok/s) | 89.1 | 70.4 |
| Sampled decode (`temp=1.0, top_p=1.0`), c16 / c8 / c1 (tok/s) | 276.6 / 210.0 / 87.1 | 167.2 / 167.6 / 68.3 |
| Poisson arrivals (0.7 TP4 / 0.42 TP3 req/s), output (tok/s) | 180.3 | 119.5 |
| Source-text prefill, 32K (tok/s) | 9,615 | 7,270 |
| Prefill, 131K (tok/s) | 10,108 | 7,502 |
| Prefill, 262K (tok/s) | 10,056 | 7,445 |
| Prefill, 524K (TP4) / 500K (TP3) (tok/s) | 9,833 | 7,259 |
| Prefill, 1M (tok/s) | 9,397 | 6,922 |
| Short-request TTFT p50 during 131K prefill, 3 / 11 (TP4) or 3 / 6 (TP3) streams | 0.73 s / 1.14 s | 0.80 s / 1.76 s |
<!-- END GENERATED: performance-summary -->

The comparisons earlier releases were promoted on are in
[docs/benchmarks.md](docs/benchmarks.md).

## Release history

Production is **spark-ds41f-1120**, serving DeepSeek-V4.1-Flash-1017 across all
twelve supported topologies ([release record](experiments/2026-10-26-spark-ds41f-candidate/README.md#spark-ds41f-1120-candidate-62),
[release notes](experiments/2026-10-26-spark-ds41f-candidate/release-notes.md)). It advances
`spark-ds41f-1107` onto CUDA `13.5.37` (hash-locked over `13.0.2`), retires B12X (`0802`), updates
vLLM through `0829` (`702` active patches), raises `TP4..TP8` concurrency from 16 to 32 streams
(`192`-row CUDA graphs, expanded KV pools up to `25.5 GiB`/rank on TP8), adds `tp1`, `tp2-direct`,
`tp2-switched`, and `tp3-triangle-4k` (with zero-copy `io_uring` expert slot cache on `TP1`/`TP2`
and RM `nvidiactl` carve-out on `TP2`/`TP3`), and passes `12,023` in-image release tests. Against
`spark-ds41f-1107` in 2-boot Sonar suites: `tp4-ring` decode grid `-7.73%` (`19.36 ms`), paired `c16`
`+3.07%` (`280.43 tok/s`), `c8 +2.70%` (`213.57 tok/s`), `single-stream c1 +7.30%` (`89.06 tok/s`),
prefill `+12.91%..+56.31%` (`9,615..9,397 tok/s`), stall `131K` prefill TTFT p50 `0.73 / 1.14 s`;
`tp3-triangle` `c8 +5.99%` (`170.68 tok/s`), `c4 +6.26%` (`131.59 tok/s`), `c1 +10.39%` (`70.36 tok/s`),
prefill `+12.17%..+55.69%` (`7,270..6,922 tok/s`); `tp4-switched` grid `-9.83%`, `tp8-switched` grid
`-9.92%` (`c16 395.40 tok/s`, prefill up to `12,429 tok/s`), and `tp3-switched` (`c8 168.92 tok/s`,
`c1 69.26 tok/s`), alongside 2-boot `bin/spark bench` matrices across all 12 shapes. The recipe
and lock are frozen in [releases/spark-ds41f-1120](releases/spark-ds41f-1120/); `spark-ds41f-1107`'s,
the rollback, in [releases/spark-ds41f-1107](releases/spark-ds41f-1107/).

Before spark-ds41f-1120, production was **spark-ds41f-1107**, serving DeepSeek-V4.1-Flash-1017 at 1M context
([release record](experiments/2026-10-26-spark-ds41f-candidate/README.md),
[release notes](experiments/2026-10-26-spark-ds41f-candidate/release-notes.md)). It is
spark-ds41f-1030 with vLLM 0568-0761 (641 active patches), the TileLang compiler refresh and
sparknet's single-slot one-shot all-reduce cap, and it is qualified for the first time on every
shape: `tp4-ring` and `tp3-triangle`, and `tp3-switched` to `tp8-switched` (tensor-parallel
sizes 3 to 8 with padded uneven shards, vLLM 0756-0759). Against spark-ds41f-1030 in one session
on the ring: paired decode +5.30% at sixteen streams (272.86 output tok/s), +4.04% at eight
(207.45) and +4.92% at one (82.62 tok/s); decode grid steps 14.36% shorter (20 of 20 shapes);
prefill 8.34-17.05% faster from 32K to 1M; a short request's first token beside a 131K prefill
in 1.02 s with 3 streams (3.30 s) and 1.44 s with 11 (3.59 s); greedy 256/256 identical. On the
triangle: paired decode +1.70% at eight streams (160.80 output tok/s), +2.04% at four (123.73)
and +2.59% at one (64.28 tok/s); decode grid steps 15.36% shorter (15 of 15 shapes); prefill
2.99-10.55% faster from 32K to 1M; greedy 256/256 identical. The recipe and lock are frozen in
[releases/spark-ds41f-1107](releases/spark-ds41f-1107/); spark-ds41f-1030's, the rollback, in
[releases/spark-ds41f-1030](releases/spark-ds41f-1030/).

Before spark-ds41f-1107, production was **spark-ds41f-1030**, serving DeepSeek-V4.1-Flash-1017 on the four-node TP4 1M
profile on the ring the site is cabled as ([release record](experiments/2026-10-26-spark-ds41f-candidate/README.md),
[release notes](experiments/2026-10-26-spark-ds41f-candidate/release-notes.md)). The release name is
the date it was promoted; the model is DeepSeek's checkpoint, unchanged. It is spark-ds41f-1025
with TileLang as the only kernel family and direct checkpoint loading, prefill yield (a short
request's first token beside a 131K prefill in 3.3-3.6 s instead of 12.4-12.8 s), a second served
name, `deepseek-v4.1-flash-weightless`, steered per request with msuiche's published GLP vector
(the stock name bit-identical beside it), the DeepSeek API's effort aliases and prompt semantics,
PDL along the decode chain and the attention front in one launch, 2-bit expert scale codes, the
startup KV line's corrected count (11.02 full 1M contexts, #586), and a fix for an engine stop
under mixed agent and chat traffic (#677). Temperature-0 text outputs are identical to
spark-ds41f-1025's except for multi-chunk prompts whose final chunk has 1-128 rows (CED now
replays a prompt's last 128 rows wherever its chunks end). Against spark-ds41f-1025 in one
session: decode 1.9-3.6% faster at every concurrency (78.3 tok/s at one stream, 198.4 output
tok/s at eight), decode grid steps 4.6% shorter, prefill 8.4-12.2% faster from 32K to 1M; greedy
256/256 identical.
On the triangle (qualified 2026-10-30, 6,144-token budget): decode 2.3-3.8% faster (63.5
tok/s at one stream, 158.6 output tok/s at eight), prefill 4.1-7.0% faster from 32K to 500K;
a 131K prefill beside 3 or 6 streams with short requests waiting is 3.7-4.6% slower, while
those requests' first token comes 2-8x sooner; greedy 256/256 identical. The recipe and lock are frozen in
[releases/spark-ds41f-1030](releases/spark-ds41f-1030/); spark-ds41f-1025's, its rollback, in
[releases/spark-ds41f-1025](releases/spark-ds41f-1025/).

Before spark-ds41f-1030, production was **spark-ds41f-1025**, serving DeepSeek-V4.1-Flash-1017 on the
four-node TP4 1M profile on the ring (since 2026-10-26; on the three-node TP3 1M profile on the
triangle from 2026-10-25 22:18 until 2026-10-26 08:10) ([release record](experiments/2026-10-25-spark-ds41f-candidate/README.md),
[release notes](experiments/2026-10-25-spark-ds41f-candidate/release-notes.md)). It is spark-ds41f-1022 with
every serving kernel in TileLang (except NCCL and a few rare-path ATen copies), the CED row
selection's nonfinal row skip with the ring's long-prefill budget at 16,384, fresh random draws in
each block-verification round (a backport of vllm-project/vllm#61161), fixes for a `prompt_logprobs`
request that killed the engine and for `logprob_token_ids` under speculative decoding, the
tokenizer's generation controls and the DeepSeek API's `thinking` switch (#447), every response's
prompt-cache hits in its usage (#469), glibc malloc pinned in every process (#439's memory growth),
GLP steering available but off unless configured (#401, msuiche's, reworked), the JIT monitor off,
and the live names moved to spark-ds41f (container, image, labels, paths, variables and caches).
Temperature-0 text outputs are identical to spark-ds41f-1022's; image answers differ (the TileLang
vision tower follows DeepSeek's reference arithmetic) and score the same on public benchmarks. At
most 32 top logprobs are served (the default is 20). Against spark-ds41f-1022 in one session: prefill is 11.8-13.9% faster from
32K to 1M (the 16,384 budget with the CED skip), decode grid steps are 2.4% shorter, and c16,
c8, Poisson and one-stream throughput are the same (75.3 tok/s at one stream, 194.9 at eight).
On the triangle, with its budget doubled to 8,192: prefill 25.5-28.2% faster from 32K to 500K
and 22.5% at 1M, decode grid steps 1.65% shorter, and the rest the same (60.5 tok/s at one
stream, 154.7 at eight).
spark-ds41f-1022's recipe and lock are frozen in `experiments/2026-10-23-next-release/1022`.

Before spark-ds41f-1025, production was **spark-ds41f-1022**, serving DeepSeek-V4.1-Flash-1017 on the four-node TP4
1M profile on the ring the site is cabled as (2026-10-23; the three-node TP3 1M profile on the
triangle since 2026-10-22,
[promotion record](experiments/2026-10-22-spark-ds41f-1022/README.md)), the first release named
`spark-ds41f-MMDD`. It is r6x with the TP group's broadcast on sparknet's one-shot all-gather
(serving never touches the PyTorch NCCL process group), sink detection and draft-zero rejection
on by default for DeepSeek V4.1 (short prompts such as `Say hi.` no longer collapse to an empty
answer), late admission (the next step is decided at decoder layer 30, so an arrival joins one
step sooner), control-token spellings in request text kept as plain text (#449), and sparknet
0.5.0's wider one-shot read phase. Outputs are identical to r6x's. Against r6x in one session on the triangle,
decode steps are 1.1% faster and eight streams, one stream and prefill level (prefill once
vLLM 0226 restored it); time to first token is level at four and eight streams, and a request
alone waits one decode step longer (sink rejection holds its first token one step). On the ring,
against r6x in one session, decode steps are 1.7% faster and sixteen streams, one stream and
prefill level, outputs identical. r6x's recipe and lock are frozen in
`experiments/2026-10-22-spark-ds41f-1022/r6x`.

Before spark-ds41f-1022, production was **r6x**, serving DeepSeek-V4.1-Flash-1017 on the four-node TP4 1M
profile on the ring (2026-10-19,
[promotion record](experiments/2026-10-18-r6x/README.md)), then on the triangle. r6x is r6v with the routed
experts' SwiGLU over the dispatch layout's live tiles, the routed experts' weight scales
kept as row bases and 4-bit codes (exact, about 2.1 GiB a rank freed), and a DSpark draft
budget that is a rule of the inputs (row cost 0.035), so nothing is timed at boot. Outputs
are identical to r6v's; against r6v decode steps are 2.5% faster and 16 and 8 streams 3.6%
and 3.8% faster, prefill level. The three-node TP3 profile served r6x at 1M context from
2026-10-19 (KV for five full 1M contexts at once, from the memory the scale codes freed;
[experiments/2026-10-19-r6x-tp3-1m](experiments/2026-10-19-r6x-tp3-1m/README.md)). r6v's recipe and lock are frozen
in `experiments/2026-10-18-r6x/r6v`.

Before r6x, production was **r6v** on both profiles: the
four-node TP4 1M profile on the ring (2026-10-17,
[promotion record](experiments/2026-10-17-r6v/README.md)), and the three-node TP3
512K profile on the triangle, whose path is r6u's (2026-10-17,
[promotion record](experiments/2026-10-17-r6u/README.md)). r6v is r6u with
Engram's sequence-parallel row exchange relayed through the neighbours on the
four-node ring, which has no cable between ranks two apart; against r6n, the
ring's previous release, prefill is 3.2-4.2% faster from 32K to 1M and decode is
level. r6u's recipe and lock are frozen in `experiments/2026-10-17-r6v/r6u` (r6s's in
`experiments/2026-10-17-r6u/r6s`, r6n's in `experiments/2026-10-16-r6s/r6n`). r6u is r6s with every kernel prepared before
any distributed work: each rank compiles its declared TileLang calls, TileKernels'
warm calls and the step's Triton kernels before memory profiling, freezes
TileLang's cache, and meets the other ranks on liveness, so a rank with a cold
cache can no longer stall a one-shot collective at startup. It follows DeepSeek's
current reference for both V4.1 Flash checkpoints (WO-A's input through
`act_quant`, prepared once per row in the attention's output store; the
2026-10-16 encoding fix), serves the 1017 checkpoint with DeepSeek's sampling
defaults (temperature 1.0, top_p 1.0), and passes the release tests
([docs/testing.md](docs/testing.md)) as part of every promotion. Prefill and
decode are level with r6s on 1017; outputs differ from r6s's by WO-A's input
quantization. r6s is r6n with faster Engram prefill: the rows of tokens the host already knows are read before the
step, Engram's sequence-parallel product is exchanged all-to-all instead of
all-gathered, and each rank's disk shard is whole hash columns, every third (or
fourth), so every rank reads every n-gram order and the ranks finish together.
Prefill is 4.8-5.6% faster, decode is level, and the temperature-0 text and the
model's distributions are identical to r6n's. r6n is r6k with Engram rows
read by a native worker in layer order and gathered in TileLang (one bounded
wait per table, this rank's rows only), collectives that stop on a timed-out
wait instead of falling back or computing on unwritten buffers, a host
rendezvous at startup, no compile on the decode path, and kernels that keep a
NaN rather than hide it. Its decode steps are 2.2% shorter, prefill and
throughput are level, and its temperature-0 text is identical to r6k's. r6k
before it removed the last known deviation from DeepSeek's reference
precision: FP32 logits, an exact FP32 mHC projection, the shared expert's
SwiGLU in FP32, MoE partials rounded once, WO on BF16 activations, and
temperature-0 outputs that do not depend on the batch.

In the candidate rounds before spark-ds41f-1107, the best validated corrected,
default-enabled base was
[candidate-40](experiments/2026-11-04-topk-live-qualification/README.md).
[C39](experiments/2026-11-04-copy-default-screening/README.md) delivered default
CopyDrafter behavior without a proven incremental serving-throughput gain;
C40 retains the independent top-k scratch-read correctness fix with neutral
aggregate serving performance. [C41's overlap simplification](experiments/2026-11-04-copy-overlap-rework/decision.md)
was rejected after its engine and cold-prefill costs failed the stated criterion.
The [PR qualification queue](experiments/2026-11-05-pr-screening/README.md) records
each source, purpose, outstanding gate and decision. [C42](experiments/2026-11-05-compiler-live-qualification/qualification.md)
records a built compiler image that failed required IKET prepare/freeze behavior,
so no serving or performance measurement ran. Its focused
[C45 correction](experiments/2026-11-05-compiler-prepared-rework/README.md)
continues qualification; the compiler idea has no performance verdict.
The metadata-only C44 proposal is withdrawn unmeasured. The built
[C43 frontend candidate](experiments/2026-11-05-frontend-lifetime-qualification/README.md)
failed a test-fixture invariant before serving; its distinct
[C46 correction](experiments/2026-11-06-worker-test-isolation-rework/README.md)
continues on C40. Preparation or build difficulty is not a performance rejection.
Each completed round reports compatible saved1030, C36, previous and current
builds, signed changes, both repeat margins and retained adverse rows.
Loop work follows the candidate's
[containment scope](experiments/2026-10-26-spark-ds41f-candidate/candidate-38/README.md#loop-containment):
preserve requested effort, end looping reasoning once, and stop continued
degeneration while protecting legitimate repeated answers.

## Names

The project was `spark3-vllm-ds41f` until 2026-10-05. The former GitHub name
now holds a pointer, and deployment checks accept configurations that name
it. The command is `bin/spark`, and experiment scripts call it; dated
experiment records keep `bin/spark3` as they ran it.

From the release after spark-ds41f-1022 (the owner's decision of 2026-10-19)
the live names are spark-ds41f's too:
- the serving container is `spark-ds41f` (its memory guard
  `spark-ds41f-memguard`), and images are tagged `spark-ds41f:<vLLM base>-<release>`;
- the image labels are `local.spark-ds41f.*` and its source trees live under
  `/opt/spark-ds41f/candidate`;
- the variables the vLLM patches read are `SPARK_DS41F_*`;
- the caches are under `cache/spark-ds41f` (`/cache/spark-ds41f` in the container);
- the hosts' boot memory policy is `spark-ds41f-kernel-memory.service`.

Earlier releases keep the names they were built and recorded with: images
tagged `vllm-ds41f-kkref:*` with `local.spark3.*` labels and `/opt/spark3`
trees read `SPARK3_*`, serve as `dsv41-karmic-kraken` and are booted from their
frozen recipes (spark-ds41f-1022's is `experiments/2026-10-23-next-release/1022`).
The tools read either label prefix and source root, and their guards recognise
either container. Each node keeps `cache/kkref` as a link to `cache/spark-ds41f`,
so those recipes keep their warm caches. Experiments, manifests and patches are
records and keep their names, as does Local Inference Lab's
`integration/karmic-kraken-beta` branch.

Lab windows also honour holds and requests written as `~/spark3-hold.json`
and `~/spark3-request.json`.

## Repository contract

There are three deliberately separate kinds of state:

1. `manifests/baselines/` contains immutable observations and benchmark identity.
2. `config/` contains the promoted desired state used to render launches.
3. `experiments/` contains isolated candidates and evidence before promotion.

An experiment never becomes the baseline merely because it is running. Promotion
requires reproducible measurements, an explicit decision, and a commit updating
the desired configuration and baseline record together.

## Commands

All commands are run from the repository root.

```sh
bin/spark doctor
bin/spark doctor --live
bin/spark status
bin/spark render dgx1
bin/spark --cluster-config config/cluster.json cluster sync
bin/spark --cluster-config config/cluster.json cluster start --replace
bin/spark --cluster-config config/cluster.json cluster stop
bin/spark cache status
bin/spark --cluster-config config/cluster.json cache clear
bin/spark bench
bin/spark upstream list
bin/spark upstream prepare vllm
bin/spark upstream prepare b12x
bin/spark build prepare
bin/spark build check
bin/spark build image
bin/spark --cluster-config config/cluster.json build image --apply
bin/spark build smoke
scripts/host-recovery check
scripts/host-recovery apply
```

`doctor`, `status`, `render`, `cache status`, and every cluster command and
`cache clear` without `--apply` are read-only. `bench` only sends API requests;
see [Benchmarking](#benchmarking).

`cache status` shows, for every node of the profile, each JIT cache directory
the profile's environment names under the `/cache` mount (`TILELANG_CACHE_DIR`,
`CUDA_CACHE_PATH`, `TRITON_CACHE_DIR`, `VLLM_CACHE_ROOT`): its size, file count
and oldest and newest file. `cache clear` empties those directories on every
node and keeps the directories themselves; `--only tilelang nv-compute triton
vllm` selects among them. It is a plan unless `--apply`, which needs
`--cluster-config` named and refuses while a serving container runs on any node
or a node cannot be reached. The start after a clear compiles the kernels
again, as a first start does.

`doctor --live` also compares the nodes' running kernel, base page size, driver
and memory policy, and verifies the staged 64 KiB kernel against
[`config/kernel-trial.json`](config/kernel-trial.json). It checks candidate
modules, headers, initramfs presence and the retained GRUB fallback; it never
selects a boot entry or applies corrections. Preparation is distinct from a
successful trial boot. The [boot trial](experiments/2026-10-02-kernel-64k/boot-test/README.md) passed compatibility checks. Matched memory profiling is in progress before deciding the default; both kernels and their matching swap files remain installed.

`doctor --live` checks the configured checkpoint's file fragmentation on every
node. It reports every shard whose current extent count exceeds ext4's best
count, even if `e4defrag` assigns it a severity score of zero; ideal layouts
remain quiet. Large files can have a best count greater than one. Run
`bin/spark doctor --fragmentation-commands` to print exact correction commands
for affected blob files, grouped by node. This only prints commands: any actual
defragmentation belongs in a coordinated maintenance window. The check reads
the checkpoint index and extent metadata using `sudo -n e4defrag -c`, resolves
shard symlinks, and neither scans tensor payloads nor flushes caches. Missing
permissions, unsupported filesystems or incomplete reports are warnings, never
reported as clean layouts. Fragmentation warnings do not block startup and do
not establish a loading slowdown. Plain `doctor` remains local.

`build prepare` writes only under ignored `.work/build/`; see
[docker/README.md](docker/README.md). `cluster sync` fetches a published commit and detaches every clean node
checkout at that exact revision; it never copies a working tree or ignored files other than the site's
node map. It then copies the image the profile names from the head to each node whose
ordered image layers or runtime configuration differ.
A commit counts as published when a branch on `origin` contains it. The
promoted configuration sets `deployment.branch: main`, so production deploys
only from `main`; experiment configurations omit the field, so they stay
deployable after their branch is merged and deleted.
The only cleanliness exception is a repository-local writable runtime mount
declared in `cluster.json` (currently `cache/`), which is preserved in place and
never enters Git.
`cluster start` preflights all three ranks, pre-arms a protected host-local
startup memory guard, waits for its first successful memory sample, and only
then starts workers before the head. Readiness fails if a guard exits or
available host memory crosses its threshold. Only a healthy API switches every
node to the lower steady-state guard. Mutating operations require an explicit
`--apply`; replacing an existing
service additionally requires `--replace`, and an experiment with
`deployment.launch_enabled=false` refuses mutation locally.

The promoted configuration is launch-enabled on `main`. Experiment
configurations set `launch_enabled` for themselves.

`scripts/` holds the memory guard that `cluster start` installs on each node
and host recovery, which has not yet moved into `bin/spark`. Experiments keep
their own scripts in their directories.

Host management-plane recovery is versioned separately under `host/recovery/`.
It arms the existing hardware watchdog and protects SSH/Tailscale without
restarting Docker or the inference service. The incident evidence and exact
policy boundary are documented in [docs/recovery.md](docs/recovery.md).

## Benchmarking

Every measurement runs through [Sonar](https://github.com/christopherowen/Sonar), the
benchmark client this checkout pins in [config/sonar.json](config/sonar.json): steady-state
load, the engine grid, model behaviour, prefill and stalls, suites and verdict tables, the
determinism gate, the quick probes and the measurement matrix below. The pinned commit is
checked out on first use under `.work/sonar/<revision>` and verified there
([spark_ds41f/sonar.py](spark_ds41f/sonar.py)); `bin/spark sonar ARGS...` runs it directly.
What stays here is the deployment's own: the suites sized to its shapes
([config/suites/](config/suites/)), the A/A floors its testbeds recorded
([scripts/sonar_table.py](scripts/sonar_table.py)), the chat template's reasoning switch, and
the conditions every bench runs under (cooling, fan curve, memory and temperature watch, the
deployment's identity in the run). The lab's extras are thin shims over Sonar commands
(`scripts/sonar_*.py`, and `scripts/distinct_streams.py` and `scripts/acceptance_sweep.py` under
their old names); each sends the requests the script it replaced sent.

A release is benchmarked against the previous one with Sonar suites in a lab session:
[docs/release-benchmark.md](docs/release-benchmark.md) has the commands. `bin/spark bench`
measures the live service from the head node with Sonar's measurement matrix and writes one
self-contained report to `results/private/bench/<UTC time>/bench.json`
(`--output` to change). By default it is a quick check of about six minutes:
the quality gate and every decode point with three or four samples each.
`bin/spark bench --full` runs every suite below and samples each decode
point to its precision target, about 35 minutes; use it for experiments that
need to resolve small differences. Before sending anything it runs the `doctor --live`
comparison and refuses a cluster that differs from its configuration. It also
records the configuration hash, each node's image ID and checkout, the
client commit and the Sonar revision.

Suites of the full run, in order (`--suites` selects a subset):

- `quality`: the fixed LRU request five times at temperature 0; all five must
  pass, or the run stops before measuring anything.
- `compliance` (explicit `--suites compliance`; not part of `--full` until
  constrained decoding beside DSpark is qualified): structured-output
  correctness on natural completion. Eight record prompts, each asking for a
  complete array of twelve objects with 4,096 tokens of room, run prompt-only
  and then constrained by the same array schema through `response_format`. A
  request passes when it stopped on its own (`finish_reason`), parsed as
  exactly twelve objects with no fence or commentary, and every object
  conforms to its schema (python-jsonschema). Outputs are saved under
  `outputs/compliance-<arm>/`. Every request receipt in every suite now
  records `finish_reason`.
- `decode`: prose and code prompts at concurrency 1, 2, 4, and 8, 256 output
  tokens, reasoning on, temperature 0, and the same prompts and metric
  (aggregate completion tokens per wall second) as every published baseline.
  Output text still differs from run to run, so each point is a random
  draw. Points are sampled in shuffled rounds after a discarded warmup round.
  With `--full`, each point keeps sampling until its 95% confidence interval
  is within `--precision` (2%) of the mean, between `--min-samples` (6) and
  `--max-samples` (40) samples; the quick default takes 3-4.
  Before r6x, single boots of one configuration differed by about 3%, because
  adaptive verification profiled its costs at startup; since r6x nothing is timed
  at boot and the same build repeats within about 0.4%.
  `--decode-cases` selects other cases. `prose-nothink`, `code-nothink` and
  `json-nothink` turn reasoning off to measure the answer itself. The
  `portable` group (`count`, `explain`, `tasks`, `rows`, `math`, `chat`,
  `essay`, `story`, `chat-sampled`) follows the protocol public DGX Spark
  benchmarks use, so figures line up with theirs:
  - reasoning off, and exactly 256 tokens (`min_tokens` with `ignore_eos`);
  - a distinct prompt per stream, and distinct code tasks for `tasks`;
  - workloads from a number sequence through code and JSON to free prose;
  - `chat-sampled` at temperature 0.7.

  Every decode point also reports `decode_window_tps`: the tokens after each
  stream's first, over the span from the earliest first token to the latest
  last token. It leaves out prefill and the start stagger of a concurrent
  wave.

  Three primary numbers accompany every point: `step_ms` (time per engine
  step; exact for one stream, a decode-window estimate for several),
  `accepted_per_verified` (accepted tokens per verified draft token) and
  `tokens_per_step` (one plus accepted drafts per step). Throughput is
  reported twice: measured (`tps` and `decode_window_tps`) and `derived_tps`,
  what the primaries imply (`tokens_per_step` times streams over `step_ms`).
  Temperature-0 output text does not repeat between samples, and draft
  acceptance follows the text, so `accepted_per_verified`, `tokens_per_step`
  and tok/s all swing between samples of the same prompt; `step_ms` stays
  within about 1%. Gate kernel, transport and host changes on `step_ms`, and
  use the acceptance numbers to explain why tok/s moved.

  The `precision` group (`prose-1k`, `code-1k`, `json-1k`) resolves small
  differences between arms: 1,024 output tokens, prompts whose natural answer
  exceeds that budget so nothing is forced to continue past its end, at least
  five and at most twelve samples per point, and convergence on `step_ms`
  rather than `tps`. `--converge-on` overrides the convergence metric for
  every point. A `WARN` line names outputs that stopped before the budget;
  lengthen the prompt when it appears. `json-1k` gives each stream a distinct
  record type whose prompt is generated from a JSON schema, and scores every
  output against that schema: completed objects, whether the truncated prefix
  parses once closed, and the share of objects that conform (`json_schema_valid`,
  with the first violation quoted). Each output is also written under
  `outputs/<point>/` in the report directory, as the raw text and, when it
  parses, the closed prefix as `.json`, so `jq` can be run on it afterwards.
  Validation uses python-jsonschema (Draft 2020-12): `sudo apt install
  python3-jsonschema` on the dgx hosts, or the hashed
  `requirements/bench-client.txt` elsewhere. `json-schema-1k` requests the
  same through the server's structured output (`response_format` with a JSON
  schema). It is opt-in and
  outside the group, because constrained decoding changes the sampler path and
  needs its own qualification beside speculative decoding before its numbers
  mean anything.
- `sampled`: DSpark accepted drafts per step at temperature 1.0 from the
  engine counters, 32 requests per case at concurrency 1 and 4.
- `prefill`: cold prefill at 2K, 32K, 64K, and 128K tokens, three unique
  uncached prompts each. `--prefill-text` chooses the text:
  - `filler` (the default) repeats 18 words, so its Engram rows stay cached;
  - `novel` uses random pseudo-words, whose rows are read from disk;
  - `source` uses real text, the Python standard library's source and
    docstrings.
- `prefix`: a 32K prompt followed by two identical replays, reporting cold and
  warm TTFT and the cache hit rate.
- `admission`: four concurrent 64K-token contexts; all four must run at once
  without preemption.

Measurements stay clean and safe:

- A sample starts only when the engine reports no running or waiting requests.
- A sample that overlaps anyone else's request is repeated. Overlap shows up
  in the engine's request counters and peak running count.
- A per-node memory monitor stops the run, cancelling in-flight requests, if
  MemAvailable falls within 1 GiB of the steady memory guard.

The report is still written if the run stops early.

With a reference run, each decode and prefill point shows its percent change
with a Welch 95% interval. The default reference is the profile's
`benchmark_reference` when it names one, as the recipe's `tp4-ring` shape
does for its four-node run,
and otherwise `manifests/benchmarks/<promoted_baseline>.json`. A run is only
comparable with a reference of the same node count, transport and kernel
backend. The bench refuses a default reference that differs in any of them;
`--compare PATH` names a reference explicitly and only warns about a
difference, and `--compare none` skips the comparison. The command exits
non-zero on a failed quality or admission check, any failed request, an early
stop, or a point that is significantly slower by more than `--tolerance` (3%).
A promotion adds a reference run to `manifests/benchmarks/` for each topology
it promotes: `<baseline>.json`, plus a suffixed run such as
`<baseline>-tp4.json` that the other topology's profile names.

## Upstreams

Canonical upstreams, contribution forks, tracking branches, pinned commits, and
applied upstream fixes live in [upstreams.lock.json](upstreams.lock.json). Local
changes are kept as ordered patch series rather than edits to copied source
trees. `upstream` always means the canonical project; `origin` is Christopher's
fork when one exists; this deployment repository is neither. See
[docs/upstreams.md](docs/upstreams.md). Repositories that are useful for ideas
but are not build inputs are kept separately in
[docs/inspiration.md](docs/inspiration.md).

## Transition status

`bin/spark build` builds the promoted image from the pinned sources in
`upstreams.lock.json` and the patch series ([docker/README.md](docker/README.md)),
and one content digest runs on all three nodes.

Canonical vLLM main is parked; the serving sources are Local Inference Lab's
`integration/karmic-kraken-beta` vLLM and B12X, TileLang, DeepSeek's
TileKernels and sparknet, with the local patch series
([docs/upstreams.md](docs/upstreams.md)).

## License

spark-ds41f is licensed under the [Apache License 2.0](LICENSE).

Files copied from upstream projects keep their own copyright and license
headers. These are the vLLM and B12X sources in experiment overlays and bundles,
and in `docs/serving-audit-2026-10-01/sources`. Each patch series changes its
upstream's code under that project's license. The upstreams are listed in
[docs/upstreams.md](docs/upstreams.md).
