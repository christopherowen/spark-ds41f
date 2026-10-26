# spark-ds41f

Reproducible Docker/vLLM deployment, tuning, and benchmarks for DeepSeek V4.1
Flash on a switchless three-node or four-node DGX Spark fabric.

Production is **spark-ds41f-1025**, serving DeepSeek-V4.1-Flash-1017 on the three-node TP3 1M
profile on the triangle the site is cabled as, and qualified on the four-node TP4 1M profile on
the ring as well ([release record](experiments/2026-10-25-spark-ds41f-candidate/README.md),
[release notes](experiments/2026-10-25-spark-ds41f-candidate/release-notes.md)). The release name is
the date it was promoted; the model is DeepSeek's checkpoint, unchanged. It is spark-ds41f-1022 with
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

The promoted kernel family is **TileLang** (r6): TileLang kernels for SM121,
with DeepSeek's TileKernels and the one-shot RoCE collectives of
[sparknet](https://github.com/christopherowen/dgx-spark-networking), in place
of B12X's attention, linear, MoE, mHC, vocabulary and collective kernels
(`kernel_backend: tilelang`, see `scripts/kernel_backend.py`). B12X stays in
the image as the alternative backend; a configuration without
`kernel_backend` means B12X. The [r6 promotion record](experiments/2026-10-05-tilelang-r6/README.md)
has both recipes and their benchmarks, and the
[decode-kernel experiment](experiments/2026-10-05-tilelang-decode-kernels/README.md)
compares the two families kernel by kernel.
[docs/fundamentals.md](docs/fundamentals.md) explains the model itself: GEMM
shapes, number formats, one decoder layer, attention, mHC, Engram, DSpark, and
how TP3 and TP4 split the layers.

The deployment tools also generate and validate a
[four-node switchless ring](docs/switchless-topology.md). One recipe,
[config/cluster.json](config/cluster.json), serves every shape on one image:
four nodes in a ring (`tp4-ring`, the checkpoint's full 1,048,576-token
context with 16 sequences, benchmarked at those limits, see
[Performance](#performance)) and three in a triangle (`tp3-triangle`, the same
1,048,576-token context with 8 sequences since r6x). The shapes differ only in their
cabling (node map, transport, NCCL settings) and their memory sizing.

This repository is being promoted from a forensic capture of the running cluster
into its only operational source of truth. Until the transition checklist is
complete, files are explicit about whether they describe observed state, desired
state, or an experiment.

## Quick start

You need three DGX Sparks cabled as a triangle, or four cabled as a ring, on
DGX OS 26.09.2 or later. Each node needs about 480 GiB of free disk, Docker
with the NVIDIA runtime, and passwordless SSH and `sudo` from the head node.
Two Sparks are not supported. [docs/setup.md](docs/setup.md) is the full
guide: what each stage changes on your machines, how to check it, and how to
undo it.

| Sparks | Shape of [config/cluster.json](config/cluster.json) | Context | Kernel | [Memory saver](https://github.com/christopherowen/dgx-spark-memory-saver) |
| --- | --- | ---: | --- | --- |
| 3 | `tp3-triangle-4k` | 262,144 | stock 4 KiB | not needed |
| 3 | `tp3-triangle` | 1,048,576 | 64 KiB | required |
| 4 | `tp4-ring` | 1,048,576 | 64 KiB | required |

Name your cabling once with `bin/spark site shape NAME --apply`; every command
then resolves the recipe at that shape (`--shape NAME` overrides it). A site that
names none gets the recipe's default, `tp4-ring`. The 4 KiB
shape needs no kernel change, so it is the shortest path to a running service.
A shape starts only once a release benchmark has qualified the recipe's image
on it (`benchmark_reference`); `cluster start --unqualified-shape` overrides that.

| Stage | Do | Check |
| --- | --- | --- |
| 1 | [Host baseline](docs/setup.md#1-host-baseline): `scripts/host-recovery apply`, headless boot, `nvidia-drm` mode setting, reboot | `scripts/host-recovery check` |
| 2 | [64 KiB kernel and memory saver](docs/setup.md#2-kernel-64-kib-profiles) (64 KiB profiles only) | `getconf PAGESIZE` prints 65536 |
| 3 | [Fabric](docs/setup.md#3-fabric-cabling-and-addressing): cable, then address with `sparknet topology discover` | 9000-byte pings on every cable path |
| 4 | [Site configuration](docs/setup.md#4-site-configuration): node map, head address, interfaces, `bin/spark site shape NAME --apply` | `bin/spark doctor` |
| 5 | [Model](docs/setup.md#5-model): the pinned checkpoint on every node | |
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
[manifests/baselines/2026-10-25-spark-ds41f-1025.json](manifests/baselines/2026-10-25-spark-ds41f-1025.json)
([release record](experiments/2026-10-25-spark-ds41f-candidate/README.md)), on the ring at 1M context;
the triangle's last release is spark-ds41f-1022. See
[docs/current-state.md](docs/current-state.md). The three-node deployment (the four-node ring
differs only in cabling, transport and memory sizing):

- three DGX Spark nodes using tensor parallelism 3, on DGX Spark 26.09.2 with
  kernel `7.0.0-1019-nvidia-64k` (`kho=off`), signed memory-saver DKMS
  0.4.1 with UVM leaf-table packing enabled, no desktop, and
  `vm.watermark_boost_factor=0`;
- direct dual ConnectX-7 paths between every pair of nodes;
- Local Inference Lab's `integration/karmic-kraken-beta` vLLM (plus Engram
  projection sharding, asynchronous Engram rows, and two tool-call and
  image-cache fixes) with the TileLang kernel family:
  - TileLang 0.1.15 with three patches that enable SM120/SM121 block-scaled
    MMAs (MXFP8, FP8 × FP4, MXFP4);
  - DS4.1 TileLang kernels for sparse MLA, the MXFP4 indexer, the block-32 FP8
    and BF16 projections, the FP8 × FP4 routed experts and the vocabulary
    heads; decode rows run on 16-, 32- or 64-row tiles with TileLang's
    unspecialized pipeline (its warp-specialized scale staging races on SM121
    at decode tiles), and prefill rasterizes weights larger than L2 in panels;
  - DeepSeek's TileKernels 2.0.0 for the MoE top-k gate, the SwiGLU and
    quantization casts and mHC, with the mHC projection a TileLang kernel;
  - the routed experts' SwiGLU and MXFP8 cast run over the dispatch layout's
    live tiles only, and the routed experts' weight scales are kept as a row
    base exponent and 4-bit codes (exact, half the scale bytes; about 2.7 GiB a
    rank freed at TP3, which pays for the 1M context);
  - TileLang kernels for RoPE, the CSA compressor and its projection, the KV
    cache writers, the Engram gate, hash and row gather, the index-head weight
    scale, the sparse-MLA metadata (the first port off Triton), the WO
    projection and the drafter's context-KV projection;
  - sparknet 0.5.0's one-shot RoCE all-reduce and all-gather, explicit and
    fail-stop, for every TP collective including DSpark's per-step broadcast
    (serving never uses the PyTorch NCCL process group); its data phase reads
    one 16-byte pack a thread with a single poller;
  - L2 weight prefetch during decode (the next layer's weights and scale
    words stream into L2 while latency-bound kernels run);
  - B12X's vision tower kernels, which the family does not replace yet, and
    its checkpoint loader;
- B12X, with all its fixes, as the alternative backend in the same image: its
  FP4 KV writer rounds like DeepSeek's reference quantizer, its indexer top-k
  breaks score ties by position, its dense GEMM and prefill kernels fence
  shared-memory stage reads before the TMA refill, and W4A8 tiny decode stays
  off; its four-node RoCEnante relay and mesh transports are present too;
- NCCL 2.30.7 rebuilt with the AArch64 InfiniBand send-path fence
  (NVIDIA/nccl#2393), which prevents a proxy-thread hang, and the four-node
  switchless ring patches (bidirectional rings, balanced channels);
- DeepSeek V4.1 Flash native FP8/FP4/BF16 weights, every served weight bit as
  the checkpoint stores it, with the vision tower loaded (up to four images
  per request, history included: the recipe's chosen limit, not the
  model's; see [images per request](docs/current-state.md#images-per-request));
  the output head is stored in an exact 12-bit packed form of
  its BF16 weights, and the embedding and output head live in the GB10
  display carve-out, loaded there directly from the checkpoint while the
  text console keeps its framebuffer;
- arithmetic at DeepSeek's reference precision (`inference/model.py`): FP32
  logits from the packed head, the mHC stream projection in FP32 (each FP32
  weight split into three exact TF32 parts), the shared expert's SwiGLU in
  FP32 with one rounding, and MoE partials rounded once per rank with the
  shared expert added in the combine, and WO-A on BF16 activations with its
  input quantized as the reference's 2026-10-12 revision does (`act_quant`,
  block 32, UE8M0);
- every row's result is independent of its batch: each kernel adds its
  reduction in a fixed order, the collectives add every rank's partial in rank
  order with one FP32 rounding, prefill chunks leave room for every stream's
  decode and draft rows, and a step that mixes decode and prefill runs sparse
  MLA per request, so decode rows keep the decode plan;
- the ratio-2 compressor carries its open pair across decode steps, so
  compressed entries for generated tokens match the ones prefill builds;
- DSpark speculative decoding with five draft tokens (the drafter's trained
  block) and block rejection, full CUDA graphs for decode batches up to 48
  tokens, and a draft budget that is a rule of the inputs (a fixed linear step
  cost, row cost 0.035), so nothing is timed at boot; verification rows
  whose drafts are unlikely to survive skip the routed experts, the drafter
  marks its padding rows for the MoE routers, and greedy drafts stay sharded
  by vocabulary over the checkpoint's BF16 drafter head and Markov projection;
- late admission: the asynchronous engine decides each step when the step in
  flight reaches decoder layer 30, so a request that arrives meanwhile joins
  the next step; a step that continues a prompt chunk decides its successor at
  once, which keeps chunked prefill's pace;
- sink detection and draft-zero rejection for DeepSeek V4.1, on by default
  (`VLLM_SINK_REJECTION=0` turns it off): a sampled token that becomes the
  model's attention sink (residual column 2455) is drawn again, so short
  prompts such as `Say hi.` no longer collapse; a request's first token waits
  one decode step for it. `vllm:sink_*` counts the flags in the outputs the
  scheduler commits; each rank's `Sink detector, rank N` log line counts every
  flagged row its GPU executed, including steps whose output is discarded (the
  step in flight when a request finishes or is aborted, stale output after a
  preemption), so the log reads at least the metric (#489);
- control-token spellings written in request text stay plain text, as
  DeepSeek's encoder treats them;
- the fabricated context of startup profiling gives each KV cache group its
  own blocks;
- sequence-parallel prefill once a prompt chunk's reduce-scatter outgrows
  the one-shot RoCE all-reduce (205 tokens);
- mutating custom ops read their argument schema once per call, not once per
  argument as torch does;
- 1,048,576-token per-request limit, eight admitted sequences, and a 4.90625
  GiB-per-rank cache that holds five full 1M contexts at once, or all eight
  sequences at up to 666,880 tokens (spark-ds41f-1025's startup line: 4,009,281
  tokens, 3.82x; [KV capacity](docs/memory-profiles.md#kv-capacity));
- one concurrent prefill, 8,192 batched tokens, and fail-closed 5 GiB startup
  and 3 GiB steady memory guards;
- FlashInfer autotune disabled (`--no-enable-flashinfer-autotune`): only the
  sampler uses FlashInfer, and the pass saved no configs.

One content-addressed image runs on all three nodes and passes the LRU
coherence gate 5/5; see [Performance](#performance).

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

spark-ds41f-1022 against r6x on four nodes, both serving DeepSeek-V4.1-Flash-1017
(2026-10-23), measured with the trimmed promotion suite in one session (spark-ds41f-1022,
r6x, spark-ds41f-1022 again; [results](experiments/2026-10-22-spark-ds41f-1022/README.md#the-ring-2026-10-23);
[suite run](manifests/suites/2026-10-22-spark-ds41f-1022-tp4/)). Bold changes exceed their 95%
interval and the run's own repeat.

| Workload | TP4 r6x | TP4 spark-ds41f-1022 |
| --- | ---: | ---: |
| Decode step time, geometric mean of 20 shapes | | **-1.7%** ±0.16 |
| Decode, 16 streams (tok/s) | 253.1 | 256.5 (+1.3% ±0.8) |
| One conversation at a time (tok/s) | 74.1 | 74.1 (0.0% ±8.3) |
| Source-text prefill, 32K (tok/s) | 6,034 | 6,030 (-0.1% ±0.5) |
| Prefill, 131K | 5,982 | 5,985 (+0.1% ±0.6) |
| Time to first token p50, 16 streams / one stream (s) | 0.396 / 0.100 | 0.400 / 0.129 |

All 64 greedy responses are identical to r6x's. A request alone waits one decode step
longer for its first token: sink rejection verifies it before it is sent.

spark-ds41f-1022 on three nodes at 1M (2026-10-22): the same comparison on its image
`-1022-2` (one session; [results](experiments/2026-10-22-spark-ds41f-1022/README.md#promotion-comparison-2026-10-22-1003-1105-cet))
measured the decode step **1.1%** faster (±0.16), eight streams 151.3 to 153.6 tok/s (same), one
stream level and prefill 0.7-0.8% lower; vLLM 0226 then restored prefill (its A/B: 0.6-1.4%
faster on every paired prompt), and the promoted image measured 4,460 / 4,384 tok/s at
32K / 131K against r6x's 4,461 / 4,386. Time to first token at four and eight streams is level
with r6x's, and one stream is one decode step longer. Reference run:
[manifests/suites/2026-10-22-spark-ds41f-1022-tp3](manifests/suites/2026-10-22-spark-ds41f-1022-tp3/).

r6x against r6v on four nodes, both serving DeepSeek-V4.1-Flash-1017 (2026-10-18),
measured with the `release` suite in one session (r6x, r6v, r6x again;
[results](experiments/2026-10-18-r6x/README.md);
[suite run](manifests/suites/2026-10-19-karmic-kraken-r6x-tp4/)).

| Workload | TP4 r6v | TP4 r6x |
| --- | ---: | ---: |
| Source-text prefill, 32K (tok/s) | 5,987 | 5,989 (0.0% ±1.1) |
| Prefill, 128K | 5,945 | 5,947 (0.0% ±0.7) |
| Prefill, 256K | 5,681 | 5,680 (0.0% ±0.2) |
| Prefill, 512K | 5,172 | 5,173 (0.0% ±0.3) |
| Prefill, 1M | 4,406 | 4,407 (0.0% ±0.6) |
| Decode step time, geometric mean of 20 shapes | | **-2.5%** ±0.15 |
| Decode, 16 streams (tok/s) | 241.7 | 250.3 (**+3.6%** ±0.9) |
| Decode, 8 streams | 184.4 | 191.5 (**+3.8%** ±1.1) |
| Poisson arrivals at 0.7 req/s, output (tok/s) | 180.4 | 180.4 (0.0% ±17) |
| One conversation at a time (tok/s, `single-stream`) | 71.0 | 74.1 (+4.5% ±4.5; TTFT p50 0.10 s) |

All 256 greedy responses are identical to r6v's; acceptance length is level (2.27).

r6x at 1M context against r6v at 512K on three nodes, both serving
DeepSeek-V4.1-Flash-1017 (2026-10-19), measured with the `release-tp3` suite in one
session (r6x, r6v, r6x again;
[results](experiments/2026-10-19-r6x-tp3-1m/README.md);
[suite run](manifests/suites/2026-10-19-karmic-kraken-r6x-tp3/)). The suite's prefill
stops at 500K; one cold 1,000,000-token prefill was measured on its own.

| Workload | TP3 r6v (512K) | TP3 r6x (1M) |
| --- | ---: | ---: |
| Source-text prefill, 32K (tok/s) | 4,425 | 4,465 (+0.9% ±0.9) |
| Prefill, 128K | 4,357 | 4,390 (**+0.8%** ±0.7) |
| Prefill, 256K | 4,167 | 4,196 (**+0.7%** ±0.3) |
| Prefill, 500K | 3,810 | 3,835 (**+0.7%** ±0.3) |
| Prefill, 1M (one cold run) | | 3,227 (310 s) |
| Decode step time, geometric mean of 14 shapes | | **-1.3%** ±0.14 |
| Decode, 8 streams (tok/s) | 147.1 | 152.7 (**+3.8%** ±1.5) |
| Decode, 4 streams | 111.5 | 116.0 (**+4.0%** ±1.9) |
| Poisson arrivals at 0.42 req/s, output (tok/s) | 117.6 | 118.3 (+0.7% ±22) |
| One conversation at a time (tok/s, `single-stream`) | 57.0 | 59.7 (+4.7% ±6.3) |
| KV pool, startup line (4,096 batched tokens) | 2,845,543 tokens (5.43x 512K; holds 7 at once) | 4,765,898 tokens (4.55x 1M; holds 5 at once) |

All 256 greedy responses are identical to r6v's. During the cold 1M prefill dgx1's
lowest MemAvailable was 5.93 GiB, above its 3 GiB steady guard.

r6v against r6n on four nodes, both serving DeepSeek-V4.1-Flash-1017 (2026-10-17),
measured with the `release` suite in one session (r6v, r6n on its production ring
curves; [results](experiments/2026-10-17-r6v/README.md);
[suite run](manifests/suites/2026-10-17-karmic-kraken-r6v-tp4/)).

| Workload | TP4 r6n | TP4 r6v |
| --- | ---: | ---: |
| Source-text prefill, 32K (tok/s) | 5,762 | 5,982 (**+3.8%** ±0.6) |
| Prefill, 128K | 5,706 | 5,946 (**+4.2%** ±1.1) |
| Prefill, 256K | 5,467 | 5,676 (**+3.8%** ±0.6) |
| Prefill, 512K | 4,988 | 5,171 (**+3.7%** ±0.2) |
| Prefill, 1M | 4,269 | 4,405 (**+3.2%** ±0.5) |
| Decode step time, geometric mean of 20 shapes | | -0.05% ±0.14 |
| Decode, 16 streams (tok/s) | 243.5 | 243.8 (+0.1% ±1.0) |
| Decode, 8 streams | 185.4 | 185.4 (0.0% ±1.1) |
| Poisson arrivals at 0.7 req/s, output (tok/s) | 180.2 | 180.4 (+0.1% ±17) |
| One conversation at a time (tok/s, `single-stream`) | | 71.1 (TTFT p50 0.10 s) |

No greedy response is identical to r6n's (WO-A's input quantization, DeepSeek's current
reference); acceptance length is level (2.27).

r6u against r6s on three nodes, both serving DeepSeek-V4.1-Flash-1017 (2026-10-17),
measured with the `release-tp3` suite ([results](experiments/2026-10-17-r6u/README.md);
[suite run](manifests/suites/2026-10-17-karmic-kraken-r6u-tp3/)). r6u is measured with
vLLM 0179 mounted over its image through 0178 (the promoted image carries the same
files), in the session after r6s's, so the decode step time is cross-session (the same
build read 1.5% apart in two sessions that day).

| Workload | TP3 r6s | TP3 r6u |
| --- | ---: | ---: |
| Source-text prefill, 32K (tok/s) | 4,414 | 4,405 (-0.2% ±1.2) |
| Prefill, 128K | 4,346 | 4,343 (-0.1% ±0.6) |
| Prefill, 256K | 4,153 | 4,154 (0.0% ±0.5) |
| Prefill, 500K | 3,801 | 3,801 (0.0% ±0.2) |
| Decode step time, geometric mean of 15 shapes | | -1.0% ±0.13 |
| Decode, 8 different prompts at once (tok/s) | 146.6 | 145.6 (-0.7% ±1.3) |
| Decode, 4 different prompts at once | 112.5 | 111.3 (-1.1% ±1.4) |
| Poisson arrivals at 0.42 req/s, output (tok/s) | 117.7 | 117.3 (-0.3% ±21) |

r6u's image through 0178 quantized WO-A's input in the GEMM's column tiles and was
2.8-3.2% slower at prefill (the same afternoon, r6u / r6s / r6u again); 0179 prepares
it once per row and changes no output bit. No greedy response is identical to r6s's
(WO-A's input quantization); r6s's teacher-forced KL on r6u's text is 0.023 nats.

r6s against r6n on three nodes (2026-10-16), measured with the `release-tp3` suite
in one session (r6s, r6n, r6s again; [results](experiments/2026-10-16-r6s/README.md);
[suite run](manifests/suites/2026-10-16-karmic-kraken-r6s-tp3/)), each on its own
build's cost curves. Each change is shown with its 95% interval and judged against
the run's own repeat; it counts (**bold**) only beyond both.

| Workload | TP3 r6n | TP3 r6s |
| --- | ---: | ---: |
| Source-text prefill, 32K (tok/s) | 4,199 | 4,425 (**+5.4%** ±1.8) |
| Prefill, 128K | 4,128 | 4,360 (**+5.6%** ±1.0) |
| Prefill, 256K | 3,963 | 4,164 (**+5.1%** ±0.8) |
| Prefill, 500K | 3,636 | 3,812 (**+4.8%** ±0.2) |
| Decode step time, geometric mean of 15 shapes | | +0.5% ±0.14 (repeat -0.4%) |
| Decode, 8 different prompts at once (tok/s) | 145.6 | 146.2 (+0.4% ±1.0) |
| Decode, 4 different prompts at once | 112.5 | 112.3 (-0.2% ±2.0) |
| Poisson arrivals at 0.42 req/s, output (tok/s) | 117.0 | 117.3 (+0.2% ±21) |

Decode is level: the first r6s arm's +0.5% is within the run's own repeat (the second
r6s arm is +0.1% against r6n). All 256 greedy responses are identical to r6n's, and the
teacher-forced distributions are identical (KL 0). These r6n figures are 0.3-4% above
r6n's own release run of 2026-10-15 (below, 8 streams the most); compare within a session.

r6n against r6h on three nodes (2026-10-16), measured with the `release-tp3` suite
in one session (r6n, r6h, r6n again; [results](experiments/2026-10-15-r6n-tp3/README.md);
[suite run](manifests/suites/2026-10-15-karmic-kraken-r6n-tp3/)), r6n on its own
build's cost curves. Each change is judged against the run's own repeat.

| Workload | TP3 r6h | TP3 r6n |
| --- | ---: | ---: |
| Decode step time, geometric mean of 15 shapes | | **-5.5%** ±0.14 |
| Decode, 8 different prompts at once (tok/s) | 137.1 | 139.7 (**+1.9%** ±1.3) |
| Decode, 4 different prompts at once | 109.3 | 112.1 (**+2.5%** ±2.2) |
| Poisson arrivals at 0.42 req/s, goodput (req/s) | 0.280 | 0.290 |
| Source-text prefill, 32K (tok/s) | 4,200 | 4,181 (-0.4% ±1.5) |
| Prefill, 128K | 4,128 | 4,114 (-0.3% ±1.2) |
| Prefill, 256K | 3,970 | 3,948 (-0.5% ±1.2) |
| Prefill, 500K | 3,638 | 3,625 (-0.4% ±0.3) |

Every decode shape is shorter (3.2-7.3%); prefill moved within the repeat's own -0.4
to -0.5%, so no size counts as slower. r6n's text differs from r6h's as r6k's did on
TP4 (top-token agreement 95.0%, KL 0.022 nats).

r6n against r6k on four nodes (2026-10-15), measured with the release suite
in one session (r6n, r6k, r6n again; [results](experiments/2026-10-15-r6n/README.md#release-against-r6k-2026-10-15-1925-utc);
[suite run](manifests/suites/2026-10-15-karmic-kraken-r6n-tp4/)). Each change
is shown with its 95% interval and judged against the run's own repeat; it
counts (**bold**) only beyond both.

| Workload | TP4 r6k | TP4 r6n |
| --- | ---: | ---: |
| Greedy responses identical, 256 prompts | | 256 of 256 (KL 0) |
| Decode step time, geometric mean of 20 shapes | | **-2.21%** ±0.15 |
| Decode step, 1 stream x 3 rows at 4K (ms) | 19.61 | 19.21 (**-2.1%** ±1.3) |
| Decode step, 1 stream x 3 rows at 256K | 19.75 | 18.88 (**-4.4%** ±1.2) |
| Decode step, 16 streams x 96 rows at 4K | 51.49 | 51.01 (**-0.9%** ±0.7) |
| Decode, 16 different prompts at once (tok/s) | 224.6 | 226.5 (+0.9% ±1.0) |
| Decode, 8 different prompts at once | 180.8 | 181.2 (+0.2% ±0.7) |
| Source-text prefill, 32K (tok/s) | 5,685 | 5,714 (+0.5% ±0.3) |
| Prefill, 128K | 5,642 | 5,669 (+0.5% ±1.2) |
| Prefill, 256K | 5,407 | 5,430 (+0.4% ±0.5) |
| Prefill, 512K | 4,942 | 4,959 (+0.4% ±0.4) |
| Prefill, 1M | 4,237 | 4,248 (+0.3% ±0.7) |

Every one of the 20 decode shapes is shorter. The 32K prefill gain is beyond its
interval but not beyond the repeat's -0.3%, so it counts as level. r6m, the
candidate before r6n, was 10-13% slower at every prefill size because its Engram
gather polled host memory from every block
([engram-rows-v2](experiments/2026-10-15-engram-rows-v2/README.md)).

r6k against r6h on four nodes (2026-10-13), measured with the
[release comparison](docs/lab.md#release-comparison) in one session, candidate
first ([results](experiments/2026-10-13-r6k/README.md);
[report](manifests/benchmarks/2026-10-13-karmic-kraken-r6k-tp4.json)). Each
change is shown with its combined 95% interval and counts (**bold**) only
beyond it.

| Workload | TP4 r6h | TP4 r6k |
| --- | ---: | ---: |
| Temperature 0, 15 streams + a long prompt (requests differing) | 0 of 16 | 0 of 16 |
| Quality gate | 5/5 | 5/5 |
| Decode, 1 stream, 16 prompts paired | | +0.6% ±2.4 |
| Decode, 4 different prompts at once (tok/s) | 133.9 | 134.5 (+0.4% ±0.9) |
| Decode, 8 different prompts at once | 177.2 | 178.3 (**+0.6%** ±0.3) |
| Decode, 16 different prompts at once | 225.8 | 223.9 (-0.8% ±0.9) |
| Source-text prefill, 32K (tok/s) | 5,835 | 5,825 (-0.2% ±1.4) |
| Prefill, 128K | 5,717 | 5,761 (+0.8% ±0.9) |
| Prefill, 256K | 5,490 | 5,519 (+0.5% ±0.6) |
| Prefill, 512K | 5,046 | 5,086 (**+0.8%** ±0.3) |
| Prefill, 1M | 4,378 | 4,400 (**+0.5%** ±0.1) |

r6k is never slower on these measures, and it is faster at eight prompts and on
the longest prefills, while computing WO at the reference's precision. Its
temperature-0 text differs from r6h's: scored on the same tokens, r6h agrees
with r6k on the most likely token at 95.1% of positions, with a KL divergence of
0.021 nats ([servebench](experiments/2026-10-14-servebench/README.md)). The
bench's single-prompt decode points follow the text and moved 9-14% either way
with draft acceptance, so the table judges different prompts at once.

r6h against r6d (2026-10-12), measured with the
[release comparison](docs/lab.md#release-comparison) in one session per
topology ([results](experiments/2026-10-12-r6h/README.md); reports for
[TP4](manifests/benchmarks/2026-10-12-karmic-kraken-r6h-tp4.json) and
[TP3](manifests/benchmarks/2026-10-12-karmic-kraken-r6h.json)). TP4 ran the
candidate first after a 30-minute cool-down; the TP3 run is trimmed and ran r6d
first. Each change is shown with its combined 95% interval and counts (**bold**)
only beyond it.

| Workload | TP4 r6d | TP4 r6h | TP3 r6d | TP3 r6h |
| --- | ---: | ---: | ---: | ---: |
| Temperature 0, 15 streams + a long prompt (requests differing) | 13 of 16 | **0 of 16** | 1 | **0** |
| Quality gate | 5/5 | 5/5 | 5/5 | 5/5 |
| Decode, 1 stream, 16 prompts paired | | **+2.6%** ±2.5 | | -1.0% ±2.3 |
| Decode, 4 different prompts at once (tok/s) | 134.0 | 136.5 (+1.9% ±2.1) | 107.5 | 107.1 (-0.4% ±1.1) |
| Decode, 8 different prompts at once | 178.5 | 184.7 (**+3.5%** ±1.5) | 137.2 | 134.9 (**-1.6%** ±0.6) |
| Decode, 16 different prompts at once | 224.1 | 226.5 (+1.1% ±1.5) | | |
| Source-text prefill, 32K (tok/s) | 5,781 | 5,870 (**+1.5%** ±0.8) | 4,304 | 4,252 (-1.2% ±1.5) |
| Prefill, 128K | 5,669 | 5,726 (**+1.0%** ±1.0) | | |
| Prefill, 256K | 5,429 | 5,496 (**+1.3%** ±0.4) | 4,123 | 4,060 (**-1.5%** ±0.4) |
| Prefill, 512K (TP3 500K) | 4,998 | 5,074 (**+1.5%** ±0.2) | 3,797 | 3,753 (**-1.2%** ±1.0) |
| Prefill, 1M | 4,346 | 4,391 (**+1.0%** ±0.3) | | |

On four nodes r6h is at or above r6d on every metric: faster on prefill at
every size, at eight concurrent prompts and on the paired sweep, level
elsewhere (the mixed-traffic p95 stream gap read +3.9%, a single run without an
interval). On three nodes it trails at eight prompts and on long prefill, by
about 1.5%: each rank holds a third of the vocabulary and experts, so the
reference-precision arithmetic costs more, and the head fix applies only past
64 rows, which three nodes never reach. The bench's concurrent decode points send
one prompt to every stream and follow the text, which a precision change
alters (code at one stream read +21%, prose -5%); the release comparison
judges throughput on different prompts at once and on the paired sweep.

The r6 acceptance benchmark (2026-10-05, `bin/spark bench` from dgx1:
quality, then prose and code prompts with reasoning on, temperature 0, 256
output tokens, three samples per decode point, and real source text for
prefill, two repeats) against r5p's, one boot each at the recipe's limits.
Each value is the mean ± its 95% confidence interval; a family is ahead only
where the two intervals do not overlap (**bold**).

Three nodes (TP3, eight sequences at 512K):

| Workload | B12X r5p | TileLang r6 |
| --- | ---: | ---: |
| Quality gate (fixed LRU task, 5 repeats) | 5/5 | 5/5 |
| Prose, 1 stream (aggregate tok/s) | 50.4 ± 12.2% | 52.2 ± 0.7% |
| Prose, 2 streams | 75.9 ± 10.6% | 84.4 ± 0.6% |
| Prose, 4 streams | 118.8 ± 12.9% | 125.7 ± 0.5% |
| Prose, 8 streams | 173.3 ± 8.4% | **198.6 ± 0.4%** |
| Code, 1 stream | 60.1 ± 2.0% | **63.0 ± 0.5%** |
| Code, 2 streams | 90.8 ± 5.0% | 97.3 ± 8.5% |
| Code, 4 streams | 136.3 ± 6.3% | 159.2 ± 44.2% |
| Code, 8 streams | 192.3 ± 3.2% | **240.3 ± 1.0%** |
| One-stream prose / code step time | 41.92 / 46.44 ms | **38.21 / 42.43 ms** |
| Source-text prefill, 32K (tok/s) | 3,778 ± 14.9% | 4,306 ± 20.9% |
| Source-text prefill, 256K | 3,632 ± 2.2% | **4,034 ± 0.3%** |
| Source-text prefill, 500K | 3,432 ± 0.6% | **3,743 ± 0.8%** |
| Lowest MemAvailable, dgx1 | 5.93 GiB | 5.99 GiB |

Four nodes (TP4 1M recipe, 16 sequences at 1,048,576 tokens):

| Workload | B12X r5p | TileLang r6 |
| --- | ---: | ---: |
| Prose, 1 / 2 / 4 streams | 62.1 ± 3.8% / 92.7 ± 8.0% / 141.6 ± 7.9% | 62.4 ± 1.2% / 100.2 ± 2.7% / 147.1 ± 10.2% |
| Prose, 8 / 16 streams | 215.7 ± 7.5% / 297.3 ± 1.2% | 222.1 ± 0.3% / **318.4 ± 1.6%** |
| Code, 1 / 2 / 4 streams | 72.5 ± 7.5% / 111.3 ± 5.4% / 168.3 ± 9.6% | 76.0 ± 0.6% / 116.8 ± 18.9% / 174.9 ± 0.6% |
| Code, 8 / 16 streams | 245.7 ± 1.3% / 323.5 ± 1.5% | **264.3 ± 0.6% / 349.8 ± 2.5%** |
| One-stream prose / code step time | 34.00 / 37.72 ms | **31.79 / 34.75 ms** |
| Source-text prefill, 32K / 256K (tok/s) | 5,117 ± 1.7% / 4,873 ± 2.5% | **5,836 ± 4.2% / 5,471 ± 3.1%** |
| Source-text prefill, 500K / 1M | 4,645 ± 1.7% / 4,026 ± 0.9% | **5,072 ± 1.3% / 4,370 ± 0.6%** |
| KV capacity, 10.5 GiB per rank | 11 full 1M contexts at once (the startup line at 8,192 batched tokens: 8,580,566 tokens) | same |
| Lowest MemAvailable | 20.4 GiB (dgx1) | 19.35 GiB (dgx1) |

TileLang is never behind: it is ahead on single-stream step time, at eight
and sixteen streams and on every prefill point except TP3's 32K, and level
elsewhere. B12X's intervals are wider because its text changes from sample to
sample (DSpark acceptance moves with it); TileLang generates the same text
every sample, so its intervals are tight except where two- and four-stream
scheduling interleaves differently between samples. The intervals cover
sample-to-sample spread on one text; across 13 texts at TP3, DSpark
acceptance per text varied about ±5% while the two families' means were level
([TP3 benchmark](experiments/2026-10-05-tp3-benchmark/README.md)). Step time,
which does not depend on acceptance, is the kernel comparison. The B12X TP4
500K point is from [its own run](experiments/2026-10-05-tp4-500k/README.md).

The native benchmark reports
([TP3](manifests/benchmarks/2026-10-05-karmic-kraken-r6.json),
[TP4](manifests/benchmarks/2026-10-05-karmic-kraken-r6-tp4.json), and r5p's
[TP3](manifests/benchmarks/2026-10-05-karmic-kraken-r5p.json) and
[TP4](experiments/2026-10-05-r5p-promotion/runs/tp4-bench.json)) record
intervals, prompts, memory and thermal results. The
[64 KiB deployment record](experiments/2026-10-02-memory-saver-capacity/README.md)
covers the TP3 profile's long-context admission (four simultaneous 485K-token
contexts) and near-limit retrieval (3/3 at 519,142 tokens), measured on r5o.

The tables below retain the historical **4 KiB r5l** reference (r5o added the
top-k tie rule and TMA stage-release fences). These were measured with
`bin/spark bench` from dgx1: prose and code
prompts, temperature 0, 256 output tokens. With reasoning on (the server
default) every measured token is reasoning text:

| Prompt | Streams | Aggregate tok/s | Per-stream decode tok/s | First token |
|---|---:|---:|---:|---:|
| prose | 1 | 50.3 | 52.2 | 0.20 s |
| prose | 2 | 76.1 | 41.4 | 0.33 s |
| prose | 4 | 111.8 | 31.0 | 0.42 s |
| prose | 8 | 162.0 | 22.4 | 0.53 s |
| code | 1 | 59.7 | 62.8 | 0.23 s |
| code | 2 | 90.4 | 49.6 | 0.35 s |
| code | 4 | 131.4 | 36.6 | 0.45 s |
| code | 8 | 187.4 | 26.6 | 0.55 s |

With reasoning off (the answer itself), aggregate tok/s at 1/2/4/8 streams:

| Prompt | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| prose | 59.1 | 84.8 | 124.9 | 174.5 |
| code | 80.6 | 118.2 | 165.1 | 233.0 |
| JSON | 77.0 | 109.7 | 165.2 | 234.3 |

| Other measurements | |
|---|---|
| Quality gate (fixed LRU task, 5 repeats) | 5/5 |
| Long-context retrieval (phrase at 10%, 50%, 90% depth) | 3/3 at 152,914 tokens |
| Single-stream decode step | about 42 ms on prose and 47 ms on code; accepted drafts per step 1.2 (prose), 1.9 (code), 3.2 (code answers) |
| Cold prefill, repeated filler | 2K 4.4k, 32K 4.9k, 64K 4.9k, 131K 4.7k tok/s |
| Cold prefill, real text (Python source) | 4K 3.9k, 16K 3.8k, 32K 3.8k, 64K 3.8k, 131K 3.8k, 200K 3.8k tok/s |
| Prefix-cache replay, 32K prompt | 6.47 s cold, 0.24 s warm |
| Four concurrent 64K contexts | all admitted without preemption, peak KV use 20%, 13.1 tok/s per stream |
| Four concurrent 180K contexts (r5k) | all admitted without preemption, peak KV use 36%, 11.9 tok/s per stream |
| KV capacity | all eight sequences at 262,144 tokens at once in 2.2 GiB per rank (vLLM's startup line: 1,348,708 tokens, 5.1 full 256K contexts) |
| Host memory headroom | dgx1 at least 6.38 GiB MemAvailable under load (3 GiB guard); startup passes the 5 GiB guard |

The quick default takes three or four samples per decode point, about ±2-12%
at 95% confidence; temperature-0 outputs differ between identical requests
(the indexer's radix top-k keeps an arbitrary subset of positions tied at its
threshold), which moves acceptance from sample to sample. Report:
[decode, prefill, prefix cache, and admission](manifests/benchmarks/2026-09-29-karmic-kraken-r5l.json).
See [Benchmarking](#benchmarking) to reproduce them.

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

`doctor`, `status`, `render`, and every cluster command without `--apply` are
read-only. `bench` only sends API requests; see [Benchmarking](#benchmarking).

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
checkout at that exact revision; it never copies a working tree or ignored files.
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

A release is benchmarked against the previous one with servebench suites in a lab session:
[docs/release-benchmark.md](docs/release-benchmark.md) has the commands. `bin/spark bench`
measures the live service from the head node and writes one
self-contained report to `results/private/bench/<UTC time>/bench.json`
(`--output` to change). By default it is a quick check of about six minutes:
the quality gate and every decode point with three or four samples each.
`bin/spark bench --full` runs every suite below and samples each decode
point to its precision target, about 35 minutes; use it for experiments that
need to resolve small differences. Before sending anything it runs the `doctor --live`
comparison and refuses a cluster that differs from its configuration. It also
records the configuration hash, each node's image ID and checkout, and the
client commit.

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
