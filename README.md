# spark-ds41f

Reproducible Docker/vLLM deployment, tuning, and benchmarks for DeepSeek V4.1
Flash on a switchless three-node DGX Spark fabric.

The deployment tools also generate and validate a
[four-node switchless ring profile](docs/switchless-topology.md) with the
balanced [RoCEnante ring relay](experiments/2026-10-02-rocenante-ring4/README.md)
and tuned NCCL. Its [TP4 1M recipe](experiments/2026-10-04-tp4-memory-tuning/README.md)
serves the checkpoint's full 1,048,576-token context with 16 sequences on the
same image as the three-node deployment, and was benchmarked at those limits
(see [Performance](#performance)). See the
[serving decision](experiments/2026-10-03-collective-serving/decision.md),
[hardware results](experiments/2026-10-02-rocenante-mesh4/README.md) and
[four-path transport comparison](experiments/2026-10-02-mesh4-fourpaths/README.md).
The promoted configuration below is the three-node deployment.

Named [TP3/TP4 transport tuning profiles](experiments/2026-10-03-transport-profiles/README.md)
keep the measured limits and NCCL settings together. `bin/spark tuning show tp4`
shows the TP4 recipe; `tuning create` generates a complete configuration
for a site node map. The generated configuration starts with launch disabled.

The TileLang kernel family (TileLang, DeepSeek's TileKernels and sparknet
collectives in place of B12X's kernels) is an alternative kernel backend
(`kernel_backend: tilelang`, see `scripts/kernel_backend.py`). Its TP3 and TP4
configurations ([TP3](experiments/2026-10-05-tp3-benchmark/tilelang.json),
[TP4](experiments/2026-10-04-tilelang-1m/candidate.json)) were benchmarked
against r5p's in the same windows
([TP4](experiments/2026-10-05-tp4-validation/README.md),
[TP3](experiments/2026-10-05-tp3-benchmark/README.md)).

This repository is being promoted from a forensic capture of the running cluster
into its only operational source of truth. Until the transition checklist is
complete, files are explicit about whether they describe observed state, desired
state, or an experiment.

## Current baseline

The active baseline is recorded in
[manifests/baselines/2026-10-05-karmic-kraken-r5p.json](manifests/baselines/2026-10-05-karmic-kraken-r5p.json)
([promotion record](experiments/2026-10-05-r5p-promotion/README.md)):

- three DGX Spark nodes using tensor parallelism 3, on DGX Spark 26.09.2 with
  kernel `7.0.0-1019-nvidia-64k` (`kho=off`), signed memory-saver DKMS
  0.2.0 with UVM leaf-table packing enabled, no desktop, and
  `vm.watermark_boost_factor=0`;
- direct dual ConnectX-7 paths between every pair of nodes;
- Local Inference Lab's `integration/karmic-kraken-beta` vLLM (plus Engram
  projection sharding, asynchronous Engram rows, and two tool-call and
  image-cache fixes) and B12X (plus the switchless RoCEnante patch, and its
  CuTe DSL pin moved to the 4.7.1 that vLLM requires; its FP4 KV writer
  rounds like DeepSeek's reference quantizer, and its indexer top-k breaks
  score ties by position, so selections repeat, and its dense GEMM and prefill
  kernels fence shared-memory stage reads before the TMA refill), with B12X attention,
  linear, MoE, and mHC kernels and L2 weight prefetch during decode (the
  next layer's weights stream into L2 while latency-bound kernels run); the
  image also carries B12X's four-node RoCEnante relay and mesh transports,
  inactive on the triangle;
- NCCL 2.30.7 rebuilt with the AArch64 InfiniBand send-path fence
  (NVIDIA/nccl#2393), which prevents a proxy-thread hang, and the four-node
  switchless ring patches (bidirectional rings, balanced channels);
- DeepSeek V4.1 Flash native FP8/FP4/BF16 weights, every served weight bit as
  the checkpoint stores it, with the vision tower loaded (up to four images
  per request); the output head is stored in an exact 12-bit packed form of
  its BF16 weights (a sign-and-mantissa byte and a 4-bit exponent code, with
  out-of-window values kept in a per-row list), and the embedding and output
  head live in the GB10 display carve-out, the firmware's scanout reserve
  that ordinary allocations never use, loaded there directly from the
  checkpoint while the text console keeps its framebuffer;
- the ratio-2 compressor carries its open pair across decode steps, so
  compressed entries for generated tokens match the ones prefill builds;
- DSpark speculative decoding with five draft tokens (the drafter's trained
  block) and block rejection, full CUDA graphs for decode batches up to 48
  tokens; verification rows whose drafts are unlikely to survive skip the
  routed experts, and greedy drafts stay sharded by vocabulary over the
  checkpoint's BF16 drafter head (the target head's own tensor) and Markov
  projection;
- the requested RoCE collective policy is enforced: a declined or lost RoCE
  backend fails startup rather than falling back to another path, and the
  fabricated context of startup profiling gives each KV cache group its own
  blocks;
- sequence-parallel prefill once a prompt chunk's reduce-scatter outgrows
  the one-shot RoCE all-reduce (205 tokens): the encoder layers' row-wise
  work runs on a third of the rows per rank, and so does the sparse-attention
  indexer, whose cost grows with context depth (a 4,096-token chunk at 200K
  of context runs 15% faster);
- mutating custom ops read their argument schema once per call, not once per
  argument as torch does, which keeps short prompts from waiting on the
  host at every MoE launch;
- B12X W4A8 tiny decode disabled (`B12X_W4A8_TINY_DECODE=0`): it omits the
  model's SwiGLU clamp and caused the incoherence seen in earlier images;
- 524,288-token per-request limit, eight admitted sequences, and 2,845,543
  KV tokens (5.43 full windows) in a 3.5 GiB-per-rank cache;
- one concurrent prefill, 4,096 batched tokens, and fail-closed 5 GiB startup
  and 3 GiB steady memory guards;
- FlashInfer autotune disabled (`--no-enable-flashinfer-autotune`): only the
  sampler uses FlashInfer, and the pass saved no configs.

One content-addressed image runs on all three nodes and passes the LRU
coherence gate 5/5; see [Performance](#performance).

The machine-readable desired configuration is [config/cluster.json](config/cluster.json).
The named [4 KiB and 64 KiB profiles](docs/memory-profiles.md) retain the
previous capacity as a fallback and select the larger 64 KiB profile by default.
To reproduce the deployment on your own three Sparks, follow
[docs/replicate.md](docs/replicate.md).

## Performance

The r5p acceptance benchmark (2026-10-05, `bin/spark bench` from dgx1:
quality, then prose and code prompts with reasoning on, temperature 0, 256
output tokens, three samples per decode point, and real source text for
prefill) measured, on the three-node deployment against r5o's 64 KiB
reference:

| Workload | r5o | r5p |
| --- | ---: | ---: |
| Quality gate (fixed LRU task, 5 repeats) | 5/5 | 5/5 |
| Prose, 1 / 2 / 4 / 8 streams (aggregate tok/s) | 52.8 / 79.4 / 117.9 / 171.2 | 50.4 / 75.9 / 118.8 / 173.3 |
| Code, 1 / 2 / 4 / 8 streams | 64.8 / 90.7 / 135.6 / 195.8 | 60.1 / 90.8 / 136.3 / 192.3 |
| One-stream prose / code step time | 40.84 / 45.36 ms | 41.92 / 46.44 ms |
| Source-text prefill, 32K / 256K / 500K | 3.77K / 3.68K / - tok/s | 3.78K / 3.63K / 3.43K tok/s |
| Lowest MemAvailable, dgx1 | 5.83 GiB | 5.93 GiB |

Every point is the same as r5o within its interval. Single-stream steps are
1.1 ms longer because the drafter reads the checkpoint's BF16 heads instead
of NVFP4 re-quantizations; the packed output head recovers part of that.

The TP4 1M recipe on four Sparks, same image and protocol, up to 16 streams
and 1M-token prefill:

| Workload | TP4 1M recipe |
| --- | ---: |
| Prose, 1 / 2 / 4 / 8 / 16 streams | 62.1 / 92.7 / 141.6 / 215.7 / 297.3 |
| Code, 1 / 2 / 4 / 8 / 16 streams | 72.5 / 111.3 / 168.3 / 245.7 / 323.5 |
| One-stream prose / code step time | 34.00 / 37.72 ms |
| Source-text prefill, 32K / 256K / 1M | 5.12K / 4.87K / 4.03K tok/s |
| KV capacity | 8,580,566 tokens in 10.5 GiB per rank (8.18 full 1M windows) |
| Lowest MemAvailable | 20.4 GiB (dgx1) |

The native benchmark reports
([TP3](manifests/benchmarks/2026-10-05-karmic-kraken-r5p.json),
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
| KV capacity | 1,348,708 tokens in 2.2 GiB per rank (5.1 full 256K contexts) |
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
it. The command is `bin/spark`; dated experiment records and their scripts
keep `bin/spark3` as they ran it.

Names built into images or installed on the hosts keep `spark3` until they
are rebuilt or reinstalled:
- the `SPARK3_*` environment variables that the vLLM patches read;
- the `local.spark3.*` image labels and the `/opt/spark3` paths inside the
  image;
- the hosts' `spark3-kernel-memory` service and RoCE marker.

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
bin/spark cluster sync
bin/spark cluster start --replace
bin/spark cluster stop
bin/spark bench
bin/spark upstream list
bin/spark upstream prepare vllm
bin/spark upstream prepare b12x
bin/spark build prepare
bin/spark build check
bin/spark build image
bin/spark build image --apply
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

`bin/spark bench` measures the live service from the head node and writes one
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
  Single boots of one configuration differ by about 3%, because adaptive
  verification profiles its costs at startup, so effects smaller than that
  need several boots per arm.
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

With a reference run (by default
`manifests/benchmarks/<promoted_baseline>.json`, or `--compare PATH`), each
decode and prefill point shows its percent change with a Welch 95% interval.
The command exits non-zero on a failed quality or admission check, any failed
request, an early stop, or a point that is significantly slower by more than
`--tolerance` (3%). A promotion adds its reference run to
`manifests/benchmarks/`.

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
`integration/karmic-kraken-beta` vLLM and B12X with the local patch series
([docs/upstreams.md](docs/upstreams.md)).
