# NVMe KV tier for session state

Date: 2026-11-22 to 2026-11-25.
Base: the release `spark-ds41f-1120`, which is vLLM `local-inference-lab/vllm` at `19f2c20e` with the
release's 702 patches (tree `8fc99ca7`).
Cluster: three DGX Spark nodes (GB10) at tensor parallel 3 over a triangle of direct RoCE cables, shape
`tp3-triangle-4k`, on the stock kernel `7.0.0-1019-nvidia` with 4 KiB pages.

The fork's `SimpleCPUOffloadConnector` keeps KV blocks outside the GPU pool, and it already had a disk
backend. That backend did not start for DeepSeek-V4.1-Flash, so this experiment makes it start and adds
what it needs to serve: a checksum on every slot it reads, a file reserved in full at start, an eviction
order that keeps the start of each session, and fault handling that turns a failed or slow read into a
cache miss. The connector stores each full KV block in a preallocated file on the NVMe drive of each node
as soon as the block is computed, so the block is still on disk after it is evicted from the GPU pool,
and a session that returns after eviction loads its prefix from that file instead of being prefilled
again. The tests ran in one home lab, called the lab below. On the cluster above, a returning request
with a 500,000-token prefix took about 1.7 s while two or four other streams were decoding, of which
about 1 s was the load from disk, against about 62 s for the same request cold with no other request on
the engine.

The recipe is this repository's configuration file `config/cluster.json`, which holds the settings that
all shapes share and a table of shapes. A shape is one way of cabling the nodes, with its own node map,
transport, NCCL settings, and memory sizing (README.md of the repository). A release such as
`spark-ds41f-1120` is a GitHub release with a git tag of the same name and a folder `releases/<name>/`,
and its lock, `upstreams.lock.json`, pins the vLLM revision and the list of patches, which for 1120 is
the list kept in the experiment folder of `candidate-62` (AGENTS.md, "Hold benchmarks of unreleased work"
and "Sources of truth"). Promoted state is the accepted configuration in `config/` (docs/methodology.md,
"State model"). `candidate-73` is a later frozen candidate, published as a GitHub pre-release from the
repository's default branch, `main` (AGENTS.md, "Sources of truth").

The change is 17 vLLM patches, numbered 9030 to 9046. The file `vllm/series` names the release's list of
702 patches (candidate-62, 0001 to 0829) through an include line and then lists the 17, so the recipe's
build reads 719 entries from it. In this README "the series" means the 17 patches, and `vllm/series`
means the file. The last of them, 9046, lets the fork's sink rejection run while this one connector is
configured. Without it the tier cannot start, because sink rejection is on by default for this model and
the release refuses every KV connector while it is on.

Every hardware run used an image of the release, `spark-ds41f:1120`, which the lab built from the
release's lock, with the eight engine files that the series changes mounted over the image's copies.
These files are in `overlay/`. No image has been built from the series.

The same 17 patches apply without change after the 742 patches of the later release
`spark-ds41f-1126` (candidate-74), and `vllm/series-1126` names that list and the 17. Of the eight
overlay files, only `model_runner.py` differs between the two releases, because nine of the patches that
1126 adds change it, two of them for kernels with 64 KiB pages. The folder `overlay-1126/` therefore
holds that one file, built from 1126 and the series (tree `e45d318c`), and the profiles whose names end
in `-1126` mount it in place of the copy in `overlay/`. Those profiles extend the release's own recipe,
`releases/spark-ds41f-1126/cluster.json`, so they name the 1126 image and lock from any checkout. The
profiles without that suffix extend `config/cluster.json` and belong to a checkout of 1120. Started from a checkout of 1126, they mount the 1120 copy of `model_runner.py` over the
1126 image, so the engine runs without those nine patches. On 1126 the series has passed the CPU tests
only, and it has not run on hardware there.

## Terms

A disk hit is a request whose cached prefix the engine loaded from the tier file, and a pool hit is the
same request served from the KV cache in GPU memory. A profile is a file that extends one of the recipe's
shapes; the recipe also calls such a file an arm. Each hardware check is a scripted test, and every check
that ran had to pass before the lab kept the tier on in its own serving. A capacity check and a hung-read
check were also planned, and they did not run. The sink detector reads one column of the residual stream
(column 2455 for this model) after the last layer, at the row of each sampled token, and flags a row
whose value there passes a threshold of 85,000. The sink experiment found values from 108,032 to 125,440
at sunk tokens, where an answer collapses (`experiments/2026-10-19-sink-rejection-v2/README.md`, "The
threshold"). The detector reads a token's row in the step that takes the token as input, so sink
rejection holds each newly sampled token for one step as a provisional token, which the recipe calls
draft zero, and the next step either commits the token or replaces it ("The contract" in the same
README). The tier is degraded after a disk fault, which means that every later lookup misses until the
engine restarts. An overrun is a load or a store that is still making progress when it reaches its time
bound. The node that runs rank 0 also runs the scheduler, so the memory checks read that node.

## Hypothesis

A session that returns after its prefix left the GPU pool loads that prefix from the tier in less time
than a cold prefill of the same prompt. The disk hit returns the same tokens and top-5 logprobs as a pool
hit, and streams that are decoding at the same time keep at least 0.90 of their decode rate, measured
over the whole run of the streams against a run in which no session is sent.

## Changed variable

The one variable is the tier. The profile `tp3-4k-b6144-m1m-kv35-nvme200.json` is the profile
`tp3-4k-b6144-m1m-kv35.json` with `--kv-transfer-config` added and the eight overlay files mounted. The
base profile is included as the starting point of the measurement, and this pull request does not propose
its serving options for the recipe. The base profile differs from the promoted `tp3-triangle-4k` in four
serving options: a prefill budget of 6,144 tokens, prefill chunks of 6,096 tokens, a context of 1,048,576
tokens, and a KV pool of 3.5 GiB a rank. The tier has not run on the promoted shape without those
changes.

The lab's copy of `config/cluster.json` also differed from the release's tag, `spark-ds41f-1120`, in four
settings besides the node addresses. It had no `glp` block, so the engine served only the stock model
name; a profile with that block also serves a second name, `deepseek-v4.1-flash-weightless`, steered per
request by a GLP vector from a file on Hugging Face (docs/setup.md, "GLP steering file"). It also left
`NCCL_IB_GID_INDEX` unpinned and set `host.fan_control` to false. It turned off the display carve-out
weights for `tp3-triangle-4k` by setting `SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS` to `"0"`, so the
embedding and output-head tables stayed in regular memory instead of in the memory reserved for the
display, where TP3 keeps them by default (docs/current-state.md). None of these settings is part of this
change, and `base.json` lists them. Their effect on the figures below was not measured.

## Risks

A disk hit could return a wrong prefix without an error, for example after a misdirected read or a slot
that was rewritten for another block. The slot header, the checksum, and the block key exist to turn such
a read into a cache miss, and the equivalence check compares disk hits with pool hits token by token.

A single disk fault degrades the whole tier until the next engine start, so one bad read on one rank
turns the tier off for every session. The request that hit the fault is computed again.

A restore reads from disk while other requests decode, so it can slow them down. The decode check
measures that slowdown, and the two memory checks read the free memory of the node that runs rank 0 while
the tier starts and while it serves.

Patch 9046 lets sink rejection run beside the connector because a reading of the code found no path on
which the connector changes a request that holds a provisional token. If the refusal of KV transfer
covers a case that the reading missed, the most likely symptom is an assertion in the worker or the
scheduler.

All eight overlay files replace the image's copies whole, so the two profiles with a tier assume an image
whose vLLM tree is `8fc99ca7`, and the profiles that end in `-1126` assume the tree `f3783b0e` of 1126 in
the same way. The two files outside the connector, `model_runner.py` and `sink.py`,
carry the larger risk, because the whole engine runs them and later candidates change them:
`candidate-73` changes `model_runner.py`. The lab's check before the start therefore compared those two
files with the base tree. The other six are files of the connector that, between `8fc99ca7` and
`candidate-73`, only the series changes.

## Acceptance criteria

Between 2026-11-22 and 2026-11-24 the lab wrote the criteria into its plan for the run, and two of them
changed. The memory thresholds were lowered on 2026-11-24, before the run started, for the reason given
under "Hardware runs". The equivalence check also required that the answers to three prompts, which hid a
code in their filler text and asked for it at the end, contain that code; during the run this requirement
was relaxed, as "Runs that failed" describes. A disk hit must equal a pool hit at the same block boundary
in every generated token and in the set of top-5 token ids at every position, and each of its top-5
logprobs must lie within a tolerance of the pool hit's. The tolerance is the largest logprob difference
between two pool hits of the same request. Each disk hit must be proved to come from the tier by the
prefix-cache counters. Streams that decode while a 500,000-token session restores must keep at least 0.90
of their decode rate, measured over the whole run of the streams against a run in which no session is
sent. The free memory of the node that runs rank 0 must stay above a threshold of 5,250 MiB while the
tier starts and above 4,200 MiB while it serves; the reason for those thresholds is under "Hardware
runs". A disk fault on one rank must end as a recomputed request with the tokens of a cold run and a
degraded tier. No load may fail in the checks that do not inject a fault.

## Design

The connector already had a disk backend, which wrote offload blocks through pinned staging slots with
`pwritev` and `preadv`. The series makes that backend start for this model and keeps it safe while it
serves.

### Slot layout

The packed block of this model is 227,840 bytes, which is not a multiple of 4096. Before the series, the
disk backend refused that size at start with a check of its own, which required a multiple of 4096 under
O_DIRECT; the refusal did not come from the kernel. The NVMe drives of the lab's three nodes have 512-byte
logical blocks, and every unpadded slot offset is a multiple of 512, so Linux would have accepted those
slots there. A drive with 4096-byte logical blocks refuses an O_DIRECT offset, length, or buffer address
that is not a multiple of 4096 with `EINVAL`, and an unpadded slot after the first one starts at such an
offset. That case was not reproduced, because no drive in the lab has 4096-byte logical blocks. Patch 9030
therefore stores each block in a slot of 229,376 bytes, padded to a multiple of 4096 bytes.
The padding holds a 40-byte header with a magic number, a version, the checksum kind, the slot id, a
generation count, an 8-byte block key, and a checksum of the block bytes, which is CRC32 by default
(patches 9034 and 9039). The block key is a BLAKE2b digest of the block hash and the KV cache group, so a
slot that was rewritten for another block between the lookup and the load fails the check. The load
thread checks the header before it copies anything to the GPU.

### Tier file

The worker reserves the whole tier file with `posix_fallocate` at start, so a start without enough space
fails at once (9035). Each rank has its own file, whose name ends with `.rank_<device index>`. The worker
removes the file at start, so the tier is empty after every engine start.

### Hit boundary

DeepSeek-V4.1-Flash keeps sliding-window KV for its decoder layers and for its DSpark drafter, which
proposes tokens for speculative decoding (README.md of the repository). Those windows are private to each
request and are not prefix-cached, so on a hit the engine rebuilds them by computing the last 128 prompt
tokens again; the V4.1 paper calls this bounded replay, and the bounded-replay boundary is the point 128
tokens before the end of the prompt, beyond which no hit may reach
(`experiments/2026-10-19-prefix-replay/README.md` and
`experiments/2026-10-26-ced-prompt-anchor/README.md`). A disk hit now ends at the same boundary as a pool
hit (9032). Before 9032, a short new turn after a long cached prompt could get a disk hit that left fewer
than 128 tokens; a CPU test shows a turn of 64 tokens with only 64 left to compute. The expected effect
is a changed answer with no error. That effect was not observed, because the tier never ran on hardware
without 9032.

### Eviction

When a request finishes, the tier refreshes the blocks of its prefix in order from the last to the first,
so the first block ends at the recent end of the LRU list (9037). A full tier therefore evicts each
session from its end, and the start of a recent session stays usable. On the hardware, eviction ran only
on the tier of 4 GiB on each rank.

### Faults

Before the series, an exception in an IO thread ended the thread, and the failed load and every later
load on that rank were never reported, so each request with a disk hit waited forever while `/health`
stayed up. A CPU-only probe showed this; it was not seen on hardware. Since 9033, a failed, short,
corrupted, or hung read, a failed write, or a stopped IO thread on any rank fails the transfer. A failed
load is then a cache miss on every rank, and the request is computed again because
`kv_load_failure_policy` is `recompute`. After the first fault the tier is degraded, so every lookup
misses and no store is selected until the next engine start.

### Time bounds

A transfer that completes no slot for `disk_io_timeout_s`, counted from the moment its IO thread takes it
from the queue, is a fault (9040). A transfer that keeps making progress but runs past its total bound,
which is `disk_io_timeout_s` plus `disk_io_slot_budget_s` multiplied by the number of slots in the
transfer, is an overrun (9041 and 9042). An overrun is a miss for the requests of that transfer, and the
tier stays up. The retry of such a request is a cold prefill, so a slow disk cannot hold one request in a
loop of loads (9043 and 9044).

### Expandable segments

The recipe sets `expandable_segments:True`, and the connector declares itself safe under that setting,
because it registers no GPU pages (9031).

### Sink rejection

Patch 9046 adds the function `kv_transfer_refusal` in `vllm/v1/sample/sink.py`, which allows
`SimpleCPUOffloadConnector` with `kv_role` `kv_both` while rejection is on. It still refuses every other
connector and role, a `MultiConnector`, and a `kv_connector_module_path`. The reasoning behind the patch,
and the question it leaves for the author of the refusal, are under "Sink refusal of KV transfer" at the
end.

## Settings

The flag `--kv-transfer-config` turns the tier on. The values below are those of the 200 GiB profile.

| Field | Value | Meaning |
|---|---|---|
| `kv_connector` | `SimpleCPUOffloadConnector` | the connector |
| `kv_role` | `kv_both` | stores and loads in one engine; the only role that 9046 allows under rejection |
| `kv_load_failure_policy` | `recompute` | a failed load is a cache miss; the default `fail` ends the request with an error |
| `kv_offload_backend` | `disk` | GPU to NVMe through pinned staging slots |
| `disk_path` | `/cache/kvtier/ds41f` | the file of each rank is `<disk_path>.rank_<device index>` |
| `disk_capacity_bytes` | 214748364800 | 200 GiB a rank, or 936,228 slots |
| `disk_preallocate` | `true` | reserve the whole file at start (the default) |
| `disk_checksum` | `crc32` | the checksum in each slot header (the default); also `sum64` or `none` |
| `disk_io_timeout_s` | 15 | seconds without a completed slot before a transfer is a fault (the default) |
| `disk_io_slot_budget_s` | 0.02 | seconds that each slot adds to the total bound of a transfer (the default) |
| `disk_buffer_slots` | 4 | pinned host staging slots in each direction (default 2) |
| `use_page_cache` | `false` | O_DIRECT, because the page cache would take host memory |
| `lazy_offload` | `false` | every hashed block is stored when it is computed |

In the profiles, `/cache` is a mount of `{checkout}/cache`, where `{checkout}` is `deployment.path`, the
node's clone of this repository (docs/setup.md, the node paths of a profile). Both time bounds must be
finite and above 0, and the worker logs both of them at start, one line for each rank. The debug variable
`VLLM_KVTIER_DEBUG_READ_DELAY` lets a test hang a read, and it is off unless it is set. Its value has the
form `<read>:<seconds>[:<rank>]`. The worker then waits `<seconds>` before the slot read whose number is
`<read>`, counting the slot reads of the process from 1. The wait happens on every rank, or only on
`<rank>` when that part is given.

## Metrics

The scheduler exports five counters: `vllm:kv_offload_disk_loads_total`, which counts load events,
`vllm:kv_offload_disk_load_failures_total`, `vllm:kv_offload_disk_stored_blocks_total`,
`vllm:kv_offload_disk_store_failed_blocks_total`, and `vllm:kv_offload_disk_io_errors_total`. It also
exports two gauges. `vllm:kv_offload_disk_degraded` reads 1 once the tier is degraded, and
`vllm:kv_offload_disk_load_seconds_max` holds the slowest load since the start, including the time in the
queue and the wait for every rank.

## Patches

The numbers start at 9030 because `main` already uses 0830 to 0868 for the recipe's own patches. Each
patch header states the upstream base, the rationale, the quality implications, the tests, and the
upstream status.

| Patch | Change |
|---|---|
| 9030 | pad disk slots to 4096 bytes under O_DIRECT |
| 9031 | declare the connector safe under `expandable_segments` |
| 9032 | cap external hits at the bounded-replay boundary |
| 9033 | turn disk faults into cache misses, degrade the tier, and export the metrics |
| 9034 | check a slot header and a checksum on each load |
| 9035 | reserve the tier file at start |
| 9036 | test that a disk hit equals a pool hit on the DeepSeek-V4.1 KV layout |
| 9037 | evict a full tier from the end of each session's prefix |
| 9038 | test that refreshing a finished prefix keeps the pins of a load in flight and stops at a missing block |
| 9039 | add a block key to each slot header |
| 9040 | count the I/O timeout from the moment a transfer leaves the queue, and add the slowest-load gauge |
| 9041 | bound the total time of each disk transfer |
| 9042 | treat an overrun of the total bound as a miss |
| 9043 | make the retry of an overrun load a cold prefill |
| 9044 | end the overrun mark when the cold retry is admitted |
| 9045 | test that a degraded rank forgets the marks of abandoned transfers |
| 9046 | run sink rejection beside `SimpleCPUOffloadConnector` with `kv_both` |

With the recipe's `git am` settings, the release's patches and the series give the vLLM tree `47e196a2`,
whose last commit is `fddb82dc`. The 17 patches also apply without edits after the 741 patches of the
pre-release `candidate-73`, whose tree is `0aa091fa`, and give the tree `152445b9`. `candidate-73` is the
newest pre-release on `main`. Of the files that the series touches, `candidate-73` changes only
`vllm/v1/worker/gpu/model_runner.py`. Patch 9046 changes that file in three hunks: one imports
`kv_transfer_refusal`, and two in `_init_sink` call the function and replace one entry of the refusal
list. On `candidate-73` those hunks apply at offsets of 3, 671, and 671 lines. `candidate.json` holds the
identifiers for both bases.

Code comments and one test name inside 9030 to 9038 still carry the numbers 0830 to 0838, which these
patches had before they moved to 9030 (for example `# 0834: slot header` and
`test_without_0832_three_of_six_cases_over_hit`). Those numbers refer to patches of the series. The
recipe's own 0830 to 0838 are other patches, and the comments were left unchanged so that the files stay
equal to the files that ran on the hardware.

## Files

| Path | Content |
|---|---|
| `vllm/series`, `vllm/90*.patch` | the patch list and the 17 patches of the series |
| `vllm/series-1126` | the patch list of 1126 (742 patches) and the same 17 patches |
| `overlay/` | the eight files that the series changes, at tree `47e196a2`, with `SHA256SUMS` |
| `overlay-1126/` | `model_runner.py` from 1126 and the series, at tree `e45d318c`, with `SHA256SUMS`; the other seven files of 1126 equal those of `overlay/` |
| `tp3-4k-b6144-m1m-kv35.json` | the base profile, with no tier |
| `tp3-4k-b6144-m1m-kv35-nvme200.json` | the base profile with a tier of 200 GiB a rank |
| `tp3-4k-b6144-m1m-kv35-nvme4.json` | a test profile with a tier of 4 GiB a rank, a context of 262,144 tokens, and a KV pool of 0.75 GiB |
| `tp3-4k-b6144-m1m-kv35-1126.json` | the base profile on 1126, with no tier, which extends `releases/spark-ds41f-1126/cluster.json` and so pins the 1126 image and lock |
| `tp3-4k-b6144-m1m-kv35-nvme200-1126.json`, `tp3-4k-b6144-m1m-kv35-nvme4-1126.json` | the two profiles with a tier on 1126, which extend that base and mount `overlay-1126/vllm/v1/worker/gpu/model_runner.py` |
| `tools/make_overlay.sh` | rebuilds the vLLM tree from the series and compares it with `overlay/`, or with `overlay-1126/` and `overlay/` when `RELEASE=1126` |
| `tools/cpu_suite.sh` | runs the CPU tests of the series in a prepared tree |
| `base.json` | the release, the lock, the trees, and the lab's differences from the tag |
| `candidate.json` | the source identifiers |
| `runs/tier-gates.jsonl` | the check script's own record of every check, including the failed and skipped ones |
| `runs/kv-soak-12h.tsv` | the tier's counters every 30 s for the first 12 hours of the lab's traffic |
| `results.json` | the figures of every run, each with the line of its record |
| `decision.md` | what the lab decided, and what is still open |

The 4 GiB profile exists so that the tier fills and starts to reuse its slots within minutes. The file
`runs/kv-soak-12h.tsv` holds the rows from 01:00:25Z to 12:59:23Z on 2026-11-25. Each file of `overlay/` and `overlay-1126/` keeps
its vLLM licence header (Apache-2.0).

## Commands

```sh
experiments/2026-11-22-kv-disk-tier/tools/make_overlay.sh /tmp/kvtier
PYTHON=/path/to/venv/bin/python DS41_ODIRECT_DIR=/path/on/xfs \
  experiments/2026-11-22-kv-disk-tier/tools/cpu_suite.sh /tmp/kvtier/vllm
bin/spark --cluster-config experiments/2026-11-22-kv-disk-tier/tp3-4k-b6144-m1m-kv35-nvme200.json \
  cluster start
```

The first command fetches the vLLM fork, applies the release's patches and the series, checks the tree
after each against its recorded hash, and compares the eight files with `overlay/`. With `RELEASE=1126`
in front of it, the command builds 1126 and the series from `vllm/series-1126`, checks the trees
`f3783b0e` and `e45d318c`, and compares `model_runner.py` with `overlay-1126/`. On a checkout of 1126, the
last command takes the profile `tp3-4k-b6144-m1m-kv35-nvme200-1126.json` instead. The CPU tests need
Python 3.12 with a CPU build of torch, the packages in the tree's `requirements/common.txt`, pytest, and
the config of `facebook/opt-125m` in the Hugging Face cache, because many of the tests build a model
config for that model with `HF_HUB_OFFLINE=1`. The last command uses `bin/spark`, the repository's
command-line tool for building the image and for syncing, starting, and stopping the nodes (AGENTS.md,
"Change discipline", and docs/upstreams.md). It prints a plan, and its `live:` and `shape:` lines show
what is running and which shape the profile resolves to; read both before adding `--apply`. The tier
needs 200 GiB of free space under `{checkout}/cache` on every node.

## CPU tests

Each suite ran on 2026-11-25, and on 2026-11-26 for the 1126 tree, with `VLLM_TARGET_DEVICE=cpu`, on an x86-64 machine with no GPU, with Python
3.12 and torch 2.13.0+cpu, and with one pytest command at a time, so that the timing tests of the series
did not share the CPU with another suite.

| Tree | Offload suite | `test_ds41_tier_sink.py` alone | Sink scheduler suite |
|---|---|---|---|
| the release's 702 patches (`8fc99ca7`) | 88 passed, 2 skipped | not present | 65 passed |
| the series (`47e196a2`) | 278 passed, 2 skipped | 58 passed | 65 passed |
| `candidate-73` and the 17 patches (`152445b9`) | 278 passed, 2 skipped | not run alone | 65 passed |
| `spark-ds41f-1126` and the 17 patches (`e45d318c`) | 278 passed, 2 skipped | 58 passed | 65 passed |

The offload suite is `tests/v1/simple_kv_offload` together with the connector's unit test, and the sink
scheduler suite is `tests/v1/core/test_sink_draft_zero.py`. The difference between 88 and 278 is the
tests that the series adds. The offload suite runs every file in `tests/v1/simple_kv_offload`, so its 278
include the 58 tests of `test_ds41_tier_sink.py`, which patch 9046 adds; 48 of those 58 drive sink
rejection through the core scheduler beside the tier, and the other 10 check the rule that allows the
connector. On `candidate-73` the 58 were counted only inside the offload run. The 2 skipped are two whole
modules, `test_worker.py` and `test_integration.py`, which pytest skips because they need CUDA. The suite
reads its O_DIRECT folder from `DS41_ODIRECT_DIR`, which must be on a filesystem that supports O_DIRECT,
such as ext4 or xfs. The test files come with the patches, and each patch header lists its tests and the
pass count of the offload suite after that patch.

## Hardware runs

### Method

The lab built its image of the release, `spark-ds41f:1120`, from the release's lock with `bin/spark build
prepare` and `build image`. The 200 GiB profile ran every check in the table below except the slot-reuse
check and the fault check, which ran on the 4 GiB profile. The tier file of each rank sat on the node's
root NVMe drive. Before the first start, a script compared the image's `model_runner.py` and `sink.py`
with the files of tree `8fc99ca7`, because the overlay replaces both. The checks ran during a planned
outage of the lab's serving, on an engine that served no other request. The outage began at 23:44Z on
2026-11-24. The 200 GiB profile started at 23:46Z, and its checks ran on that start; the 4 GiB profile
ran its checks on its first successful start. The 200 GiB profile then started again at 00:55Z to serve,
the lab's traffic returned to the engine at 00:59Z, and the outage ended at 01:00Z on 2026-11-25. The
recipe asks for a boot and a repeat boot (AGENTS.md, "Change discipline"), and it uses "boot" for one
start of the engine from an image (docs/release-benchmark.md). Each figure below was measured once, on
one boot. The only reading taken on the second start of the 200 GiB profile is one of the two
start-memory readings, and that start then served traffic. Where a check ran twice, the figures come from
the passing second call, and "Runs that failed" describes the first.

The check script is a Python client of the engine's HTTP API. It is not part of this pull request, and
the checks did not run through the recipe's own benchmark client, but the script can be added to this
folder on request.

The equivalence check and the slot-reuse check compared a disk hit with a pool hit at the same block
boundary. Each request went to `/v1/completions` with token ids, temperature 0, `seed` 1, `max_tokens`
512, `ignore_eos`, and `logprobs` 5. The script sent each prompt cold, then twice as a pool hit, and then
once more after other prompts had pushed it out of the GPU pool. On the 200 GiB profile the equivalence
check sent 8 prompts of 500,000 tokens for that purpose. On the 4 GiB profile, whose context is 262,144
tokens, the slot-reuse check and the fault check each sent 3 prompts of 200,000 tokens; these are the
script's defaults, and the recorded commands did not change them. Every prompt begins with its own head,
made of the script's process id, a counter, and a nanosecond time, so no two prompts share a block, apart
from the prompt that is built to share 60% of a stored session, and the first request of each prompt was
cold. The prefix-cache counters proved for each disk hit that its prefix came from the tier. In the
equivalence check the tolerance is the largest difference over all seven prompts, and it was 0, so the
two pool hits of each prompt had identical logprobs and any difference in a disk hit's logprobs would
have failed the check. The chat check and the slot-reuse check use the same comparison, but each
conversation or prompt has its own tolerance from its own pair of pool hits. Those pairs had equal tokens
and equal top-5 sets, because the check stops otherwise, but their tolerances were not recorded, so
whether their logprobs were identical is not known.

### Results

| Check | What it checks | Result |
|---|---|---|
| Start | each rank starts with the tier and with sink rejection on | passed at both starts of the 200 GiB profile |
| Memory at a start | lowest free memory of the node that runs rank 0, threshold 5,250 MiB | 5,787 and 5,936 MiB |
| Equivalence | disk hit against pool hit for seven prompts of 131,072 tokens | 7 of 7 equal |
| Shared prefix | a prompt that shares 60% of a stored session is a disk hit | passed |
| Chat | disk hit against pool hit through `/v1/chat/completions` | 3 of 3 equal |
| Memory under load | lowest free memory of the node that runs rank 0, threshold 4,200 MiB | 4,815, 4,815, and 4,633 MiB |
| Decode during a restore | decode rate while a 500,000-token session loads | passed, see the next table |
| Slot reuse | on the tier of 4 GiB a rank after it wrapped, a session stored afterwards loads correctly | passed |
| Fault | the tier file of one rank truncated to 0 bytes while the engine runs | passed |

The start check ran three times. It ran twice on the first start, because the script's second call, after
the hidden-code fix, ran the start check again before the equivalence and decode checks, and once on the
second start. Each time every rank logged its 936,228 slots and the sink line with rejection on, and each
tier file had 214.7 GB (200 GiB) reserved.

Four of the seven equivalence prompts were a stored prefix of 131,072 tokens followed by a new turn, a
short run of question text sent after the prefix, of 1, 64, 127, or 200 tokens. The script expected a
turn shorter than 128 tokens to hit only up to the block boundary before the 128-token replay window, and
the 200-token turn to hit the whole prefix, which is the boundary that patch 9032 sets; the lab's plan
notes that without 9032 the three short turns would have hit the whole prefix. The other three hid a code
in their filler text and ended with a question that asked for the code. The check passed on its second
call, with no load failure in 8 loads, one for each of the seven prompts and one for the shared-prefix
prompt, and 0.215 s as the slowest load. The shared-prefix prompt was sent with `max_tokens` 16 and was
not compared token by token; the check only proved that it was a disk hit, through its 78,848 cached
tokens and the hit counters. The chat check covered a restore while another request was decoding, a
conversation with an image, and a conversation with thinking on, and each of the three was a disk hit.

The memory checks read the lowest MemAvailable of the node that runs rank 0, once at each of the two
starts and three times under load; two of the readings under load happened to give the same value. Their
first thresholds, 5,632 MiB at a start and 4,915 MiB under load, came from readings taken on 2026-11-22.
On 2026-11-24 the base profile without any tier already read below them, at 5,403 MiB at a start and
4,511 MiB after 7.7 hours of traffic, so the thresholds were lowered to 5,250 and 4,200 MiB before the
run started. Readings vary from one start to the next, so these figures do not measure how much memory
the tier itself takes.

The slot-reuse check stored 39,118 blocks into 18,724 slots, then found the disk hit equal to the pool
hit, with a restore of about 0.2 s. In the fault check the request answered in 18.2 s, against 17.9 s
cold, with the tokens of the cold run, and the gauge `vllm:kv_offload_disk_degraded` went to 1. A second
session that had been stored before the fault then got no disk hit.

### Decoding during a restore

The decode check ran 2 and then 4 streams of 4,096-token prompts at temperature 0, with 600 tokens each,
and used a different 500,000-token session for each of the two runs. Each level had three runs, each with
fresh stream prompts. The first run, the baseline, had only the streams. In the second, when every stream
had produced its first token and 1 s more had passed, one 500,000-token session (1,959 slots) went in as
a disk hit, asking for one token. The third run, the control, sent the same session the same way with new
streams, and because the second run had put the session back in the GPU pool, the control's request was a
pool hit. The two sessions were the first two of the eight 500,000-token prompts that the equivalence
check had sent, one at a time, to push its own prompts out of the GPU pool. Their cold times come from
those requests, which ran while no other request was on the engine.

| Figure | 2 streams | 4 streams |
|---|---|---|
| restore time (gauge `vllm:kv_offload_disk_load_seconds_max`) | 1.034 s | 1.070 s |
| restore time for each slot | 0.53 ms | 0.55 ms |
| the whole request, restore included | 1.685 s | 1.739 s |
| the same request as a pool hit (control) | 0.618 s | 0.604 s |
| the same session cold | 62.4 s | 62.0 s |
| decode rate over the whole run, against the rate with no session sent | 0.93 | 0.92 |
| decode rate while the disk-hit request was in flight | 0.89 | 0.83 |
| decode rate during the extra time of the disk hit over the control | 1.04 | 0.94 |
| longest gap between two tokens of a stream, no session sent | 0.092 s | 0.104 s |
| longest gap between two tokens of a stream, with the disk hit | 0.369 s | 0.378 s |
| longest gap between two tokens of a stream, with the pool hit | 0.412 s | 0.380 s |

The check passed because the decode rate over the whole run stayed above 0.90. The time in which the
disk-hit request was in flight also holds the request's own admission and the prefill of its last tokens,
so the rate in that time does not measure the disk load alone, and the hypothesis and the criterion use
the rate over the whole run. The whole-run rate compares the second run with the baseline. The extra time
is the time by which the disk-hit request of the second run outlasted the pool-hit request of the
control, 1.067 s with 2 streams and 1.135 s with 4, and its rate counts the tokens that the streams
produced during the disk-hit request less those produced during the control's request. One run of each
gives no measure of the spread from run to run, so the ratio of 1.04 does not show that the disk load
speeds the streams up. The longest gap between two tokens was about the same with the disk hit and with
the pool hit, so the 500,000-token request itself causes the gap, whether its prefix comes from disk or
from the pool. The default slot budget of 0.02 s is about 36 times the measured restore time for each
slot.

### Sink rejection during the checks

Sink rejection was on in every check. The engine counted 38 draft-zero rejections during the checks on
the 200 GiB profile, every compared disk hit still equalled its pool hit, and no rank logged an
`AssertionError`, a `ValueError`, or a traceback. The repeated filler text of the test prompts most
likely triggered the detector. The counter does not say which requests the rejections hit, and no check
set out to force a rejection right after a disk hit.

### Traffic after the checks

After the checks the 200 GiB profile stayed on and served the lab's own client traffic, with the same
image and the same overlay. In the first 12 hours, a script that read the tier's counters every 30 s
(`runs/kv-soak-12h.tsv`) counted 125 disk loads, and the gauge put the slowest at 1.650 s. The counters
of load failures, I/O errors, and failed stores stayed at 0, and the tier was never degraded.

Over the same 12 hours the counter of stored blocks rose from 1,413 to 154,961. The counter counts a block
once, and every rank stores its own slot of 229,376 bytes for it, so each rank wrote 153,548 slots, about
35.2 GB (32.8 GiB), while the 125 loads read sessions back. Most of what the eager store writes is never
read, and the volume grows with the rate of new prompt tokens, so on a busy engine it is a cost in drive
wear and in write bandwidth. The setting `lazy_offload: true` stores a cached block only when it nears
eviction from the GPU pool, and it also turns off partial hash hits. Most of the series' tests use the
eager store, one parametrized test of 9046 also runs the lazy store, and the lazy store has not run on the
hardware, so the two have not been compared.

## Runs that failed

In an earlier outage on 2026-11-24, the 200 GiB profile with patches 9030 to 9045 did not start. Every
rank stopped at model load with "The sink detector and rejection (vllm.v1.sample.sink) do not support KV
transfer." That refusal led to patch 9046, and no check ran in that outage.

On 2026-11-25 the first call of the equivalence check failed on its hidden-code prompts, after five disk
hits had each equalled their pool hits. The check required the model to answer with the code, but the
script sends raw token ids without a chat template, so the model continued the filler text whether or not
the tier was on. The token comparison had already shown each disk hit equal to its pool hit. The check
was therefore changed to fail only when the disk hit and the pool hit disagree about the code. The check
script had skipped the decode check after the failed call, so the equivalence check and the decode check
then ran again, and both passed. The second call ran on the same engine start, so the tier still held the
first call's blocks, but every prompt of the second call had a new head, so none of them could hit those
blocks. The cold times of 62.4 s and 62.0 s for the two 500,000-token sessions also show that their first
requests found no cached prefix; the records hold nothing more on this.

The first start of the 4 GiB profile also failed. The engine refused a KV pool of 0.5 GiB for a context
of 262,144 tokens, which needs 0.53 GiB, so the pool went to 0.75 GiB, and the slot-reuse and fault
checks ran on that profile.

## Limits

Several restores at once, a queue of loads, and one load event that serves more than one request did not
happen on the hardware. A capacity check, in which 18 and then 40 stored sessions return and each return
must be a pool hit or a disk hit, would have covered them, but it was not run in this outage, and neither
was a check that hangs a read. No load overran its time bound on the hardware and no read hung there, so
the overrun path of patches 9041 to 9044 and the hang path have run only in the CPU tests. No check
forced a rejection right after a disk hit, no request was preempted on the hardware, and a tier of 200
GiB on each rank never filled up; slot reuse ran only on the 4 GiB profile.

The eager store, which writes each block to the tier as soon as it is computed, has an older behaviour
that the series does not change. After a preemption, the resumed request reaches the eager store as a new
request, and the function `_prepare_eager_store_specs` in `manager.py` appends the request's new block
table, its list of GPU blocks, to the old one. The store walks that list by block position, from its
count of stored blocks up to the number of blocks that the computed tokens fill, so after the append
those positions point at entries of the old table, and the blocks computed after the resume are never
stored. This mechanism comes from a reading of `_select_eager_blocks_to_store`. A CPU probe showed the
missing stores. The behaviour costs coverage of the tier, because a later return of that session computes
those blocks again, and by a reading of the code it serves no wrong content, because the blocks that are
stored hold the right tokens.

The tier is lost at every engine start, because the worker removes the file and the index lives only in
the scheduler on the node that runs rank 0. Only the shape `tp3-triangle-4k` ran on the hardware, with
the 200 GiB profile and the 4 GiB test profile, and the figures of each profile come from one boot. Other
shapes, pipeline parallelism, the `candidate-73` image, the 1126 image, and an image that holds the series
have not run.
The cost of the checksum on the ARM cores of a GB10 was not measured on its own, although it is included
in the restore times above.

## Sink refusal of KV transfer

While rejection is on, the function `_init_sink` in `model_runner.py` refuses any KV connector. The
commit of patch 0221
(`experiments/2026-10-19-sink-rejection-v2/vllm/0221-Sink-detector-and-draft-zero-rejection.patch`) says
that the runner "refuses rejection with the features it does not handle". The README of that experiment
lists KV transfer among the features refused at startup (lines 104 to 107) and adds "None of these is in
the recipe." No text gives a reason for the entry.

Patch 9046 rests on a reading of the connector's scheduler side. A load is asynchronous and reaches only
a request that is still waiting for remote KV, and such a request holds no provisional token. A failed
load is a cache miss for that request alone. A hit always leaves at least the last 128 prompt tokens to
compute on this model, and a store copies only committed tokens. If the refusal covers a known case in
which a connector breaks the held draft zero, that case is not tested here, and 9046 should change before
it is merged.
