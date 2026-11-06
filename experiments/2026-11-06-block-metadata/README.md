# Block-table update audit

Status, 2026-11-06: **component screen completed on `dgx1` across two fresh
bounded container boots ([`boot-1.json`](runs/27453e5b-dgx1/boot-1.json),
[`boot-2.json`](runs/27453e5b-dgx1/boot-2.json),
[`comparison.json`](runs/27453e5b-dgx1/comparison.json)) — do not promote or
include in Candidate `1107`.**

Producer-driven incremental page-table staging already covers the host
construction and upload opportunity. While the batch-order GPU gather rewrites
its configured capacity, the two-boot component screen on `dgx1` (`27453e5b`)
shows that adding per-CTA `(owner, epoch, count)` cache state and barrier
synchronization introduces a net negative/mixed tradeoff:
- **Correctness and bounds**: All 7 CPU contract tests (`bundle/test_host.py`)
  and all 55 GPU parity/transition gates (`bundle/test_gpu.py`) passed in both
  fresh bounded worker containers ([`steps.json`](runs/27453e5b-dgx1/steps.json),
  [`bounds.json`](runs/27453e5b-dgx1/bounds.json)).
- **Captured GPU gather (`gpu.steady` & `gpu.reorder`)**: The existing full-copy
  TileLang gather is already ~`1.86 µs` at `c1` and ~`8.0–9.3 µs` at `c16` across
  the 17-group runner geometry. While static-order `c16` steady gathers save
  ~`0.9–4.2 µs` at 512K–1M, `c1` steady gathers regress by up to **+1.25 µs**
  (`+64%` in `span` 512K `c1`: `+1.24 µs` in both boots; `span` 1M `c1`:
  `+1.15 µs` / `+1.20 µs`) or flip sign across boots, and **batch-reordered
  gathers (`gpu.reorder`) regress across all 45 cases in both boots (+0.27 to
  +2.01 µs)** because reordered rows incur both a full reset rewrite and per-CTA
  state check/barrier/update overhead.
- **Preparation (`preparation.*`)**: Uncaptured `page_crossing` and `overwrite`
  host-enqueue and completion 95% confidence intervals span zero in all 45 cases
  across both boots (`0/15` reproducibly faster in every mode).
- **Memory observations**: Peak two-arm GPU allocation is `29,930,496` bytes
  (~`28.54 MiB`) in both boots; the candidate adds `36,480` bytes of GPU cache
  state (`gather_state`) and `384` bytes of host UVA epoch buffers. Host peak
  RSS inside the bounded worker container was `1,501,952 KiB` (boot 1) and
  `1,500,288 KiB` (boot 2). Pre-boot `dgx1` `MemAvailable` was `10,101,504 KiB`
  (boot 1, `47 °C`, `0%` GPU util) and `10,098,400 KiB` (boot 2, `44 °C`, `0%`
  GPU util).

## Two-boot component screen summary (`runs/27453e5b-dgx1/`)

| Mode | Cases | `gpu.steady` both boots sig. faster | `gpu.steady` both boots sig. slower | `gpu.steady` sign-flip / inconclusive | `gpu.reorder` both boots sig. slower | `prep.page_crossing` completion both boots sig. faster |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `row` | 15 | 3 | 1 | 11 | **15 / 15** | 0 / 15 |
| `chunk` | 15 | 6 | 4 | 5 | **15 / 15** | 0 / 15 |
| `span` | 15 | 4 | 5 | 6 | **15 / 15** | 0 / 15 |

Selected 512K / 1M paired mean changes (`µs`, candidate minus control, `[95% CI]`):

| Mode / Case | `gpu.steady` boot 1 | `gpu.steady` boot 2 | `gpu.reorder` boot 1 | `gpu.reorder` boot 2 |
| --- | ---: | ---: | ---: | ---: |
| `row` 1M `c1` | −0.15 `[−0.16, −0.15]` | +1.17 `[+1.16, +1.17]` | +0.50 `[+0.50, +0.51]` | +1.69 `[+1.67, +1.70]` |
| `row` 1M `c16` | −4.22 `[−4.74, −3.70]` | −2.20 `[−2.67, −1.73]` | +0.61 `[+0.39, +0.83]` | +0.59 `[+0.25, +0.94]` |
| `chunk` 1M `c1` | −0.11 `[−0.11, −0.10]` | +1.14 `[+1.14, +1.15]` | +0.50 `[+0.50, +0.51]` | +1.72 `[+1.70, +1.74]` |
| `chunk` 1M `c16` | −2.61 `[−3.03, −2.19]` | −2.65 `[−3.00, −2.30]` | +1.24 `[+1.06, +1.41]` | +1.10 `[+0.91, +1.29]` |
| `span` 512K `c1` | +1.24 `[+1.23, +1.26]` | +1.24 `[+1.23, +1.25]` | +1.77 `[+1.76, +1.79]` | +1.75 `[+1.75, +1.76]` |
| `span` 1M `c1` | +1.15 `[+1.13, +1.17]` | +1.20 `[+1.19, +1.20]` | +1.70 `[+1.69, +1.70]` | +1.40 `[+0.78, +2.02]` |
| `span` 1M `c16` | −3.93 `[−4.43, −3.42]` | −0.94 `[−1.67, −0.21]` | +0.98 `[+0.66, +1.30]` | +0.78 `[+0.14, +1.43]` |

`bundle/incremental.py` remains a harness-only prototype and is not installed in
the serving runtime. Because `c1` and `reorder` gathers regress and `page_crossing`
preparation shows no significant gain, the prototype does not survive the
component screen and should not be integrated into vLLM or Candidate `1107`.

## Follow-up component screen, ready specification

- Published source code has been syntax-checked on SP8. Seven CPU tests in
  `bundle/test_host.py` pass there: three host-publication checks (appends leave
  replacement generations alone, same-size and empty replacements advance them,
  publication follows IDs/counts, and exhaustion fails before staging) and four
  worker-bound/ownership checks (rendered container commands enforce `--cpus=2`,
  `--memory=16g`, `--memory-swap=16g`, `--pids-limit=512`, read-only mounts, and
  compile/thread caps across both fresh boots; in-container cgroup v2 verification
  fails closed when `cpu.max`, `memory.max`, `memory.swap.max=0`, `pids.max` or
  thread caps are missing or exceeded; bounds are preserved and recorded across
  both fresh correctness/benchmark boots; and host launch requires exact window
  ownership and locked image identity). These are CPU checks, not GPU validation.
- `bundle/test_gpu.py` defines 55 GPU gates covering the five context labels at
  c1/c8/c16, integer IDs, expanded kernel blocks, holes, non-aligned widths, padded
  rows, compaction, preemption, same-size/shorter/empty reuse, alternate-output
  aliasing, layout rebuild, int32/int64 request mappings and captured consumers
  across shape changes. The source-file guard requires the audited control.
- `bundle/bench.py` compares the full gather with three reuse modes. Its
  17-group geometry follows the recorded runner benchmark; the receipt spells
  out live counts and does not claim an observed serving allocation. It records
  paired captured GPU times for steady/reordered batches and CPU enqueue plus
  completion time for steady, page-crossing and overwrite preparation. Each
  comparison retains all eight pairs and a 95% interval. Every host timing
  retires the prior step before UVA storage can be reused.
- `components.json` and `bundle/candidate.json` name one node, dgx1, under an
  exclusive coordinated window, with explicit container cgroup bounds (`--cpus=2`,
  `--memory=16g`, `--memory-swap=16g` for zero swap, `--pids-limit=512`) and
  compile-worker/thread limits (`MAX_JOBS=2`, `CMAKE_BUILD_PARALLEL_LEVEL=2`,
  `OMP_NUM_THREADS=2`, `TVM_NUM_THREADS=2`, `TORCHINDUCTOR_COMPILE_THREADS=2`,
  `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_MAX_THREADS=1`).
  The bundle uses the inspected candidate-40 image ID
  `sha256:7ae96e98e361601d8232cd99e6af1c6044b7c282bf6904c989e6ad033c147f82`,
  matching the locked source head and tree above. No released build is rerun.
- Estimated bounded window: **35–45 minutes**, including cooling/restoration.
  `bundle/run.py` renders and launches two fresh bounded worker containers (or
  verifies active `/sys/fs/cgroup/{cpu.max,memory.max,memory.swap.max,pids.max}`
  and thread caps inside the worker before every step across both boots, recording
  `bounds.json` and per-step bounds in `steps.json`). Each fresh benchmark process
  is preceded by a fresh GPU correctness process. Per round: host gate 30 s, GPU
  gates 300 s, benchmark 600 s maximum. It stops at the first failure, preserves
  partial receipts, and refuses to overwrite an existing output directory. This
  is two component rounds, not two serving boots.
- Before execution: wait for the named reservations and earlier legitimate
  claims; acquire an actual exclusive lease; inspect the current serving commit,
  profile and all-rank image identity; publish/sync the exact experiment revision
  with the explicit configuration; inspect `live:` and `shape:`; use normal
  memory/thermal/watchdog guards and restore the recorded original state. No
  request file has been written and no GPU access is scheduled by these files.
- A surviving mode still needs an upstream runtime patch with prepared-kernel
  declarations and all destination writers accounted for, followed by native
  output/batch/cross-rank gates and full-step/end-to-end serving measurements
  over a boot and repeat boot. A component win cannot make this work ready.

Remote scratch: `/home/chris/spark-work/block-metadata/` on SP8, with a clean
detached published deployment checkout. CPU receipts remain under its `results/`
directory. No model, image, or prepared-source copy was created locally.

This is a source audit and an offline operation-count screen. It is not the
32K–1M c1/c8/c16 serving-hardware comparison that a new candidate would require.
In particular, neither local integer equality nor a historical microbenchmark
establishes native-output equivalence, cross-rank agreement, current full-step
latency, throughput, or memory preservation for a proposed cache.

## Identity and scope

- Deployment base: `eb8a49f6d92231acaaf3f9fd63a4b7d3a4183854`.
- Isolated branch: `feat/incremental-block-metadata`; the shared checkout was
  not edited or switched.
- Promoted baseline: `2026-10-30-spark-ds41f-1030`, vLLM patch head
  `b6de07aad5cb21f2cf8381036c1a9eaf471b07fc`, tree
  `ea1281f880e7535e3d87725be0156a8e38fc2fb2`.
- Audited unreleased recipe: candidate 40's `tp3.json` and `upstreams.lock.json`.
  Base vLLM `19f2c20ed4d64ae22bd009ceacadd85e2a81e354`; all 574 patches
  freshly applied to reproduce head `5f926df6d3126266a9c88c82c06544060008c33c`
  and tree `ca1496c6481dc5bc4b4234b743e4ca023a6c7597`.
- Intended variable screened: how an unchanged-address page table is updated
  (full destination copy, snapshot changed spans, coalesced spans, or existing
  producer-supplied deltas). No execution extent or eviction rule was changed.

[Source hashes and recorded costs](runs/source-and-recorded-costs.json) identify
the audited functions. The append, staged-apply, gather, gather-kernel,
gather-chunk and stage-write ASTs are identical between the promoted source and
candidate 40. This is not a claim that all their source or runtime behavior is
identical. [Fresh series receipt](runs/fresh-series.log).

## Existing paths

Paths and line numbers below refer to the audited vLLM head.

| Stage | Implementation | Work and lifetime |
| --- | --- | --- |
| Scheduler | `vllm/v1/core/sched/scheduler.py:2998` | Supplies newly allocated IDs for continuing requests; full replacement is explicit. |
| Target request table | `vllm/v1/worker/gpu/block_table.py:152` | Appends supplied IDs at the previous count, or overwrites from zero. Expands IDs only when a KV block spans multiple kernel blocks. Does not compare old and new context rows. |
| Host staging and upload | `vllm/v1/worker/gpu/buffer_utils.py:178` | Empty writes stage nothing. Payload and descriptors describe changed spans. Multiple groups are applied in one fused launch, not one launch per span. |
| Counts publication | `vllm/v1/worker/gpu/block_table.py:174` | Publishes the small group-by-request-slot count array after staging. This remains per-step work even when page-ID payload is empty. |
| Batch gather | `vllm/v1/worker/gpu/block_table.py:188`, `tilelang_inputs.py:124` | One launch copies request-slot rows into batch order, loads live IDs, zeros the tail and padded rows, and writes the full configured destination width. Persistent destination addresses. |
| Target metadata | `vllm/models/deepseek_v4_1/tilelang/metadata.py:258` | Fused window/decode metadata writes live main pages; page conversion is already bounded by visible pages. Token positions and window metadata still change inside a page. |
| Layer sharing | `vllm/models/deepseek_v4_1/attention.py:1545` | Shares decode metadata for matching groups within a forward. Graph-pool outputs must be rewritten on replay; this is deliberately not persistent cross-forward reuse. |
| Drafter | `vllm/v1/worker/gpu/spec_decode/speculator.py:369` | Uses slices of the target's input block tables. Normal DSpark/DFlash does not separately upload a full context table. The additional DFlash gather is in the PCP branch. |

The target's request-update, staged-apply and metadata preparation order in
`model_runner.py` provides the publication sequence. The UVA buffer pool retains
in-flight host buffers until reuse is safe. A new strategy cannot publish counts
before corresponding IDs become visible to their consumers.

An allocation-free step and a circular-window wrap have no new page-ID payload.
A page crossing appends IDs. This does not mean all metadata is unchanged:
positions, sequence lengths, allowed reads, window parts and slot mappings advance
independently. Block ID zero and unwritten tails retain their current treatment.

## Alternatives and invalidation

1. **Full host snapshot upload:** introduces context-sized host construction and
   transfer where continuing requests currently send explicit deltas. It is not
   an improvement over the existing staging path.
2. **Snapshot changed spans:** avoids some payload but scans the snapshot,
   retains previous contents, and creates descriptors. A stable decode step is
   particularly unfavorable: the existing producer has no page-ID work, whereas
   a snapshot comparison still scans the context.
3. **Coalesced snapshot spans:** reduces descriptors by retransmitting unchanged
   gaps. The current fused writer already applies all groups/spans in one launch,
   so fewer spans do not automatically mean fewer launches.
4. **Producer-driven incremental GPU destination:** could avoid some gather
   stores without a host scan, but requires destination ownership and generation
   tracking, not just page counts. It was not implemented or GPU timed during
   the initial audit. The follow-up component prototype above tests this idea;
   no timing or correctness result is available yet.

Same-size slot reuse replaces pages without changing counts. Batch compaction
changes the source slot for a destination row without changing the request's
page list. Shorter reuse requires zeroing the old tail. Preemption/resumption,
padded rows, dummy runs that zero input tables, alternate `out` destinations and
graph-buffer reuse also invalidate cached destination state. A correct candidate
would preserve publication ordering and addresses while tracking these events.
These are requirements for a new cache, not defects found in the current gather.

The related changes are already tracked separately: drafter metadata graph
placement ([deployment PR 919](https://github.com/christopherowen/spark-ds41f/pull/919)),
verification shapes ([PR 921](https://github.com/christopherowen/spark-ds41f/pull/921)),
and prefix endpoint retention ([PR 923](https://github.com/christopherowen/spark-ds41f/pull/923)).
This audit does not duplicate their changes or infer their pending validation.

## Offline operation screen

`screen.py` executes the pinned `BlockTables.append_block_ids` and
`StagedWriteTensor.stage_write` methods, extracted without importing CUDA. Device
application and gather are Python reference operations. It checks full-copy,
changed-span and coalesced-span reconstruction exactly, with coalescing gaps of
16 and 64 words. It does not execute the GPU kernel or simulate HTTP serving.

Geometry: one 256-token paged group with 4096 configured columns, plus a one-ID
circular group. Context labels are 32K, 64K, 256K, 512K and 1M; concurrency is
1, 8 and 16. For the 1M label the initial state is one page short, so the crossing
reaches exactly 1M. These are metadata arrays, not a claim that c16 at 1M fits the
serving cluster. The fixture is not the runtime's complete group layout.

Ten transitions cover new requests, growth inside a page, page crossing, circular
wrap, reversed compacted batches, sparse overwrite, preemption, resumed pages,
same-size slot reuse and shorter reuse. Sparse overwrite is API stress; this
audit does not establish that the active scheduler emits that pattern.

Result: **150 cases, 7,500 exact row comparisons, zero failures**.
[Full receipt](runs/operation-screen.json).

For the 1M/c16 fixture, the full gathered destination is 262,208 bytes. Every
snapshot strategy scans 65,552 words per event. Selected operation counts:

| Event | Existing producer payload / spans | Exact destination changes / spans | Coalesce 16-word gaps: destination bytes / spans |
| --- | ---: | ---: | ---: |
| Growth inside page | 0 / 0 | 0 / 0 | 0 / 0 |
| Page crossing | 64 / 16 | 64 / 16 | 64 / 16 |
| Circular wrap | 0 / 0 | 0 / 0 | 0 / 0 |
| Batch reorder | 0 / 0 | 262,208 / 32 | 262,208 / 32 |
| Sparse overwrite | 262,208 / 32 | 15,424 / 3,856 | 261,184 / 16 |
| Same-size slot reuse | 262,144 / 32 | 262,144 / 32 | 262,144 / 32 |
| Shorter reuse | 128 / 32 | 262,144 / 32 | 262,144 / 32 |

These columns measure different boundaries: producer payload updates the
request-slot source; destination changes describe the gathered batch output.
They are not measured transfer savings or latency comparisons. The current GPU
gather still runs after source updates. The sparse example illustrates why fewer
bytes can mean many more descriptors, and why coalescing can erase byte savings.

Both local runs are retained. V1 left two pages free at the largest context; V2
corrected the fixture to reach the exact 1M boundary. Both pass. V1: 0.57 s elapsed,
31,047,680-byte peak RSS; V2: 0.58 s, 31,326,208 bytes; zero swaps in both.
These are whole-script Mac resource observations, not host-algorithm timings,
GPU memory observations, or performance confidence intervals.

## Existing hardware evidence, not a new before/after

The runner receipt is from `tilelang-integration-kernels-26`, dgx1, documented
source `26382f5cf` in the [migration record](../2026-10-24-tilelang-completion/README.md).
The attention receipt is `audit-attention-6`, dgx4, documented source `fb73c80cf`
in the [kernel audit](../2026-10-25-kernel-audit/README.md).
They preserve their original benchmark fixtures and do not represent new runs
of the promoted release or candidate 40.

| Already-recorded operation | Earlier median µs | Shipped median µs | Median change | Paired mean change µs [95% interval] |
| --- | ---: | ---: | ---: | ---: |
| Batch gather, c1: Triton → TileLang | 5.198 | 2.074 | −60.1% | −3.077 [−3.219, −2.934] |
| Batch gather, c16: Triton → TileLang | 10.731 | 8.755 | −18.4% | −2.145 [−2.608, −1.682] |
| Decode metadata, page 256, c1/64K: every column → live pages | 3.262 | 1.687 | −48.3% | −1.591 [−1.662, −1.519] |
| Decode metadata, page 256, c1/1M | 3.128 | 2.324 | −25.7% | −0.865 [−0.915, −0.814] |
| Decode metadata, page 256, c16/64K | 4.037 | 1.711 | −57.6% | −2.384 [−2.576, −2.191] |

The percentages compare the displayed medians; the interval belongs to the
original paired mean difference, which need not equal their subtraction.
[Runner raw receipt](runs/recorded-integration-runner-26.log),
[attention raw receipt](runs/recorded-audit-attention-6.log). The complete receipts
also retain insignificant results, including the page-table-only c1/64K interval
spanning zero; that result does not demonstrate no regression.

The historical `request-inventory-5/ports` serving summaries contain 13 gathers
per rank over approximately 535 ms: 65.8–67.5 µs in total, 5.06–5.19 µs/call,
0.0123–0.0126% of the traced window. All four ranks are recorded in the extracted
receipt with hashes of the input summaries. These are aggregate profiled times;
the summaries do not provide per-call samples for a confidence interval.
The older Triton `decode-kernels-next-2.json` reports roughly 5.48 µs at c1 and
30.80 µs at c16, but its implementation differs and it must not be represented
as the current gather cost.

The later kernel fixture narrows the opportunity to a few microseconds at c1
and under 9 µs at c16 in that measurement. Removing all of it would still be
only a component ceiling for those cases. Current long-context serving exposure,
CPU cost of a proposed strategy, c8, the full context/concurrency matrix, and
boot-to-boot memory remain unmeasured. The main page conversion and visible-token
metadata also have costs that cannot be removed merely by skipping a gather.

## Reproduction and checks

In the isolated deployment checkout, after preparing the exact candidate-40 tree:

```sh
/usr/bin/time -l python3 experiments/2026-11-06-block-metadata/screen.py \
  --source .work/vllm \
  --output experiments/2026-11-06-block-metadata/runs/operation-screen.json
python3 experiments/2026-11-06-block-metadata/audit.py \
  --source .work/vllm \
  --recorded-root /Users/christopherowen/projects/spark-ds41f \
  --output experiments/2026-11-06-block-metadata/runs/source-and-recorded-costs.json
bin/spark doctor
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-40/tp3.json doctor
git diff --check
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-40/tp3.json build check
```

The fresh series was applied sequentially from candidate 40's `vllm/series`
using `git am --committer-date-is-author-date` from its locked base, on the named
`metadata-base` branch. The matching author/committer identity reproduced the exact
head and tree. The receipt retains the pre-existing whitespace warning from a
patch; there were no application failures or source edits.
[Preparation and receipt-retrieval commands](runs/commands.txt).

Both doctor checks pass ([check receipt](runs/checks.json)). Build check reports missing prepared build inputs at
`.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-a35ced21`; no image was built.
The initial environment checks found no NumPy on the local host or sp8. The
screen uses only the standard library; no dependency was installed. No GPU
experiment, cluster operation, serving request, or node mutation was performed.
An existing cluster holder and a pending request were observed and respected.

There is no old-fails/new-passes runtime reproducer and no new candidate
performance or preserved-performance evidence. A persistent GPU cache should be
reopened only with a demonstrated exposed cost and a design that survives the
listed invalidations, then tested on serving hardware with two boots and the
normal correctness, native-output, all-rank, memory and restoration gates.

## Local resource cleanup

At the owner's request, the clean disposable `.work/vllm` prepared tree was
removed after checking its status and locked identity. The isolated deployment
checkout now sparsely contains this experiment. Its allocated footprint fell
from 536,304 KiB to 636 KiB (548,524,032 bytes reclaimed across the two operations).
No source, evidence or unfinished changes were removed; no shared object store
or other task's checkout was touched. The audit's prepared source can be
reproduced from its pins and series on a remote host. Further compilation,
dependencies, tests and bulk artifacts belong on the DGX/Spaceports; local work
is limited to lightweight edits and Git. [Cleanup receipt](runs/local-cleanup-20261106.json).
