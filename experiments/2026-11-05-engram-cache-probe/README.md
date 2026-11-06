# Engram row-cache locality

Status: unpromoted source experiments. Thirteen native planner designs have been
screened; two cache-metadata layouts are awaiting exact-source hardware checks.
There is no measured serving improvement, correctness repair, or promotion.
The initial wider-probe work is retained in [bounded-probe.md](bounded-probe.md).

## Scope and identities

The deployment base is `2c4b23afcd834ba2adf6a15ca434e59fa4f49f74`.
Every source variant starts from candidate 38's ordered vLLM head
`1f33102547e4da5c64712390604695d1abd577c4` (tree
`c061e5f6931306fa4655ba531428f639c365bf10`), after canonical upstream
`19f2c20ed4d64ae22bd009ceacadd85e2a81e354`. Each final layout changes only
`engram_rows/_row_plan.h`; arithmetic, row bytes, capacity, admission, replacement,
probe limits, coalescing and worker scheduling remain the same.

| Layout | Source head | Source tree | Default cache allocation per reader |
| --- | --- | --- | ---: |
| Existing interleaved layout | `1f33102547e4da5c64712390604695d1abd577c4` | `c061e5f6931306fa4655ba531428f639c365bf10` | 17.125 MiB |
| Separate IDs, aligned payloads | `745da6586fdebf8d48133639002ed41858831e13` | `98ea1c85791a093f9a95179a4dd13dee9415fc52` | 17.5 MiB |
| Compact per-set miss filter | `e6fe9fb614d1b6488bb1310044a408822feaa6f6` | `d55e502217ea8b8752f999c47cb546dc5c98cd3c` | 17.25 MiB |

The aligned layout keeps exact IDs in a compact array and aligns payload sets to
64 bytes. The filter keeps the original layout and adds one 64-bit summary per
set. Every inserted ID is represented; a collision falls through to exact ID
comparison. It does not approximate a hit or change the row chosen.

The `aligned/` and `filter/` directories own separate ordered patch series, locks,
source manifests and TP3/TP4 overlays. Their deployment branch is
`feat/engram-cache-probe-20261105`; their image tags are reservations until a build
receipt says otherwise. The older top-level profiles retain the initial 128-row
probe hypothesis. No promoted configuration or immutable baseline is changed.

## Search and measurements

The finite screen compares the original 64-row probe limit, 96/128-row limits,
separate IDs, aligned payloads, a miss filter, a hit-rate-adaptive filter, and
separate weight/scale planes. Layout variants at both 64 and 96 rows explicitly
test the interaction with probe coverage. The final candidates keep the original
64-row policy so their layout effect is isolated.

The production-flag screens use GCC `-O2`, matching `_native.py`, and actual
`_row_plan.h` code. They compare the same seeded IDs in forward and reverse process
orders. The earlier Clang `-O3` screen is preliminary and retained separately.
The final 72-case screens include TP3 and TP4, 6/8/48/96 token rows, requested hot
fractions of 0/50/75/100%, 1,024/16,384 seeded rows per column, empty cache,
65/129-row boundaries, and 6,144-row chunks. Large warm pools exceed the cache's
capacity, so requested hot fractions are not measured hit rates. Actual cache
hits, misses, planned bytes, jobs, process RSS and every sample are retained.

The extended fixture independently encodes the full row ID in both payload
planes. The original preliminary fixture's simple byte pattern could alias IDs
modulo 256; its receipts remain, but its oracle is not claimed to exclude that
class of error. All extended screens preserve exact byte, admission and job-count
checks. The final-source sanitizer run additionally checks the cleaned patch
headers rather than the generated search headers.

Illustrative mean planner times in microseconds, TP3, 1,024 seeded rows per column:

| Token rows / requested hot IDs | Original A/B | Aligned A/B | Filter A/B |
| --- | ---: | ---: | ---: |
| 6 / 0% | 4.40 / 4.39 | 3.13 / 2.75 | 2.91 / 2.71 |
| 6 / 100% | 1.68 / 1.60 | 1.43 / 1.39 | 1.59 / 1.57 |
| 48 / 0% | 38.32 / 38.85 | 29.70 / 27.77 | 26.69 / 26.04 |
| 48 / 75% | 18.34 / 18.14 | 17.32 / 17.48 | 18.11 / 17.80 |
| 48 / 100% | 11.09 / 11.50 | 10.26 / 10.06 | 11.37 / 11.17 |

`sweep-results.json` contains matched 95% Student-t intervals over distinct seeded
batches. These are CPU fixture intervals, not independent serving boots. They do
not establish GPU wait or request throughput. The filter favors cold lookups;
aligned payloads are more consistent on warm and partially warm batches. The
unaligned split, adaptive filter, and separate weight/scale planes did not justify
their additional tradeoffs. No global optimum is claimed.

Wider probing has a separate limit: TP3's eight streams with five speculative
drafts use 48 verification token rows, already covered by the original policy.
TP4's 16 streams reach 96 token rows. The original 128-row probe candidate therefore
does not target full TP3 verification. Broad long-prefill probing was rejected:
coalescing already removed most repeated pages, while probing added CPU work.

## Exact-source correctness checks

Both final layouts pass 36 native CPU reader cases each, with real O_DIRECT
reads, early-staging overlap, and short-read/missing-source/timeout failure
checks. The exact final headers also pass 152 combined ASan/UBSan planner
cases. Their complete ordered patch stacks apply freshly to the recorded
heads/trees, and TP3/TP4 static doctor checks pass. These are not live topology
or GPU qualifications. Complete image inputs and image tests are still pending.

## Remaining bottleneck and hardware plan

The pinned model snapshot `dba1be0a40aa45a94ad051997016db3960a90277` has Engram
layers `[1, 14]`. Recorded C38 default TP3 steps are 19.536/19.428 ms for 1x6
verification and 28.436/28.456 ms for 8x48 verification. Simply adding the two
largest observed CPU planner savings gives only about 0.1% of the latter step.
This is an optimistic screening estimate, not a bound on CUDA-mapped memory or
an end-to-end performance claim. Actual exposed I/O and GPU wait must decide
whether either layout warrants serving qualification.

`gpu-screen.py` is an unreleased component fixture for that decision. It uses two
readers, the exact native source snapshots, real direct reads, CUDA write-combined
mapped destinations and the production GPU gather. It records issued reads,
cache hits/misses, planning/publication times and GPU waits, with bitwise BF16
checks. Cases include single-stream/full TP3 verification, full TP4 verification,
and 129/6,144-row chunks with both early and GPU-provided IDs. Changing the native
source in a fixed environment is component evidence; it is not a qualified image
or a new measurement of a released serving build.

If the affected-path result justifies it, the selected source needs an image
built from complete verified inputs, all native Engram gates, then a boot and
repeat boot with the recorded C38 workload identity: distinct traffic at C1/C8/C16,
32K through 1M cold prefill, correctness/batch invariance, memory, failure and
cooling checks. Compare compatible recorded runs; never rerun a released serving
build to manufacture a control. Preserve adverse outcomes and intervals. A
confidence interval spanning zero alone does not demonstrate preservation.

Packed-row reader helpers are already present, but the model loader still
registers original planes. This experiment does not enable packing or claim its
benefits. Candidate 38 already coalesces direct reads and uses one layer-order
worker with early staging; no new scheduling policy is warranted by these CPU
results.

## Evidence and resource record

All heavy work after the workstation interruption runs remotely on `sp8`.
Measured CPU screens used core 19 (2 MiB L2), one CPU and 1 GiB, with isolated
output directories. The final correctness-only reader/sanitizer checks can run
alongside another bounded build, still capped at one CPU and 1 GiB; their timings
are not performance evidence. Source preparation is capped at one CPU and 2 GiB.
Systemd's implausibly small memory peaks are not used; C/Python receipts record
process `getrusage` RSS. No GPU tests or serving changes are implied by CPU checks.

Compressed raw receipts are byte-preserving gzip archives; `evidence-index.json`
records compressed and original SHA-256 hashes. Inputs include the exact generated
headers, fixture source, commands and all output/errors. Historical failures,
including the workstation interruption, missing Clang sanitizer runtime, initial
configuration mismatches and the build-lock refusal, remain in the ledger.

Hardware operations are authorized, but require a legitimate exclusive cluster
handoff. Existing holds and requests are preserved. Branch testing uses exact
published commits and coordinated node operations; the main-only lab queue guard
is not weakened. No merge or promotion is authorized.
