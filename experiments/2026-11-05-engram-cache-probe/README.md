# Bounded Engram cache probes

Status: unbuilt, unpromoted source experiment. The CPU planner checks pass;
CUDA, physical I/O, output/quality gates and serving boot/repeat qualification
are pending. There is no measured serving improvement and no release comparison.

## Identity and one variable

- Deployment base: `2c4b23afcd834ba2adf6a15ca434e59fa4f49f74`.
- Canonical vLLM base: `19f2c20ed4d64ae22bd009ceacadd85e2a81e354`.
- Candidate-38 ordered source head: `1f33102547e4da5c64712390604695d1abd577c4`,
  tree `c061e5f6931306fa4655ba531428f639c365bf10`.
- Experimental source head: `f3c6d0fe262f1ed061d825d6cc3b851d4760a912`,
  tree `d2f16b4f0697452766593281f7412b5f72a59073`.
- Variable: allow direct-to-mapped steps of 65–128 rows to probe already
  populated Engram cache entries. Keep the existing admission limit, cache
  capacity, immutable bytes, row ordering, arithmetic and worker behavior.

The source stack and overlays are confined to this directory. `tp3.json` and
`tp4.json` inherit their candidate-38 profiles. Their image tag is a reservation;
no image with these inputs has been built or distributed. No promoted recipe,
release baseline, existing benchmark, node checkout or service was changed.

## Source findings

Candidate 38 already has asynchronous direct I/O, sorted/coalesced page reads,
one layer-order worker queue, host-ID read-ahead, cached early staging and a
four-way row cache. Adding another worker policy is not justified by this screen.

Its direct-to-mapped path admits and probes cache rows only through 64 rows.
Cached early staging has a separate existing exception. Thus a direct step of
96 rows, the TP4 maximum sequence count, can bypass already cached rows. This
patch separates probing from admission and caps the additional probe range.
Ordinary direct steps through 64 rows and above 128 rows retain their policy.

Packed-row helpers and native tests also exist, using 15 rows of 264 bytes per
4 KiB page. However, `common/engram.py:open_disk_rows` registers the original
disk sources; it does not call `pack_engram_columns` or select packed files.
Reader support alone is not evidence that serving uses packed layout. Packing
is a separate variable and is not enabled here.

## Fixture and measurement boundary

`probe.c` includes the actual source `_row_plan.h`. It creates no files and
performs no disk, CUDA, io_uring or serving operation. "Planned jobs" counts the
native planner's coalesced requests, not issued I/O, device completions or GPU
wait. The original header is a CPU fixture control, not a repeated benchmark
of a released build.

The fixture uses 24 columns, rank-1 TP3/TP4 masks, 16 million rows per column,
where each token row contributes 24 lookups before masking. Batch row counts
in this document mean token rows, rather than individual column lookups. It uses
unaligned original weight/scale offsets of 4093/123, 256/8 bytes per row, and
the production default cache capacity of 65,536 rows. A fixed xorshift seed
generates identical IDs for each header. Warm trials seed 1,024 IDs per owned
column through the real cache insertion code. Hot percentages are requested
ID distributions, not assumed hit rates; set conflicts can leave misses.
Every 97th lookup is dead. The byte oracle independently checks all destination
bytes, untouched unowned/dead slots, unchanged cache storage, no admission of
large-step misses, and rejection of a missing source on an uncached row.

Each final process covers 26 paired fixtures: TP3/TP4 crossed with 16, 64, 65,
96, 128, 129, 512, 6,144 and 16,384 rows, including fully hot, 75%-hot, all-miss
and empty-cache cases. Each optimized fixture has 11 steps; summaries omit
only the first warmup. All samples are retained. The two processes replay
identical IDs. Their intervals describe ten synthetic ID batches within each
process, not independent production boots or real request traffic.

## Results

Both final CPU processes produced the same job counts. The table gives mean
jobs per step; 95% Student-t intervals are for paired jobs saved across the
ten non-warmup batches. Full timing intervals and memory records are in
`results.json`; raw commands, standard output, errors and every sample are in
`runs/*/receipt.json`.

| 96-row fixture | Control jobs | Candidate jobs | Reduction | Jobs saved, 95% interval |
| --- | ---: | ---: | ---: | ---: |
| TP3, 75% warm IDs | 492.6 | 375.5 | 23.77% | 117.1 [115.73, 118.47] |
| TP4, 75% warm IDs | 359.1 | 272.4 | 24.14% | 86.7 [81.97, 91.43] |
| TP3, all warm IDs | 96.5 | 0 | 100% | 96.5 [93.24, 99.76] |
| TP4, all warm IDs | 71.0 | 0.4 | 99.44% | 70.6 [68.33, 72.87] |
| TP3, all misses | 1517.0 | 1517.0 | 0% | 0 [0, 0] |
| TP4, all misses | 1137.6 | 1137.6 | 0% | 0 [0, 0] |

The CPU cost is material. At 96 rows with 75% warm IDs, extra planning time
was 43.28/8.99 microseconds in TP3 process A/B and 25.05/9.38 microseconds in
TP4 A/B. All-miss extra time was 36.90/19.09 microseconds for TP3 and
24.61/15.89 microseconds for TP4. Even the fully warm planner was slower in
process A and faster in B. This drift prevents a CPU speedup claim. The
tradeoff only helps serving if actual avoided I/O outweighs probing and
copying under the real warm-row distribution and worker schedule.

All long direct-step fixtures retained identical planned job counts. The
rejected unbounded variant saved almost no planned I/O in the long 75%-hot
fixture because the planner had already coalesced repeated hot pages, while
adding CPU work on misses. Its runs are retained rather than substituted by
the bounded result.

Cache allocation is exactly 17,956,864 bytes (17.125 MiB) per reader in both
headers, including set metadata and padding. The fixture's peak process RSS
was about 36 MiB for short steps, and up to about 158 MiB for the largest
TP3 fixture. This includes fixture destinations, metadata and cache snapshots;
it is not model or serving memory. There is no claimed allocation reduction,
device-memory reduction or extra cache. Systemd's implausibly small reported
memory peaks are not used; the fixture records `getrusage` RSS. GPU and CPU
memory remain shared physical memory on the target hardware.

## Run ledger and failures

| Record | Outcome |
| --- | --- |
| `runs/local-interruption.json` | Workstation crash interrupted the initial local attempt. No complete measurement receipt and no accepted local timing. No surviving local jobs after recovery. |
| Initial remote launch | Failed before compilation because the service working directory was omitted; Python looked for `/home/chris/run.py`, exit 2. Retried with an explicit working directory. |
| `sp8-a`, `sp8-b` | Two complete broad-probe screens, 18 paired fixtures each; rejected variant preserved in `runs/inputs/rejected-unbounded.diff`. |
| `bounded-a` | Initial bounded screen, 26 paired fixtures; not CPU-pinned. Retained as preliminary. |
| `pinned-a`, `pinned-b` | Final bounded screen and repeat, 26 paired fixtures each, CPU 19; byte, cache and missing-source assertions pass. |
| `sanitizers` | Clang compilation failed because its AArch64 ASan runtime libraries were absent. Exact error retained. No packages installed. |
| `gcc-sanitizers` | GCC 13.3 ASan + UBSan, 26 paired fixtures with two steps each; all pass. |
| `runs/setup/doctor-initial.log` | Spaceport's switched node map did not match the inherited cabled topology. Isolated ignored fixture maps were replaced with the tracked examples. |
| `runs/setup/lock.log` | Candidate series locked and freshly applied successfully; candidate doctor then caught the example head address differing from the inherited master address. |
| `runs/setup/doctor-tp3.log`, `doctor-tp4.log` | Pass after aligning only the isolated ignored fixture maps' head address to the inherited profile. Configuration validation, not live hardware qualification. |
| Deployment commit first attempt | Git index preloading exceeded the service's 16-task cap (`unable to create threaded lstat`); failed before staging or committing. Retry disables index preloading and uses one index thread, retaining the resource limits. |
| Publication export first attempt | The remote commit succeeded, but its metadata export used an invalid root-tree path (`HEAD^:.`). Corrected to the parent commit's tree expression; no source change or measurement rerun. |

The fresh historical source stack emits an inherited blank-line-at-EOF warning
while applying an earlier patch. Its final tree passes the source diff check;
the new two-file patch passes its own diff check and reproduces the recorded
head/tree. Complete non-vLLM build inputs are absent in this isolated checkout,
so `build check`, image smoke tests and full build validation have not passed.

The pre-RSS preliminary probe revision was not archived byte for byte; its
receipt records the hash, and the earlier runners and broad header diff are
retained. Reproduction uses the final committed probe. Do not present the
preliminary timings as fully reproducible final evidence.

## Reproduction and resource limits

After the workstation interruption, compilation, source preparation, checks,
analysis and Git commits ran on `sp8` over SSH under
`/home/chris/projects/engram-cache-probe`. CPU probes used one CPU, a 1 GiB
memory cap, eight tasks, a 240-second timeout and the shared build lock.
Source preparation used one CPU, 2 GiB and 32 tasks, with one build thread
and Git packing thread. No local fallback, GPU work, node synchronization,
cluster operation, hold replacement or queued-request replacement occurred.

The exact native compile and execution arguments are in each receipt. This
wrapper reproduces the final resource limits with the final committed runner
and a fresh output directory:

```sh
systemd-run --user --unit=engram-reproduce-a --collect --wait --pipe \
  --working-directory=/home/chris/projects/engram-cache-probe \
  -p CPUQuota=100% -p MemoryMax=1G -p TasksMax=8 \
  /usr/bin/timeout 240 /usr/bin/flock -n /tmp/spark-build.lock \
  /usr/bin/taskset -c 19 /usr/bin/python3 \
  deployment/experiments/2026-11-05-engram-cache-probe/run.py \
  --base base --candidate bounded --out runs/reproduce-a
```

The repeat used a distinct unit and `runs/pinned-b`. The sanitizer run used
`run-sanitizers.py --base base --candidate bounded --out runs/gcc-sanitizers
--sanitize --cc cc` under the same limits. `run.py` here is that final runner.
Header directories must contain the original/candidate `_row_plan.h`; their
SHA-256 fingerprints are recorded in the receipts. The source commits above
identify both versions without relying on running containers or image tags.
Run `python3 analyze.py` remotely to regenerate `results.json` from the two
final receipts. `runs/inputs/` records the source preparation/locking scripts.

Fresh application and static configuration checks:

```sh
bin/spark --cluster-config experiments/2026-11-05-engram-cache-probe/tp3.json \
  build prepare --lock --only vllm
bin/spark --cluster-config experiments/2026-11-05-engram-cache-probe/tp3.json doctor
bin/spark --cluster-config experiments/2026-11-05-engram-cache-probe/tp4.json doctor
git diff --check
```

## Qualification still required

The patch includes 12 CUDA regression cases at 65, 96 and 128 rows, with full
and TP3 column masks and both host-known and GPU-provided IDs. They compare
cached/uncached output bits, check the mathematical reference, exercise dead
IDs and verify that repeated large misses do not evict the small warm set.
Python syntax passed; these GPU cases have not run.

Once an authorized shared window is available, first build/verify the image
and run the native Engram test file, including reader failure/short-read,
timeout, early-staging, packed/original layout and cache cases. Then qualify
the candidate on a boot and a repeat boot, retaining every run, output/quality
and batch-invariance gates, worker and GPU wait, actual read calls/bytes,
cache hit/miss counts, shared-memory observations, errors and confidence
intervals. Compare to compatible recorded release runs; never remeasure a
released build. Candidate-38 TP3 recorded evidence lives under
`experiments/2026-11-03-candidate-38-tp3-qualification`; it is not a TP4 or
per-reader I/O reference and cannot establish the benefit claimed by this
synthetic TP4 fixture. Preserve workload identity before comparing throughput.

Conclusion: retain the bounded patch as a reviewable hypothesis and reject
the unbounded variant. Do not promote on planned-read reductions. Current
production performance and memory are unchanged.
