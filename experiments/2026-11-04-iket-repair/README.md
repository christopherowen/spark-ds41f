# IKET capacity checks and loading-abort isolation

Owner, 2026-11-04: try the IKET repairs. Authorized hardware is idle sp8,
candidate-38 only; both serving clusters remain running.

Base deployment commit: `f1957b71cccb76fae67d256fd04f43db569e2fb0`.
Candidate image: `sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b`,
vLLM tree `c061e5f6931306fa4655ba531428f639c365bf10`.
No serving or build recipe, upstream series, image, or baseline is changed.

## Capacity repair

The existing profiling wrapper gains `--event-sources`, exact replay-kernel
selection with `--event-kernels`, and `--single-pass-max-ts auto`. It records
source-manifest hashes, adds four timestamp events to the largest finite
bound, then rounds up to eight slots. A 60-event bound therefore selects 64.
Explicit smaller values, unknown/malformed bounds, missing selected kernels,
and an explicit pool above `--max-pool-gb` are refused before any launch.
Historical unchecked invocations remain compatible; certified single-pass
replays use the new source check.

Local checks: 21 passed, covering the original wrapper and full-layer helpers
plus capacity selection and refusal cases. No affected upstream patch series:
the changed file is an experiment's profiling wrapper.

## Abort experiment

One causal variable per comparison: no profiler, original NVIDIA warp-trace
library, or a private copy with the existing raw-dump bypass. Workload and
buffer size stay fixed within each comparison. Small and loading-size inputs
are distinct workloads. Every point boots twice and retains failures.

`reproduce.py` calls the image's unchanged `overlay_byte_codes` and
`narrow_code_offsets`, which exercise ATen boolean indexing, gather, scatter,
and compaction during MoE scale preparation. The large dimensions are 384
experts, 1,152 rows, 160 scale blocks. A simple integer-add control checks
basic profiler operation. Every result is checked against its expected
values and hashed. There are no model weights, Engram, or real peers.

`matrix-initial.json` defines all 18 jobs. Pools are 0.25 GiB for small jobs,
2 GiB for large jobs, with a 32 GiB container cap. The runner verifies the
image and clean published checkout, reserves sp8, takes its GPU lock,
refreshes a heartbeat, starts below 55 C, samples memory, and kills only its
own container below 8 GiB host MemAvailable. Failed jobs remain in the matrix;
a memory guard ends the window. It removes its container and hold on exit.

Only a clean published Git checkout is used on the node. The primary node
checkout and all serving containers stay unchanged. Exact Docker argv,
identity, output, memory samples, results and all failures are written below
`/home/chris/spark-lab/iket-repair-20261104/<run>/`.

Initial command, from the isolated published checkout on sp8:

```sh
python3 experiments/2026-11-04-iket-repair/run_matrix.py experiments/2026-11-04-iket-repair/matrix-initial.json --run initial
```

## Results and conclusion

All 50 jobs are retained in `runs/*.json`, including five deliberate/reproduced
failures and the compile-only preflight. Each workload comparison has a boot
and a repeat boot. This is a profiling-tool experiment on one GPU with dummy
weights, not a serving-quality or multi-rank release qualification.

| Comparison | Boot / repeat results | Conclusion |
| --- | --- | --- |
| Integer add, small/large scale overlay; plain/original/private dump-bypass library | 18/18 pass, matching expected output hashes | Basic injection and scale overlay work |
| Large `pack_expert_scale_codes`; same three modes | 6/6 pass | Packing alone does not explain the abort |
| Small/large `tile_expert_weights`; same three modes | 12/12 pass | One strided signed-byte copy works |
| Full 43-layer loading, original library, 8 GiB pool | SIGABRT in both boots, plus one debugger follow-up | Dump bypass is not required for the abort |
| Three large weight tilings, original library, 256 MiB pool | SIGABRT in both boots | Small reproducer of the same failure stack |
| Same three tilings, only pool increased to 1 GiB | Both pass, hashes equal to both plain controls | Pool exhaustion triggers this failure |
| Bounded BF16 GEMM replay with certified phase events | Both phase boots and both plain controls pass | Usable, bounded capture workflow |

The debugger stops at SIGABRT and then kills the inferior, so its job receipts
show exit 247 (child SIGKILL), rather than the original SIGABRT. The captured
stack passes through `_Unwind_SetGR`, `__gxx_personality_v0`, NVIDIA's injection
library, `cudaLaunchKernel`, and ATen's signed-byte copy. The attempted
`__cxa_throw` breakpoint did not capture an exception message; no claim is made
about the profiler's internal exception implementation.

The buffer accounting is decisive. All three full-model failures had recorded
8,502,325,632 bytes in an 8,589,934,592-byte pool: only 87,608,960 bytes remained.
The reduced tiling test needs 212,336,640 bytes for its copy record. With a
256 MiB pool, the initial fill and first copy consume 265,420,800 bytes; the
next copy aborts with the same stack. A 1 GiB pool holds all six launches,
799,580,160 bytes, in both boots. The large packing/tiling tests use
`--no-finalize`: their success validates execution and output, not profiler
exit/export. The small GEMM tests do finalize and export normally with the
original library.

The repair is to bound the profiled workload and size its buffer for every
launch, including preparation. Increasing a whole-model pool from 5 to 8 GiB
only moved the exhaustion point. Do not assume that allocation failure will
produce a useful error from the installed profiler. Its native abort remains
unpatched; the bounded replay avoids it.

Candidate-38 manifest checks also run on the real prepared sources:
`mhc_project_finalize_kernel` selects 64 slots from its 60-event bound; explicit
42 refuses before launch; selecting all sources refuses the unknown bounds
for `bias_kernel` and `compute_slot_mappings_kernel`. These unknown bounds
remain open for broader native-event captures.

No memory guard fired. The minimum host MemAvailable was 22.65 GiB (full-model
debugging); other matrices stayed above 105 GiB. The hottest sampled thermal
zone was 55.8 C. Every job started below 55 C. sp8 is idle and its reservation
is released. Both serving endpoints returned HTTP 200 with candidate-37 after
the tests. No serving operation was performed.

## Bounded replay recipe

`matrix-gemm.json` compiles the image's unchanged `bf16_gemm(128, 512)` without
launching it, writing candidate-specific event manifests. Two independent
plain boots and two instrumented boots then execute a `[6, 512] @ [128, 512]^T`
GEMM with an exact, non-uniform CPU reference. Each instrumented boot checks
its generated manifest hashes against the certified preflight before launch.
The finite bound is 12 events, automatic capacity is 16, and each exported
trace records six events, three valid ranges, and 16 warp lifetimes. Both
exported traces pass timestamp/capacity validation. All four outputs hash to
`8edb0d93a9f96345a627a6c01ed7469622673fcf8b716affaa6de0eede12c935`.

Exact workload and profiler argv are retained in the run receipts. Each
matrix uses the same command form as the initial run, substituting
`matrix-{pack,tile,full-debug,full-throw,overflow,gemm}.json` and its matching
`--run` name. These commands require an authorized idle-sp8 window and a
clean published checkout; they are not cluster-start instructions.

For a bounded replay with precompiled, matching sources, the wrapper invocation
is:

```sh
python3 experiments/2026-10-31-iket-full-43l-and-runtime-opt/harness/iket_profile.py OUT/profile \
  --single-pass-max-ts auto --event-sources PREFLIGHT/sources \
  --pool-gb 0.25 --max-pool-gb 0.25 -- WORKLOAD
```

Use separate plain/Nsight runs for application timing. These captures validate
event collection and numerical equivalence; they do not remove or quantify
the earlier whole-model instrumentation distortion.

Validation: 21 local helper/capacity tests pass; `git diff --check` passes.
`bin/spark doctor` passes from the configured primary workspace. The isolated
worktree lacks its ignored site map, so its initial doctor invocation refused
that missing configuration. `bin/spark build check` reports the promoted build
tree is not prepared locally; no build was attempted. There is no affected
upstream series to reapply, and no runtime/build input changed.
