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

Pending the hardware runs. The abort's cause is not assumed from the last
logged launch or from its signal alone. Nsight remains the independent
application-timing control; changing profiler logging did not remove the
previous whole-model timing distortion.
