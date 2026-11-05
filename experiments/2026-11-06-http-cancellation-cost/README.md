# HTTP cancellation cost and repeated cancellation

This is a CPU-only follow-up to the [request ownership experiment](../2026-11-05-http-cancellation/README.md).
It tests the useful-capacity consequence of abandoned work and closes a further
cleanup defect. The only causal variable is the exact `api_utils.py` implementation.
The base deployment commit is `0c91cc7da2b5b689291289dfb190257a52fcf9f9`.
The selected source follow-up is `6ca2b7daa3e8ab3e857004569fbc351ea8eb7446`,
on top of `385a31417f9d484dd130ccedd868bc7c88615734`.

These results prove recovery of simulated occupied capacity through the native
request wrappers and real HTTP transport. They do not measure model inference,
tokens per second, KV reclamation, or production agent throughput. The serving
cluster was reserved by another holder and was not used. No released build was
re-benchmarked: the control is the unchanged API module in the unpromoted
candidate-38 image, with a synthetic handler.

## Small, reproducible workload

All jobs ran serially on sp8, each in a fresh, network-isolated container capped
at two CPUs, 4 GiB and 256 processes. No GPU device or host port was exposed.
The immutable runtime image is
`sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b`.
Each implementation is loaded as a complete native module; only the handler's
engine work is simulated. Source hashes, package versions, raw samples, exact
commands and memory high-water marks are retained in `runs/` and `results.json`.
Peak Python RSS across all completed runs was 1.42 GiB, below the 4 GiB cap.

`probe.py` uses four independent requests holding four simulated service slots.
Each long job lasts 200 ms, advances a work counter at 1 ms intervals, and needs
5 ms of cooperative abort cleanup. A useful new job needs 5 ms. Once all four
jobs have started, the harness either cancels their caller tasks or sends an
ordinary disconnect, then immediately submits the useful job. It counts work
after cancellation, occupied slot time, orphan tasks and load counts before
explicit fixture cleanup. A separate case completes 1,000 normal JSON responses
per sample, including their native background callbacks and exact response bytes.

`http_probe.py` exercises the same native wrappers through FastAPI, Uvicorn and
httpx over real loopback sockets. Four distinct jobs hit a fixed 35 ms application
deadline; all four must return the same explicit HTTP 504, then a distinct short
request must return the same HTTP 200 and body. The 200 ms job and 5 ms cleanup
remain identical across arms. It measures both the entire request wave and the
subsequent short request, followed by an uncontended short request to establish
the remaining transport/service floor. This fixture's deadline policy is not a
new production setting. It covers handler-phase cancellation, not post-handoff
stream errors or retry deduplication.

Every timing comparison has 12 paired trials, alternating arm order, on each of
two fresh container boots. The intervals below are paired, two-sided 95% t
intervals within each boot (11 degrees of freedom). They describe these small
samples, not production variability. The normal-path case totals 24,000 measured
successful responses per arm for each candidate, excluding warmup.

## Results for the selected wait-based cleanup

| Real HTTP measurement | Boot 1: original / fixed | Boot 2: original / fixed |
| --- | ---: | ---: |
| Short request after the four deadlines | 198.77 / 47.04 ms | 195.60 / 46.99 ms |
| Entire wave through useful short completion | 255.38 / 110.08 ms | 253.65 / 108.93 ms |
| Uncontended short request | 47.37 / 47.32 ms | 47.16 / 47.27 ms |

The short request is 76.0-76.3% faster **in this fixture**. Its paired latency
reduction is 151.73 ms (95% CI 138.30-165.17) and 148.61 ms (135.10-162.12).
The full wave improves by 145.29 ms (129.96-160.62) and 144.72 ms (131.82-157.61),
including the time spent waiting for cleanup before sending the timeout response.
Every fixed HTTP trial has zero active abandoned jobs when the deadline responses
finish and zero work steps after those responses. The original leaves all four
jobs running. The fixed short-request latency reaches the measured uncontended
HTTP floor; there is no remaining abandoned-job queue delay to optimize here.

The direct wrapper probe likewise improves useful completion from 211.6-213.6 ms
to 11.4-13.4 ms. The mechanism's minimum is the unchanged 5 ms abort plus 5 ms
short job, before event-loop scheduling. Fixed caller cancellation performs zero
additional simulated work steps and leaves no orphan tasks or retained load
count. The control continues hundreds of work steps and leaves four orphan
listeners and a load count of four before fixture cleanup.

Ordinary disconnects do **not** show a useful speedup: useful completion remains
about 11 ms. Successful requests also show no resolved improvement or regression:
the wait-based wrapper's paired change is -3.06 us (CI -24.08 to 17.96) and
+2.65 us (-28.11 to 33.40) per request. These noisy, very small frontend costs
cannot be turned into a token-throughput claim.

## Alternatives and the additional defect

The first fix shields cleanup from AnyIO cancellation but directly awaits a
gather. A second `Task.cancel()` propagates through that gather and cancels the
children again, interrupting their abort or listener cleanup. The new tests
demonstrate this during handler abort, repeated timeouts and response handoff.

| Cleanup strategy | Stronger lifetime suite | Evidence and decision |
| --- | --- | --- |
| First fix: shielded scope, direct gather | 14 pass, 4 fail | Retained in the preceding experiment; incomplete for repeated cancellation. |
| Shielded gather with pending cancellation replayed after cleanup | 18 pass | Correct alternative. Adds a shield future; normal-path paired cost was +21.55 us and +12.83 us in its two boots. |
| Wait for both children, retrieve outcomes, replay pending cancellation | 18 pass | Selected. `wait()` does not forward caller cancellation to children; avoids the extra shield future. All response and error semantics remain covered. |

Both robust alternatives recover the same simulated capacity. Their separate
normal-path confidence intervals overlap, so this is not a claim of a statistically
proved direct speed difference between them. The selected design has fewer
future objects on the successful path and no detected normal-path penalty in
its two runs. The changes to either alternative, relative to the first fix, are
preserved in `alternatives/`; no unsuccessful run was discarded. The earlier
canonical-proposal comparison remains in the preceding experiment.

The first audit also exposed an inadequate old test fixture: its listener's
cleanup waited forever and depended on a second cancellation to kill it. Both
robust variants reached the 60-second test timeout. Those runs are retained.
The strengthened fixture now explicitly permits cooperative cleanup to finish
and asserts that it actually did finish. With that fixture, the first fix fails
four cases; both robust alternatives pass all 18. The final committed source
passes all 18 lifetime cases and all 12 native HTTP boundary cases again.

## Boundaries and conclusion

The remaining simulated useful-request latency is the unchanged cleanup and
service duration plus transport/scheduling. Avoiding that required cleanup would
restore the ownership defect. Uncooperative children can still delay cleanup;
an arbitrary timeout that abandons them would not meet this experiment's
correctness requirement. A full task-group rewrite is not justified by a
remaining cost measured here and would change exception propagation semantics.

Native model arithmetic, batching, scheduler policy, admission capacity, context
limits and response payloads are unchanged by the source diff. No new arithmetic
or batch-invariance claim is made: GPU quality, production mixed-agent traffic,
KV release and model throughput were not exercised. A built candidate image and
an authorized serving window are required to quantify those effects. This is
the strongest tested cleanup implementation among these alternatives, not proof
of a global optimum or completed serving qualification.

The new source patch and the full 574-patch candidate are recorded here. Both
profiles remain launch-disabled and unbuilt. Retain the source improvement for
review; do not promote it from these CPU results alone.

## Reproduction

Use the immutable image above. Stage the final source API file as `wait.py` in
the source bundle and verify its SHA-256 is
`83db925dcb00fcc0fe7a80b885876bf610100e4dbcd95cdda6be6fc8510e235a`.
For each of two fresh containers, run `probe.py` and `http_probe.py` with
`--replacement /source/wait.py --expected-sha256 <that hash> --boot 1` (then 2)
and a fresh `--output` directory. Every invocation's complete resource limits,
mounts, image and arguments are in its `runs/*/command.txt`.

`runs/probe-initial.py` and `runs/http-probe-initial.py` preserve the earlier
harness versions. The lifetime suite is in the source PR; the preceding
experiment supplies `run_cpu.py` and the native HTTP boundary suite. The audit
used `run_audit.py`, the saved 15-case `runs/audit-test_api_cancellation.py` staged
as `test_api_cancellation.py`, and `test_repeated_cancellation.py` together.
`runs/replay-commands.json` records the fresh full-series replay. No command in
this experiment authorizes a serving-cluster operation.
