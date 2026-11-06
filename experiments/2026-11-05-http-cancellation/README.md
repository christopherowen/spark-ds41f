# HTTP request cancellation ownership

Follow-up: [repeated cancellation and measured CPU cost](../2026-11-06-http-cancellation-cost/README.md) closes an additional cleanup edge case and records the newer 574-patch candidate. The source identities and results below remain the evidence for this initial fix.

Cancelling a route can leave its handler and disconnect listener alive, retaining
request data and continuing work after the caller has gone. Ordinary disconnects
also return before asynchronous handler cleanup finishes. A response/disconnect
race can decrement the optional frontend load count twice, and cancellation
while handing a completed response back can retain that count.

This experiment changes one ownership boundary: the HTTP route owns both tasks
until they finish, and its load count is released once. Cleanup is shielded from
surrounding AnyIO cancellation; receive failures remain errors. Model arithmetic,
sampling, scheduler policy, prompt/output limits and response data are untouched.

Source PR: https://github.com/christopherowen/vllm/pull/1. It is one source commit
on the pinned upstream revision, with two changed files. The deployment candidate
appends that patch to candidate-38; both TP3 and TP4 profiles are launch-disabled
and name an unbuilt image. This is independent of the failed-engine-submission
and utility-future changes in deployment PR890 (different files and lifetimes).

## Identity and source validation

`base.json` names the deployment commit, candidate-36 and candidate-38 pins,
container image, tested file hashes, source commit and final replay identity.
Promoted 1030, committed candidate-36 and candidate-38 all inherit the same
`vllm/entrypoints/serve/utils/api_utils.py` from upstream `19f2c20e`; none of their
local patches modifies it. Its hash matches the actual candidate-38 image used
for the CPU checks. The shared checkout's in-flight candidate-36 changes were
not copied or edited.

The final 573-patch stack was applied in a fresh remote worktree from the pinned
upstream revision: head `7c99bc37b7f7e1219609347a2124edc04d847505`, tree
`c6fb7fdffa9a5ce345cc4cd850301a62787045ae`. Both changed files match the tested
source. The source patch and combined tree pass whitespace checks. The
experiment's release-test classification selects 198 files (136 gate, 62
excluded), with zero missing, ambiguous or unclassified files. The new tests
use ordinary synchronous pytest entrypoints with an isolated asyncio loop;
no additional test plugin is required by the final patch.

## Results

| Check | Original source | Final corrected source |
| --- | --- | --- |
| CPU lifetime regression cases | 15 fail | 15 pass |
| Real Uvicorn HTTP boundary cases | not rerun for comparison | 12 pass |

The lifetime cases cover caller cancellation, disconnect, asynchronous abort
cleanup, AnyIO cancellation scopes, handler and receive errors, preservation of
existing response/background data, response handoff, and load accounting. Four
separately allocated 1,047,552-token payloads remain blocked while sixteen short
requests complete; then two callers cancel, two disconnect, and all four retry
the same identifiers after cleanup. No owned tasks or retained prompt mappings
remain. This models frontend ownership, not GPU scheduling or KV capacity.

The HTTP cases use the real completion router, Pydantic request model, native
`TokenizeParams`, exception handlers, Uvicorn and HTTPX over loopback. Inference
alone is stubbed. They pass UTF-8 JSON bodies of exactly 2 MiB minus/plus one byte
and 64 MiB minus/plus one byte, chunked uploads split across JSON escapes, the
1,048,576 combined input/output boundary (including one token over), explicit
malformed-JSON and invalid-token errors, and the existing opt-in SSE keep-alive
through a 100 ms idle read timeout during a 250 ms response delay. Stream
closure releases the generator and tracked load.

Final Python maximum RSS was 959,916 KiB (0.92 GiB) for lifetime cases and
1,491,056 KiB (1.42 GiB) for HTTP cases. Containers were limited to two CPUs,
4 GiB RAM/swap and 256 processes, with no GPU devices and no external network.
These are correctness measurements, not serving-speed or memory-leak-rate
benchmarks; no throughput gain or confidence interval is claimed.

All native results and intermediate failures are retained under `runs/` and
summarized in `results.json`. The first HTTP harness omitted an exception-handler
setting and checked background completion before it could run; fixing those
harness errors produced 12 passes. A further cancellation-at-handoff case caught
a real remaining load-count leak; the final patch releases that count once.
Earlier plugin-based runs remain evidence; the final source tests need no plugin.
Initial setup errors are listed in `runs/setup-events.json`.

## Request-path findings and limits

The recipe exposes vLLM/Uvicorn directly on port 8000 with host networking. No
HTTP gateway configuration is owned by this repository. These checks do not
qualify a user's external proxy, tunnel, SDK timeout or retry policy.

The tested Python request path accepts bodies above 64 MiB without introducing
a new body-size cap. The combined token limit is enforced separately by native
rendering validation; no truncation was enabled. Indexer positions, candidates
and page-table entries use int32, while slot/position write addressing uses
int64. No corresponding index-width defect was demonstrated, so there is no
indexer change.

The recipe already uses paged KV, a 1M context, one concurrent prefill, aligned
chunks, short-request prefill yield and a ten-second yield limit. It preserves
eight TP3 or sixteen TP4 admitted sequences. Recorded release evidence remains
the authority for capacity and mixed GPU traffic; this patch adds no fixed
request partitions and makes no new GPU fairness claim. Optional keep-alives
already wrap chat/completions and completions, but the promoted recipe leaves
them disabled; the Responses router does not use that wrapper. Those policy and
coverage choices are unchanged by this focused fix.

Cancellation still relies on cooperative handler cleanup. HTTP requests are not
idempotent: a client may retry before the server observes its disconnect. This
fix prevents orphaned route-owned work and waits for cleanup before the route
returns; it does not deduplicate arbitrary retries. Stream send-failure and
unrelated background-callback failures after response handoff are outside this
handler-phase change.

## Reproduction

Use the immutable image in `base.json` on an isolated CPU host. Stage
`run_cpu.py` and `test_request_boundaries.py` beside the source test file
`tests/entrypoints/serve/utils/test_api_cancellation.py` from the prepared tree.
Copy the corrected `vllm/entrypoints/serve/utils/api_utils.py` beside them as
`corrected.py`. `run_cpu.py` records the installed and tested source hashes,
package versions, native pytest status, JUnit and maximum RSS.

The native command is:

```sh
docker run --rm --network=none --memory=4g --memory-swap=4g --cpus=2 \
  --pids-limit=256 -e CUDA_VISIBLE_DEVICES= -e VLLM_PLUGINS= \
  -v "$BUNDLE:/work:ro" -v "$RESULT:/results" \
  --entrypoint /usr/bin/timeout \
  sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b \
  180 /usr/bin/python3 /work/run_cpu.py \
  --replacement /work/corrected.py --output /results
```

Omit `--replacement` for the original source; add `--boundaries` for the HTTP
checks. Run jobs serially and use fresh result directories. This command never
starts a model or exposes a host port.

Both candidate profiles pass offline doctor with site node maps present. The
first metadata/site-map failures are preserved. Complete build inputs are not
prepared for this candidate; `build check` records that limitation. No image
was built, no serving node was synchronized or restarted, and no released
build was benchmarked. Full image/release and serving qualification remain
required before promotion; no runtime operation is authorized by these files.
