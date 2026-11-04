# Useful CopyDrafts continuations and exact search pruning

Base: deployment 6a97c71d, candidate-38 image and vLLM identities in base.json.
The candidate is opt-in; no promoted lock, recipe or baseline is modified.

Two ordered, independently measured variables:

1. Rank equal-strength suffix matches by continuation length capped at lookahead,
   then recency. Preserve a short copy when no full continuation matches at that
   level. Encode priorities in the existing int32 source buffer; both allocation
   and selection decode the same priority. Include lookahead in warmup keys and
   allocation in the bundled pipeline benchmark.
2. Visit candidates in descending priority and stop a warp after a longest-level
   hit, which also dominates all remaining candidates at every shorter level.
   This keeps exactly the first patch's choices, with no bounded window, index,
   new buffer or host synchronization. Bounds and votes are uniform per warp.

Apply series after candidate-38 vLLM head 1f33102547e4da5c64712390604695d1abd577c4
(upstream 19f2c20ed4d64ae22bd009ceacadd85e2a81e354 plus its pinned series).
Patch bodies state upstream base, rationale, quality implications, tests and
upstream status. Both are local candidates, not submitted upstream.

Each candidate boots twice for correctness and twice for the same 96-case,
21-sample cold-L2 graph timing matrix as the prior recorded assessment. Compare
with those recorded candidate-38 runs, without rerunning the released baseline.
Record every run, error and memory observation. Only idle authorized sp8 is used;
serving clusters are not started, stopped, synced or restarted. A clean published
checkout supplies patches; each throwaway test container first verifies original
source hashes, loads Git-tracked exports from the freshly applied patch series,
then verifies every changed file hash before importing vLLM. These overlays are kernel tests, not a serving image build.

Acceptance: independent reference outputs, full and short continuation boundary
cases, multiple lookaheads/minimum-match levels, per-request/slot isolation,
allocation agreement, cap growth/shrinkage, CUDA graph replay, committed-text
invariance and no errors. Preserve every unfavorable timing. No real-model
acceptance, distributed performance or release qualification is inferred.

## Results

Both variants pass all **40 tests on both boots** (160 passing test executions):
independent NumPy selection and state references, request isolation, slot reuse,
masked prefill, growth/shrinkage, allocation agreement, graph replay and toy-target
committed-text invariance. The new cases exercise short-only and full-copy
continuations, strongest-level precedence, multiple lookaheads and scan boundaries.
Both variants also pass all 96 timing/reference cases on each of two boots.

Selection alone fixes the constant-token proposal from one draft to eight at the
initial cap. Acceptance can then grow it to sixteen, covered by the regression
suite. The scan optimization preserves every selection result and offered width.
The two stages are measured separately; selection alone does not solve the
repetitive-search cost.

Final three-kernel pipeline, UVA, 1M tokens **per request**, microseconds. Ranges
are the two boot medians; timings are uninstrumented. The original numbers are
the prior recorded candidate-38 measurements, not a remeasurement.

| Input | Original, 1 request | Patched, 1 request | Original, 16 requests | Patched, 16 requests |
| --- | ---: | ---: | ---: | ---: |
| Random, no match | 81.60–81.66 | 83.97–84.00 | 468.93–472.77 | 357.09–360.13 |
| One copied tail | 96.00–104.29 | 85.73–85.73 | 465.63–477.89 | 358.40–366.27 |
| Period-64 history | 2845.41–2846.72 | 81.92–81.92 | 6061.76–6073.06 | 210.94–215.81 |
| Constant-token history | 2825.95–2831.04 | 34.50–34.53 | 6826.72–6827.01 | 75.49–75.78 |

This is about 35–82x less pipeline time for one long repetitive request, and
28–90x for sixteen. These are kernel timing ratios, not generation speedups.

Unfavorable results are retained. At one request, 1M no-match UVA is about 2.4 us
slower; device placement is about 5.0 us slower on the boot means. At 4K,
period-64 UVA is 30.72 us versus the original 28.64–28.67 us. The largest mean
percentage increase in the final 96-case matrix is 7.93% (8 requests, 64K,
device, no match: 30.72/26.78 us versus 26.66/26.62 us). Two boots give limited
generalization; the JSON retains both medians and 95% t intervals, and percentage
ratios are not significance claims. No timing is hidden by a blanket pass flag.

## Evidence and reproduction

The source patch series was applied in a fresh tree after reproducing all of
candidate-38's pinned vLLM series from deployment commit
`df50aa94efb5f7e897f9e1281a9e00c207dec762`. Both resulting trees match
`candidate.json`. Test exports are Git-tracked copies of those exact files;
original and patched hashes are checked in every GPU job before importing vLLM.

Apply the candidate after the pinned candidate-38 series, on a named source branch:

```sh
git switch -c copy-drafts-candidate 1f33102547e4da5c64712390604695d1abd577c4
git am /absolute/path/to/2026-11-04-copy-drafts-fix/0001-useful-continuations.patch
git am /absolute/path/to/2026-11-04-copy-drafts-fix/0002-prune-exact-search.patch
```

Hardware run source: deployment `4fa8339f2941ebce1dfc3bb6c25717befe88e37a`,
clean and detached on sp8. Exact container commands and matrix, all samples,
image/source hashes, failures and host memory observations are in
`runs/hardware.json`. Raw files remain under
`/home/chris/spark-lab/iket-streaming-20261104/{copy-drafts-fix-v1,copy-drafts-fix-v2}`.
The controller was launched with:

```sh
python3 experiments/2026-11-04-copy-drafts-fix/run_matrix.py \
  experiments/2026-11-04-copy-drafts-fix/matrix.json --run copy-drafts-fix-v2
```

Recreate the normalized comparison from the saved receipt (no GPU required):

```sh
python3 experiments/2026-11-04-copy-drafts-fix/analyze.py \
  experiments/2026-11-04-copy-drafts-fix/runs/hardware.json \
  experiments/2026-11-04-copy-drafts-fix/results.json
```

The successful window's lowest sampled host MemAvailable is 113.05 GiB; no
memory guard stop or workload error. The change adds no persistent GPU buffers.
The test container was removed and sp8's hold released, verified in the receipt.
No serving operation was performed.

Ruff, Python parsing, fresh-series tree checks and branch-wide `git diff --check`
pass. `bin/spark doctor` passes. `bin/spark build check` reports the promoted
build directory is not prepared locally; this is a source-overlay test, not a
new image build. Exact check outputs are in `runs/checks.json`.

After those completed runs, the optional two-boot streaming IKET check was
prepared at deployment `8a27613464f6e91e9b32c119b2ef681db99ec8ff`. SSH banner
exchange to sp8 timed out before any remote command, so no additional profile
window launched. Post-test SSH health queries to sp1/sp5 failed the same way;
service health cannot be inferred from this connection failure. Candidate-38's
original CopyDrafts IKET captures remain in the preceding assessment; no IKET
compatibility claim is made for the changed loop. `matrix-profile.json` retains
the exact pending check.

## Decision

Retain the two patches as the measured candidate: continuation selection is
corrected, the repetitive-search penalty is greatly reduced and the tested
correctness contracts hold. No bounded history search or incremental index is
needed for this improvement. Promotion still needs a built-image check and
real-model distributed serving acceptance/throughput qualification. This
experiment changes neither promoted configuration nor immutable release evidence.

Initial attempt `copy-drafts-fix-v1` (deployment 028f0226) failed all eight jobs
before GPU work: the frozen runtime image has no Git executable. Preserve its
logs and memory receipt. The harness now uses exact Git-tracked source exports,
verified against the fresh patch trees, without installing tools in the image.
