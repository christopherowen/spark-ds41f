# Overlap the attention's reduce-scatter with its WO projection (r6d)

Owner, 2026-10-10: after benchmarking the r6d candidate, overlap the
reduce-scatter with compute (step 2), then fuse its sum (step 3).

## Why

r6d's deterministic reduce-scatter (C3) carries a third more wire bytes than
NCCL's ring on ring4 and costs 2-3% of prefill. Under prefill sequence
parallelism every decoder layer reduce-scatters twice: the attention's WO
output and the MoE output. At an 8192-token step the WO projection (B12X,
about 4.2 ms) and its reduce-scatter (about 4.6 ms on the fabric) run back to
back, so hiding one under the other is worth more than C3's cost.

## What changes (vLLM branch `r6d-overlap`, on r6d's tree)

- `440d27195` (module): `SPRows.project_reduce_scatter(project, ...)`. In
  slices, each rank's block is projected piece by piece into one contiguous
  buffer per piece, and each piece is reduce-scattered on a side stream while
  the next is projected; with one slice it is the existing path.
- `0cd9a8d9c` (switch): under the TileLang family, sequence-parallel steps of
  2048 tokens or more project WO in two slices. The L2 prefetch of the next
  FFN weights still issues after the last projection, before the last
  reduce-scatter is awaited.

The bits do not change: B12X's WO rows are independent of one another (the
migration's B1 bench found its batch-invariant), and the rank-order sum is per
element. The B12X family keeps one projection, since NCCL's ring sums depend on
the message size. The MoE output is not sliced: its reduce-scatter input is
the shared and routed experts' BF16 sum per rank, which cannot be split
without changing that rounding.

## Arms ([w1.json](w1.json))

`overlap.json` mounts the two changed files over the r6d image
([make_arms.py](make_arms.py)); the control is r6d's recipe. The kernel bundle
runs the layout test in the r6d image. The lean screen adds the
temperature-0 output check and mixed-traffic latency; prefill at 16K tokens
runs two 8096-token chunks, the sliced steps.

Window 1 first rebuilds `-r6d` with the router fix of the r6d gate (patch
0072; the image job replaces the stale tag on every node) and reruns the gate
tests in it, then the layout test and the screen.

Acceptance: temperature-0 outputs identical, decode level, prefill faster
than r6d.
