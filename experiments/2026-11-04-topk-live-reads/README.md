# Guard inactive top-k merge scratch reads

A sparse candidate list can fill fewer merge input lists than the planned
indexer width. The old bitwise condition reads `WorkEntries[row, c]` even when
that list is inactive. Those entries were never written. It masks the resulting
key afterward, so ordinary output comparison misses the invalid read.

`vllm/0001-topk-merge-live-reads.patch` moves list liveness to a lazy outer
condition. Inactive lists contribute the same zero key without touching their
scratch. Live score arithmetic, tie order and selected positions are unchanged.
It is independent of clustered search and adds no attention approximation.

This is an unpromoted local fix against the exact promoted vLLM series (upstream
`19f2c20ed4d64ae22bd009ceacadd85e2a81e354`, prepared tree
`ea1281f880e7535e3d87725be0156a8e38fc2fb2`). The deployment base commit is in
`base.json`. No runtime configuration, image, baseline or frozen candidate
series is modified. Upstream status: not submitted; this module currently lives
in the deployment's local patch stack.

## Regression evidence

The independent reproducer imports only `IndexerTopK`. It plans 16,384 keys,
provides 576 equal live scores, and asks for 512 winners. The visible output must
be positions 0 through 511. Cache tails and inactive merge lists are unwritten.

On idle dgx4, using the exact recorded 1030 image and Compute Sanitizer
2025.3.1.0, the original reports **1,536 initialization errors** and the guarded
implementation **0 errors**. Both return the same exact ordered positions.
The failing and passing native receipts are retained. `provenance.json` records
their original published commit and commands: these are copied evidence from
the completed shared slot, not new GPU executions. The script and fixed overlay
are byte-identical to that run. There is no released-build timing or speed claim.

The shared slot began with about 116 GiB MemAvailable and a hottest sampled
sensor of 40.2 C. CUDA memory caching was disabled for initialization checking;
no allocator-memory peak is inferred. Production stayed unchanged, and the
window closed at 10:15:21 UTC on 2026-11-04 before the next qualification.

## Reproduce

```sh
bin/spark doctor
bin/spark upstream prepare vllm
python3 experiments/2026-11-04-topk-live-reads/check_sources.py
git diff --check
```

After publishing a clean commit, apply this ordered experimental series to the
fresh prepared vLLM tree. The fixed executable overlay must match. In an
authorized shared window, on idle dgx4, run:

```sh
scripts/lab.py kernel-local experiments/2026-11-04-topk-live-reads/original --out results/private/topk-original-new
scripts/lab.py kernel-local experiments/2026-11-04-topk-live-reads/fixed --out results/private/topk-fixed-new
```

The original is an expected failing control: its native verdict must fail even
though the functional output is exact. The fixed run must pass initcheck and
selection. Never overwrite an earlier run. This fix has no serving qualification
or promotion; adoption into a future candidate must keep that candidate's own
build, correctness and memory gates.
