# Native token-id inputs for scoring

One variable: selected-token score kernels consume the token-id dtype and
strides supplied by their producers, replacing repeated int64 casts and
contiguous materializations. This builds on the selected-logit output patch
in PR #893; both remain unpromoted source candidates after candidate-38.

The log-softmax consumes int32 or int64 matrices and the producer's contiguous validity masks.
Top-k reads either sampled-id type and writes the required int64 result in its
existing final store. Ranks read the sampled vector's stride directly. Prompt
scoring no longer widens its ids before those operations. With no top-k columns,
reported ids retain their original view and integer width; the engine's
serialization converts either width to the same Python integers. FP32 scores,
floating-point arithmetic, ordering and token ranks are unchanged.

The source patch and `vllm/series` are authoritative; `overlay/` is an exact Git
export. `base.json` identifies the frozen image, prerequisite patch and every
source hash before/after that prerequisite. The harness verifies these stages
before importing the candidate, whose complete file hashes are in
`candidate.json`. No promoted configuration, default mode or source lock changes.

Correctness checks cover native dtypes, strides, masks, prompt boundaries,
speculative scores, wire ids, graph replay, FP64 score references and startup
preparation with a cold kernel cache. The benchmark loads the exact prerequisite
source's two kernel modules as independent references: normalization and
materialization costs are included there, with no remeasurement of a released
model build. Already-int64 contiguous inputs remain comparison cases, since
their old normalization did not launch extra GPU work.

Run from a clean published checkout on owner-authorized idle sp8:

```sh
python3 experiments/2026-11-04-native-score-inputs/run.py \
  --experiments experiments/2026-11-04-native-score-inputs --run native-v1
```

The runner uses the candidate-38 image by immutable digest, holds an exclusive
sp8 GPU lock and reservation, checks idle/memory conditions, cools below 55 C,
and monitors host memory. Tests, paired graph timings, and actual decoded IKET
captures each boot twice. Every run and failure is retained. No serving operation
is authorized by this experiment; full-image/full-model qualification remains
before promotion.

## Initial variant retained

The initial variant (candidate-v1.json, published source 7b7a619d) passed 625 tests
per boot, 120 timing cases per boot, and two IKET captures. Its full evidence and
analysis are retained as runs/native-v1.json and results-v1.json. One already-int64,
contiguous-input FP32 masked case read 0.93 us (+4.78%) higher on boot means, with
substantial reference variation; no serving slowdown is inferred from it.

That variant also generalized mask strides even though the old path never
normalized masks and the producer already writes them contiguously. Patch 0002
keeps that existing contract, trimming unused stride parameters. This narrows
the candidate; it does not change the native token-id support or score arithmetic.
The final variant requires its own repeated qualification. Initial receipts are
not relabelled as evidence for the new tree.
