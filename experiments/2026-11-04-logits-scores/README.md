# Selected logits in TileLang

One variable: replace the raw_logits/processed_logits score path's ATen gather,
FP32 conversion and custom-id mask with one TileLang store. The default mode
remains raw_logprobs. This is a source candidate after the exact candidate-38
stack, with no promoted configuration or serving changes.

`base.json` pins the deployment base, upstream base, candidate-38 source tree
and immutable image. `candidate.json` pins the patched tree and every exported
file. `vllm/series` is the authority; `overlay/` is its exact export for screening.
The patch carries its rationale, output contract, tests and upstream status.

The kernel supports FP16/BF16/FP32 logits, int32/int64 ids, dynamic shapes and
strides, and an optional validity tensor supplied by the custom-token-id path.
Invalid columns store negative infinity without dereferencing their ids. Values,
signed zeros, infinities, NaNs, ordering and ranks retain the existing contract.
Every dtype, id-type and masking specialization is declared before cache freeze;
IKET's existing CUDA pass instruments the generated loads and stores.

Acceptance requires two independent container boots with correctness tests,
paired graph microbenchmarks and an actual decoded IKET capture. Tests cover
strides, large selected-id counts, empty outputs, nonfinite values, graph replay,
frozen-cache preparation, prompt chunks, speculative chunks and custom ids in
all four score modes. Benchmarks use independent ATen micro-operations, not a
remeasurement of a released model. Retain every run, including failures.

Run from a clean published checkout on owner-authorized idle sp8:

```sh
python3 experiments/2026-11-04-logits-scores/run.py \
  --experiments experiments/2026-11-04-logits-scores --run scores-v1
```

The runner verifies the frozen image's source hashes before applying the export,
holds an exclusive sp8 lock/reservation, monitors memory, pre-cools below 55 C,
and removes only its own container and hold. Each container uses a fresh virtual
environment with the image's installed packages. Native traces remain on sp8;
portable receipts include commands, hashes and memory observations.

## Results

All six jobs passed on idle sp8, using published deployment commit
`d28bb7424bf26f3f5c13db62fc5bc4864a4ce0d2` and the source tree in
`candidate.json`. Both test boots passed **87 tests**, without failures or skips.
This includes the 12 prepared specializations served with a frozen cache,
strided and masked gathers, nonfinite values, graph replay, prompt chunks,
speculative chunks, and host runner checks for all four modes and custom ids.

Both timing boots passed **36 cases**, retaining 21 cold-L2 CUDA-graph samples
per method and case. The six dtype/masking launch inventories each show one
`gather_logits_kernel` and no ATen GPU kernel for the candidate. Across the
measured cases the candidate's ratio of boot means ranges from level to 67.72%
shorter operation time. All timings, including level cases, are retained.
Two boots and these small operation times limit performance conclusions;
`results.json` records both boot medians and 95% Student t intervals (df=1).

Both native IKET captures passed, each with **12 selected-logit launches**
(six eager, six CUDA-graph replays) and decoded `load.global` and `store.global`
ranges. Each capture checks numerical outputs as well as trace contents.
The existing phase pass covers the new kernel without another instrumentation
mechanism. The capture is deliberately small and uses a 0.25 GiB native trace
pool; it does not requalify the separate disk-streaming collector.

Native evidence stays at
`/home/chris/spark-lab/logits-scores-20261104/scores-v1` on sp8.
`runs/native-v1.json` contains the exact commands, identities, outcomes,
memory samples and hashes of 97 retained files. Minimum sampled host
MemAvailable was **112.33 GiB**, with no guard stop. The exclusive reservation
was released, the test container removed, and sp8 was idle afterwards.
No serving operation was performed.

`runs/local.json` records passing source/export checks, Python parsing and doctor,
with no new lint diagnostics (three inherited test diagnostics are identified).
All 572 candidate-38 patches reproduce the pinned source before this patch
applies in a fresh tree. `runs/composition.json` also confirms that this patch
and the three preceding ATen replacement candidates apply together; that is
a source check, not combined GPU qualification. The full build check reports
that its build directory is not prepared locally.

Reproduce the portable analysis without hardware access:

```sh
python3 experiments/2026-11-04-logits-scores/analyze.py \
  experiments/2026-11-04-logits-scores/runs/native-v1.json
```

Full-image and full-model release qualification remain separate from these
kernel and runner-component checks. No model throughput improvement is claimed.
