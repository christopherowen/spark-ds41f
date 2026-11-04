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

Full-image and full-model release qualification remain separate from these
kernel and runner-component checks. No model throughput improvement is claimed.
