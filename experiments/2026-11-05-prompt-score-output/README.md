# Prompt score outputs

Base: native score-input candidate PR #895, source and deployment identities in
`base.json`. One intended variable: prompt chunks write final buffers and fuse
rank calculation into their selected-score pass. Apply `vllm/series` after the
exact candidate-38, logits-scores and native-score-inputs series. No promoted
configuration, immutable baseline or serving process changes.

Prompt chunks previously allocated intermediate result tensors, scanned ranks
separately and concatenated three outputs. Each chunk now writes its final row
slices. Logit mode emits selected scores in the existing exact rank pass;
log-probability mode counts exact ranks during the existing maximum scan. Both
native ID widths and their strides remain supported. Zero-alternative IDs remain
views. Score order and FP32 score/rank bits are preserved, including exceptional
logits. All selected variants are declared and tested under startup cache freeze.

Two combined boots pass 1,167 checks. Two focused boots pass 76 checks each;
these include cross-chunk boundaries, all four score modes and exact comparisons
against separate kernels with FP16/BF16/FP32 special values. All 120 normal score
programs, bindings and inputs match their prior per-case GPU qualification on
both boots. `source-equivalence.json` proves all six files in this patch are
byte-identical to the combined qualified source.

All ten prompt GPU timing cases pass the zero-margin gate on both boots. Two
cases received one predeclared 480-round confidence extension; primary means,
extension upper bounds and pooled bounds must all pass. Every observation is
retained. `confidence-extension.json` records both boots, with no slowdown
allowance. The 1,025-row full-vocabulary log-probability case is about 32% faster.
This is GPU work qualification, not a host or serving throughput claim.

Raw measurements, profiler traces, commands, identities and memory observations
are in `experiments/2026-11-05-tilelang-runtime-closure` on the published
`feat/tilelang-runtime-closure` branch: `combined-v4`, `fused-precheck-v4`,
`native-identity-v4` and `logits-extension-v6`. Historical unsuccessful variants
remain there too. The combined driver verifies frozen image and overlay hashes
before import, pre-cools sp8, takes an exclusive GPU reservation and runs each
point in a fresh container. Both serving clusters stay running.

A wire inventory separately found ATen packing when a strided native ID view
moves to CPU; that is an output-transfer issue outside this patch. The full
combined model inventory and built-image/real-fabric qualification remain
separate release gates. No performance result here qualifies networking.
