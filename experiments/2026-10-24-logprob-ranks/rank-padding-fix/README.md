# Exclude vocabulary padding from reported ranks

This follows the bit-identical TileLang port in a separate logical change.
The original qualification and reference kernel are retained unchanged.

## Reproduced defect

The reference rank kernel rounds its vocabulary scan up to a multiple of 8,192
and loads absent columns as negative infinity. It counts `logit >= selected`
without excluding those absent columns. If the selected token's logit is negative
infinity, the padding passes the comparison.

DS41F has 129,280 vocabulary entries, but the scan covers 131,072 columns: 1,792
phantom entries. Scoring a negative-infinity token in an otherwise finite row
therefore reports rank 131,072 rather than 129,280. The first port's GPU tests
reproduced the same defect; preserving it established equivalence to the reference.

This is a rank-reporting defect. The regular sampler chooses tokens before calling
`compute_topk_scores`, and applies an optional forced trace before that call.
Prompt-logprob scoring also calls it with a supplied token. Sink recovery copies
the computed rank into output metadata; its rejection and alternate-selection
conditions do not read ranks. The rank feeds logprob reporting, not token
selection, probability calculations, verification or the next model step.

Valid sampling does not select a zero-probability token, so normal generation
does not reach this case. Forced-token scoring can reach it; prompt scoring can
as well if the supplied token has a negative-infinity logit. There is no observed
live-generation incident associated with this finding. An all-masked distribution
is a separate invalid-input problem, which this change does not repair.

## Correction

Count a comparison only when the column is inside the real vocabulary. Ties and
NaN comparison semantics stay the same. For a finite selected logit, output is
identical to the original kernel; for a selected negative-infinity logit, the
reported rank loses exactly the number of phantom columns.

The regression tests use all three supported logit types and DS41F's actual
129,280-token vocabulary through `compute_topk_scores`. They require rank 129,280
and the original kernel's rank 131,072 on the same input, while the returned
negative-infinity score and supplied token IDs remain unchanged. Existing
nonfinite tests use an independent real-vocabulary count as their oracle.

## Qualification

On DGX4 in the same candidate-4 image, all **27 GPU tests pass**, and all 27 also
pass under CUDA 13.0 compute-sanitizer memcheck with **zero errors**. All 15 paired
performance cases pass the unchanged gate. FP32 timings in microseconds per call:

| Rows | Triton | Corrected TileLang | Time change |
| ---: | ---: | ---: | ---: |
| 1 | 17.10 | 7.79 | -54.5% |
| 8 | 17.31 | 7.97 | -53.9% |
| 16 | 17.45 | 8.42 | -51.6% |
| 48 | 47.69 | 37.57 | -20.2% |
| 96 | 168.52 | 167.99 | -0.29% |

The benchmark inputs have finite selected logits, so its comparison with the
unchanged Triton reference remains exact. [results.json](results.json) contains
all dtype/batch timings, intervals and fingerprints; [run/](run/) holds the native
receipts. The previous 24-test qualification and its raw source remain intact.

The complete [vllm/series](vllm/series) is the first port's series plus the separate
[0267 correction](vllm/0267-logprob-rank-valid-vocabulary.patch). All 224 patches
replayed from the pinned upstream into a fresh tree matching the tested source
(`a5f3d89361abe1f1a35486ab55dfb3c388cd7ab9`), at replay commit
`1a69a0d61d0b6961d2016ef7ba80e1c4d7fb2c26`.
[Source metadata](source.json), the [source lock](upstreams.lock.json),
and a [launch-disabled candidate profile](candidate-tp4.json) describe it; the
parent's [release-test classification](../release-tests.json) still gates the
same test file, now with 27 cases. Candidate configuration validation passes;
release-test classification reports 128 selected files, 71 gates and zero
problems. Whitespace checks pass, and the four bundle files match both the
committed source and the GPU receipt hashes. A full-model serving A/B and release image
remain outstanding. No serving image is promoted by this correction.

The test window closed at 12:22:21 UTC on 24 October 2026 after production was
restored and its live health check passed; the cluster hold was removed.
