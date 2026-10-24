# Selected-token vocabulary ranks in TileLang

Base: deployment `99e4e9e3`; vLLM `174a746f4577579841e5aa3e94c8a4964b0aa326`
(release candidate 4, tree `97fc3579def4ffc32eec88c9149c560559af59d4`).
The only runtime change is the V2 runner's selected-token rank calculation in
`vllm/v1/worker/gpu/sample/logprob.py`. Its log-softmax, token selection and
generation are unchanged. The original Triton rank kernel remains as the test
and benchmark reference; its runtime dispatcher and warmup are removed.

The candidate accumulates the comparison counts in thread-local fragments across
the vocabulary, then reduces once. Triton reduces each 8,192-element chunk.
Ranks count logits greater than or equal to the selected token's logit, including
ties. NaNs, infinities and the old kernel's negative-infinity padding behaviour
are preserved. Rank arithmetic is integer and independent of batch size.

The owner's startup registration declares every logit dtype/stride and both token
ID types through the existing TileLang preparation registry. A missing call after
the cache freezes remains an error. This change adds no backend switch.

## Qualification

The kernel bundle uses `spark-ds41f:19f2c20ed4d6-next-4`, image
`sha256:1ae57dac1017258128b46665308dd490f1284fb1dd6926dd2c1a3685ea04b82d`,
on DGX4. It mounts only the changed module, its call site, the focused GPU tests
and the paired benchmark. The production checkout and recipe are unchanged.

Correctness covers FP32, FP16 and BF16; full and partial vocabulary chunks; padded
row strides; integer token IDs of both widths; ties, signed zero, subnormal logits,
NaNs and infinities; batch independence; CUDA-graph replay; the public
`compute_topk_scores` result; and startup preparation.

Timings compare warmed CUDA graphs of 100 calls, alternating order, using the same
input and output buffers. Eight fresh placements supply the paired 95% interval.
Serving shapes use DS41F's vocabulary 129,280 and 1, 8, 16, 48 and 96 rows (up to
16 requests with one target token and five drafts each). FP32 is the serving
logit format; FP16 and BF16 are covered as supported inputs. The initial run
compares 128, 256 and 512 threads. The final choice is 256 for FP32, 512 for the
two half formats, independent of batch size. A separate seed and allocation
sequence confirm the timings in a repeat.

Offline lowering and nvcc compilation for SM121a succeeded for all three logit
dtypes. Generated loads were inspected for bounds checks on the last partial
chunk; automatic safe accesses remain enabled.

## Results (2026-10-24, DGX4)

Both final runs pass all 24 focused GPU tests, and all 24 pass under CUDA 13.0
compute-sanitizer memcheck with **zero errors**. The tests use the public
`compute_topk_scores` path as well as direct kernel comparisons. This is kernel
qualification in the DS41F image, not a full-model serving A/B.

The main FP32 cases (first accepted run, median microseconds per call; percentage
from the paired mean differences):

| Logit rows | Triton | TileLang | Time change |
| ---: | ---: | ---: | ---: |
| 1 | 16.94 | 7.72 | -54.5% |
| 8 | 17.19 | 7.92 | -53.9% |
| 16 | 17.31 | 8.32 | -51.8% |
| 48 | 50.91 | 38.30 | -22.5% |
| 96 | 168.48 | 168.04 | -0.27% |

The repeat gives -54.4%, -54.0%, -51.8%, -25.7% and -0.32% respectively. The
largest FP32 case streams about 50 MB per call and is bandwidth-bound. The
half-format cases at 1-16 rows take roughly half the reference time as well.
All 15 dtype/row shapes pass the unchanged paired performance gate in both
accepted runs. Full arrays, intervals and input fingerprints are in
[results.json](results.json), with native outputs under [runs/](runs/).

All attempts are retained:

- `initial`: all 24 tests pass; the 128/256/512-thread sweep has no slower shape.
- `fixed-512`: all tests pass, but the universal 512-thread choice is 0.56%
  slower on FP32 at 96 rows (paired interval +0.44 to +1.45 us). Rejected. The
  image also lacks the compute-sanitizer executable, so that attempted invocation
  fails. Later runs bind the host's CUDA 13.0 sanitizer directory read-only.
- `qualified`: the dtype-dependent thread counts pass all tests, all 15 timing
  cases and memcheck. The old lab parser nevertheless marks it failed: it selects
  the sanitizer's `ERROR SUMMARY: 0 errors` as the pytest summary. Exit status is 0.
- `repeat`: identical source, a fresh benchmark seed, sanitizer first and the real
  pytest run last. All checks and the existing lab verdict pass.

The serving service was restored on all four nodes; `doctor --live` reported OK
at 12:05:43 UTC and the lab hold was removed.

## Reproduce and integrate

The source change is [vllm/0266-tilelang-logprob-ranks.patch](vllm/0266-tilelang-logprob-ranks.patch).
[vllm/series](vllm/series) is candidate 4's complete series plus that patch. All
223 patches apply to pinned upstream `19f2c20ed4d64ae22bd009ceacadd85e2a81e354` in
a fresh disposable worktree and reproduce the GPU-tested source tree
`9b364c7d31913e474515d5168bb9e7a41074be14`; the replayed patch head is
`ea05c1c0295739d8b124bb04a221914c56809e0c`.

[source.json](source.json), [upstreams.lock.json](upstreams.lock.json) and the
launch-disabled [candidate-tp4.json](candidate-tp4.json) describe the build.
The experimental [release-tests.json](release-tests.json) adds this port's tests
to the existing release gate (128 selected files, 71 gate, no classification
problems). The deployment repository's 475 unit tests pass (five skipped), its
doctor reports configuration OK, and whitespace and Ruff checks pass.

The runnable [bundle/](bundle/) mounts just this change over candidate 4. Its
commands run the sanitizer, the paired benchmark gate and the focused GPU tests;
the sanitizer mount requires the host CUDA 13.0 toolkit. Source snapshots and
fingerprints for the earlier attempts remain with their native receipts.

No new image was built and no production promotion was performed. A full-model
serving comparison remains a release gate before this enters the promoted image.
