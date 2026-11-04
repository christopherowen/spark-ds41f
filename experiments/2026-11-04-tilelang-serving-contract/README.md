# Remove unused DS4.1 serving alternatives

Deployment base: `9799b37cb80f4e90bf397fb9769a95c04fd5f76f`.
Source base: candidate-39 vLLM `31412339aafdd2c1392c7294975355dbc523c5e1`,
tree `8e33015b1fa1d34250e686db7d1f8061021d6d5a`.

This implements the owner's first three requested cleanup items as three
independent patches, keeping the selected production path in each:

1. Sequence-parallel MoE requires the producer's MXFP8 cast and a local router.
   Missing inputs fail before any collective. The BF16 gather/recast and
   missing-router alternatives existed for CPU stand-ins. The simulations now
   supply the cast/routing interfaces and grouped gather used by the native
   path; their real-valued stand-in arithmetic remains separate from GPU
   packed-byte tests. The native grouped gather and route arithmetic are kept.
2. Delete `VLLM_DS41_ENGRAM_OVERLAP`, which has no consumer. Engram scheduling,
   overlap and kernels are unchanged.
3. Remove `VLLM_DS41_MARKOV_NVFP4` and `VLLM_DS41_DRAFT_NVFP4_HEAD` from the
   DS4.1 model. Preserve the existing default lossless Markov-head selection
   and target-shared vocabulary head. Generic vLLM NVFP4 support for other
   models remains. The [restoration patch](optional-heads/restore-nvfp4-heads.patch)
   preserves those alternatives for a separately qualified experiment; it is
   not in the serving series and has no promoted profile.

The candidate rejects all three retired settings at startup, including explicit
`0` values. Doctor applies the same rule only when the selected source manifest
records `ds41-serving-contract`; frozen source recipes retain their old support.
This does not change the separate FP32/model head precision choice, packed-head
policy, padding policy, numerical correction branches or shape-specific kernels.

The isolated [TP3 profile](tp3.json) carries the cleanup over candidate-39. It
does not revise the published candidate-39 definition or promote another
candidate number. It is independent of the frontend memory fixes in PR #890;
neither change substitutes for the other.

## Validation protocol

No serving operation or DGX experiment is authorized here. Use idle Spaceport
sp8 for bounded CPU-only tests after preparing the correction. Export a
published deployment commit, use the exact image in [base.json](base.json),
and verify every original, patched and test byte. The changed source modules
are byte-identical between the candidate-38 test image and the candidate-39
base; their hashes are recorded individually.

The [controller](run_spaceport.py) verifies host identity, idle containers,
reservation and headroom; applies only the three published serving patches on
the host; and runs disposable containers with two CPUs, 8 GiB memory, no network
and no GPU devices. The [test runner](run_tests.py) installs the same updated
tests in both controls, with source changes only in the corrected control.
It records successful collection, exact test count, JUnit, native pytest exits,
and all output. Collection errors or skips cannot count as success.

Expected original result: ten failed assertions (retired-option contracts and
missing-MoE-input rejection), with the remaining cases passing. Expected
corrected result: every collected case passes. Keep all unsuccessful attempts.
The suites exercise CPU row partitions/collectives, full-row model simulations,
retired-option rejection and target-head sharing; they do not qualify GPU
arithmetic or end-to-end serving performance.

Exact isolated controller invocation after publishing `<commit>`:

```
python3 run_spaceport.py --commit <commit> --apply
```

Before review, replay the complete ordered series in a fresh named source
worktree; verify head/tree, doctor, experiment-local release-test classification,
repository tests and whitespace. Run `build check` only when complete build
inputs exist. Preserve setup failures, all raw controls and source identifiers
under `runs/`. New image, full release gate, two-boot serving qualification and
performance comparison remain necessary before promotion; no old measurements
are assigned to the new source.

Upstream status: local, not submitted. No promoted configuration, immutable
baseline or serving cluster is changed.
