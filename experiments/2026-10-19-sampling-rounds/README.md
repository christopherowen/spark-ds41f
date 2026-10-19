# Block verification: fresh draws across rounds

Backport the upstream round-seed repair (#61161) and residual lookahead bounds repair (#60945) onto r6x. Temperature-zero output must remain identical; seeded sampled sequences intentionally change to recover the target distribution.

Candidate source fd1b57e06 on r6x f3b58ed9d. The kernel bundle runs the multi-round unbiasedness regression, existing block/standard verification tests, and the repeated-position DSpark test. No serving configuration changes.

## GPU kernel gate, 2026-10-19 (DGX4)

On the promoted r6x image, the new multi-round regression fails both block modes (deterministic and probabilistic drafts), while both standard modes pass: 2 failed, 2 passed. With candidate fd1b57e06 mounted, all 74 sampler/DSpark tests pass. This is a reproduced baseline failure and a passing repair, not a full-model release gate. Temperature-zero determinism and sampled serving throughput remain to be measured.

`model-gate.json` measures the repair as three runtime source overlays, with no tests in serving: candidate, reference, candidate on TP3. It runs screen-tp3 and sampled as a companion suite, including teacher-forced scoring against the first arm. This is a screening gate; full release and determinism gates remain required before promotion.
