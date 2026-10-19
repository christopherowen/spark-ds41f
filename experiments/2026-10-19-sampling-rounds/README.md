# Block verification: fresh draws across rounds

Backport the upstream round-seed repair (#61161) and residual lookahead bounds repair (#60945) onto r6x. Temperature-zero output must remain identical; seeded sampled sequences intentionally change to recover the target distribution.

Candidate source fd1b57e06 on r6x f3b58ed9d. The kernel bundle runs the multi-round unbiasedness regression, existing block/standard verification tests, and the repeated-position DSpark test. No serving configuration changes.
