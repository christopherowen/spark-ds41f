# Decision

PR #850 closed as unnecessary to the current runtime work; other unique PRs retained.
Candidate-39 default CopyDrafter source is merged in #842. No new GPU or performance verdict
exists yet. Keep source preparation, image tests, correctness and measured core
metrics distinct. Spaceport is unavailable to management checks and must not be
used for serving changes until access is resolved under the hardware rules.

The first dispatch yielded to the clustered-indexer request before GPU work.
Candidate-36 remained healthy. Preserve that attempt and wait for the reservation
and request queue before resuming with a distinct attempt label.
