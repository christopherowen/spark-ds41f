# Carry the ownership correction; retain the field-memory question

Keep the request-submission rollback as an isolated candidate-39 source patch.
The original candidate-38 source demonstrably retains frontend requests after
failed/cancelled submissions, including partial sampling children. The corrected
source passes the same ten Spaceport CPU regressions, while the original fails
the eight failure/cancellation cases. No live service was modified.

Do not close #884 or claim its relative candidate-38/candidate-36 memory drift
is explained. The same host defect exists in both candidates, and the field
run did not identify a growing process or count cancellations during submission.
Further retained allocations may exist.

Do not promote the new recipe on these unit results. A new candidate image,
full gates and any serving/performance work require their own source identity
and evidence. The owner restricted this investigation to read-only work until
a defect was corrected, then authorized tests on Spaceport. No DGX experiment,
serving restart, guard change, reduced context or unrelated patch reversal was
performed here.
