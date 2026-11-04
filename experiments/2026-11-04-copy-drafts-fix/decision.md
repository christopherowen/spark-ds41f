# Retain as candidate

Keep both ordered source patches for candidate-38: useful continuation selection
and exact search pruning. The isolated kernel/proposer gates pass on both boots,
with large repetitive-history timing gains and small retained regressions in some
other cases. Results and confidence intervals are in results.json.

No promotion or serving deployment. Build-image checks, real-model distributed
acceptance/throughput and IKET compatibility of the new loop remain unqualified.
The optional IKET attempt could not start because post-test Spaceport SSH banner
exchange timed out; the completed test window had already released sp8.
