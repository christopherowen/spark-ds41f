# Replay captured C36 operands on idle dgx4

Hardware execution remains pending. `candidate.json` selects the exact C36
image. Its indexer overlay is C36's freshly reproduced source with only the
independent top-k scratch guard from PR #878 applied. It does not mount the
1030 indexer into C36, and contains no input-capture hook. `check_sources.py`
reverses the guard to verify the exact recorded C36 source hash.

The harness compares every selected key and each cluster bound with the
existing scorer, then times only the complete candidate. It copies only
visible scores for the bound audit, preserving unwritten-tail discipline.
Raw activations stay outside Git under the existing node-local input mount.
No replay, real-model pruning or output-quality claim has been made yet.
