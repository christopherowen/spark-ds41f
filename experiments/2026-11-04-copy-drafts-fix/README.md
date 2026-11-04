# Useful CopyDrafts continuations and exact search pruning

Base: deployment 6a97c71d, candidate-38 image and vLLM identities in base.json.
The candidate is opt-in; no promoted lock, recipe or baseline is modified.

Two ordered, independently measured variables:

1. Rank equal-strength suffix matches by continuation length capped at lookahead,
   then recency. Preserve a short copy when no full continuation matches at that
   level. Encode priorities in the existing int32 source buffer; both allocation
   and selection decode the same priority. Include lookahead in warmup keys and
   allocation in the bundled pipeline benchmark.
2. Visit candidates in descending priority and stop a warp after a longest-level
   hit, which also dominates all remaining candidates at every shorter level.
   This keeps exactly the first patch's choices, with no bounded window, index,
   new buffer or host synchronization. Bounds and votes are uniform per warp.

Apply series after candidate-38 vLLM head 1f33102547e4da5c64712390604695d1abd577c4
(upstream 19f2c20ed4d64ae22bd009ceacadd85e2a81e354 plus its pinned series).
Patch bodies state upstream base, rationale, quality implications, tests and
upstream status. Both are local candidates, not submitted upstream.

Each candidate boots twice for correctness and twice for the same 96-case,
21-sample cold-L2 graph timing matrix as the prior recorded assessment. Compare
with those recorded candidate-38 runs, without rerunning the released baseline.
Record every run, error and memory observation. Only idle authorized sp8 is used;
serving clusters are not started, stopped, synced or restarted. A clean published
checkout supplies patches; each throwaway test container first verifies original
source hashes, applies patches, then verifies all changed file hashes before
importing vLLM. These overlays are kernel tests, not a serving image build.

Acceptance: independent reference outputs, full and short continuation boundary
cases, multiple lookaheads/minimum-match levels, per-request/slot isolation,
allocation agreement, cap growth/shrinkage, CUDA graph replay, committed-text
invariance and no errors. Preserve every unfavorable timing. No real-model
acceptance, distributed performance or release qualification is inferred.

Results pending.
