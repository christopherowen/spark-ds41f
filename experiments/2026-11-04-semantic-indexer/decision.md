# Decision

Retain the semantic grouping, coupled-bound and projection diagnostics as
research. Keep the current native runtime and PR in draft. No model weights,
attention scores, output rule, image or baseline is promoted.

This round fixes the previous zero-pruning failure on dense operands. Practical
seed cutoffs and reuse leave15.39–58.26% of dense keys to evaluate once. All48
native selection checks and12 source block/count/reindex checks pass across two
fresh processes. The native scores used for seed accounting come from an untimed
full oracle; this is not an executable sparse kernel benchmark. The source
handoff retains99.85% of keys with this seed policy, and native eight-key
expansion removes more of the individual-key savings.

The cost gate remains unsatisfied. Rank1/2/4 shapes lose most pruning. Full
seven-vector shapes require at least4096 FP32 center/factor bytes per eight-key
group, before bounds and residual metadata, against544 native packed-key bytes.
The diagnostic also uses query-dependent slope optimization and fresh metadata.
A production implementation needs cheaper certificates, actual seed-score reuse,
individual-key gathering where useful, explicit cache ownership and a qualified
write/invalidation protocol. Persisting the current expensive shape does not by
itself establish a net saving.

Compact key maps through32 dimensions retain every key. Rank64 saves at most
0.42%; rank96 helps one dense sample but already exceeds the native storage per
key. Distinct-query fitted maps through64 dimensions also retain every key. The
initial positive source fit reused the evaluated query and is excluded. Preserve
both initial and corrected evidence. These finite screens reject the tested
designs, not every possible exact search or trained auxiliary map.

The corpus is one public-source prompt, nine dense snapshots and repeated
source queries across three ranks. Execution mode was not recorded. Fixed-input
mask equivalence does not qualify a full serving implementation, persistent
metadata, teacher-forced output distributions, varied prompts, or request speed.
Do not substitute deterministic pruning counts for latency confidence intervals.

All seven GPU jobs passed within their memory/lease/duration guards. C37 was
restored coherently on sp5–sp7 at18:38:25UTC, followed by independent live and
per-node verification. Configured experiment checkouts advanced to3b8397f0;
original806c5797 checkouts and the original C37 image/model/settings remain.
sp8's foreign request and the DGX qualification were left intact. No active
runtime window remains from this work.
