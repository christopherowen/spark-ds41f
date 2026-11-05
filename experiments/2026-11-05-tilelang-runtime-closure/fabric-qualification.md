# Physical networking qualification

The four 43-layer traces use one GPU and TP3/TP4-shaped layers. They prove the
recorded request's local compute inventory; they do not exercise physical peers.
Bulk collectives still select NCCL. Sparknet's small-message TileLang kernels
have IKET hooks, but that does not instrument NCCL's prebuilt device kernels.

The Sparknet adapter currently limits reduce-scatter, exchange and row all-to-all
to one slot. Its existing multi-slot one-shot path is not a qualified replacement
for bulk NCCL: the recorded design measurements show a bandwidth disadvantage.
Removing the remaining communication kernels needs a separately measured bulk
algorithm, rather than a policy change that enables that known slower path.

## Required matrix

- Resolve the candidate-38 triangle and ring profiles against the real site map;
  record the exact deployment commit, source manifest, image digest, node map,
  cable/interface map, driver and library versions on every rank.
- Exercise all-reduce, all-gather, reduce-scatter, exchange and row all-to-all at
  TP3 and TP4. Include 4 KiB, 64 KiB, each slot boundary minus/at/plus one aligned
  word, 1 MiB, 8 MiB and 64 MiB payloads, for every supported payload dtype.
- Check opaque copies byte-for-byte, rank ordering and output ownership. Check
  reductions against the declared rounding/order contract, with graph replay,
  repeated buffer reuse, delayed peers and multiple slot wraps.
- Pair each candidate case with its fixed-library component control in mirrored
  role order. Boot twice, retain every run/error and require each five-standard-
  error upper ratio to be at most one. Reuse recorded released-build benchmarks;
  do not rerun a released build.
- Collect CUPTI launch inventories on every rank and IKET ranges for each new
  TileLang communication family. Attribute remaining NCCL launches explicitly;
  fail the zero-external-compute claim if any remain in the request window.
- After component qualification, run the recorded full-model workload on both
  physical shapes, preserving token/logprob checks and memory observations.

## Window and execution boundary

The current authorization covers idle sp8. It does not cover changing either
serving cluster. Before physical tests, acquire an owner-authorized peer window
through the existing hold/request protocol. Source preparation and builds stay
remote, with two build workers and one coordinated job per node. Use the
repository's coordinated cluster commands, inspect their live/shape plans, and
name the cluster configuration on every apply. Every rank must use the same
published clean commit and OCI digest; no one-rank replacement or serving overlay.

No promoted configuration or immutable baseline is changed by this plan. A built
image, real-weight quality/throughput and the physical networking matrix remain
release gates after the single-GPU cleanup checks.
