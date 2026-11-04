# Decision

Status: retain for measurement. No performance win is established by this
experiment yet and no variable is approved for release.

- FP8: CPU reference codecs and real-tile collector prepared. Require actual
  tile ratios, metadata costs, then a fused GPU consumer and serving receipts.
- Expert scales: reuse the existing exact compaction design in an isolated
  promoted-base arm. The known persistent-memory saving does not establish
  faster decoding. Two boots and unchanged kernel/output bits remain required.
- Engram: reconcile existing packed-reader evidence and later page-layout fixes;
  measure a coherent isolated layout, including packing time and disk space.
- Cold prefix: bounded disk storage prototype prepared. A complete-state export,
  all-rank restore barrier, vLLM adapter and reuse-after-eviction measurement are
  outstanding. Storage round trips do not qualify continuation correctness.

The active cluster window belongs to another experiment. No serving operation,
GPU job or target-host performance measurement has been started for this work.
