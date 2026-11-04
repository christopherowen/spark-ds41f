# Decision

Status: retain for GPU/serving measurement. Exact storage reductions are
measured on checkpoint samples; no serving performance win is established and
no variable is approved for release.

- FP8: 44 real N128/K128 checkpoint tiles, 720,896 bytes, two independent CPU
  processes with five observations per block and mode. Every reconstruction is
  byte exact. On the 40 dense tiles, including frame overhead, raw LZ4 and
  field-split LZ4 save no payload (RAW selected); bitplane-LZ4 saves 5.956%,
  exponent Huffman 16.426%. On the four Engram table tiles the latter two save
  7.698% and 16.711%. The sample is not a whole-checkpoint compression ratio. Retain
  transformed/entropy encodings for a parallel GPU decoder and fused-consumer
  screen; plain LZ4 is rejected for the sampled hot FP8 representation.
- Expert scales: reuse the existing exact compaction design in an isolated
  promoted-base arm. The known persistent-memory saving does not establish
  faster decoding. Two boots and unchanged kernel/output bits remain required.
- Engram: reconcile existing packed-reader evidence and later page-layout fixes;
  measure a coherent isolated layout, including packing time and disk space.
- Cold prefix: bounded disk storage prototype prepared. A complete-state export,
  all-rank restore barrier, vLLM adapter and reuse-after-eviction measurement are
  outstanding. Storage round trips do not qualify continuation correctness.

Another experiment held the cluster during preparation. At 14:31 UTC on
2026-11-04 its hold was absent. The standard lab kernel job stops the serving
cluster, so target work still requires the explicit operation authorization
specified by AGENTS.md. No serving operation, GPU job or target-host performance
measurement has been started for this work.
Only a bounded read-only checkpoint export ran on dgx4; codec timing ran locally.
Native receipts are losslessly archived under `runs/fp8-local-*.json.gz` and
their original JSON hashes are recorded in `results.json`. No weight bytes are
committed.

The original collector grouped every Engram-named tensor together, including
its GPU projection. `analyze_screen.py` corrects that grouping from the recorded
tensor names without editing either original receipt. The collector's version-2
classification keeps disk tables and GPU projections separate for future exports.
