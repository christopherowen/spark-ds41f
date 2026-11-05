# Decision

No variable qualifies for release. Reject all ten measured hot-FP8 layouts
(three Huffman, three fixed-field and four independent-plane LZ4); retain the
isolated expert-scale and Engram arms for target measurement.
Cold prefix storage needs a model-compatible restoration path before timing.

- FP8: all six fresh GPU containers reconstructed every sampled weight byte and
  special encoding exactly. For 32/64/128 lanes, compressed/raw first-repeat time
  ratios are 1.633/1.633, 1.848/1.860 and 1.815/1.810. Exploratory 95% bootstrap
  intervals are [1.629, 1.637], [1.838, 1.870] and [1.790, 1.819]. Real input
  storage savings are 11.839%, 10.385% and 7.477%, counting metadata and padding.
  Decode cost exceeds saved traffic in this standalone shared-memory consumer.
  This is neither GEMM nor serving performance. A cheaper exact representation
  would need a new paired GPU screen, then unchanged GEMM arithmetic and two
  serving boots. Plain LZ4 selects RAW on the sampled hot FP8 tiles. The six
  fixed-field runs also reconstruct exactly, save 8.687% and take 1.362–1.912
  times their paired raw duration. The eight independent-plane LZ4 runs save
  5.232–6.299% and take 1.740–4.698 times their paired raw duration. Every
  first/repeat ratio interval is above 1. The families use different consumer
  widths, so their absolute times are not a cross-family comparison. Complete
  native evidence is retained in each family's `gpu-results.json` archive map.
- Expert scales: reuse the existing exact compaction design in an isolated
  promoted-base arm. Prior persistent-memory observations are hypotheses for this
  arm. The corrected full gate passes on all three ranks (7,337 passed, 355 skipped
  each). The complete initial primary matches all 256 recorded generated sequences
  and 65,536 returned score vectors exactly. Its newer stream collector prevents
  treating it as the matched 1030 performance comparison. The deliberate handoff
  restored C36; all observations and errors remain in
  `expert-scales/first-primary-results.json`. Unchanged scale words, kernel/output
  bits, two matched serving boots and the remaining functional gates are required.
- Engram: the packed-row arm now rejects negative column deltas before signed
  subtraction. The portable C fixture reproduced the old overflow and passed
  the corrected guard plus 1,025 valid rows. Native reader tests, packing time,
  disk space, direct-I/O waits, memory and two serving boots remain outstanding.
- Cold prefix: the bounded disk prototype is byte exact. The promoted-source
  compatibility audit excludes immediate use of the generic recurrent adapter
  for this model; the filesystem tier's host-local shared mapping does not
  certify multi-host rank persistence. A model-specific checkpoint or complete
  prefix-block replay adapter, all-rank restoration barrier and forced-eviction
  continuation proof are required. Storage round trips do not establish these.

## Evidence and workload identity

The original CPU screen contains 44 N128/K128 checkpoint tiles (40 dense, four
disk tables). It remains unchanged in `results.json` and
`runs/fp8-local-{a,repeat}.json.gz`. The original classifier grouped Engram
projections with tables; analysis corrects that grouping from recorded names.

Version 2 selects 46 tiles, including two additional `layers.1.engram.wkv.weight`
projection tiles. All 42 dense identities match every GPU run. The separate
two-process CPU reference `fp8/cpu-v2-results.json` gives 5.870% bitplane-LZ4 and
16.274% Huffman savings with its CPU frame. That frame differs from the GPU
frame; the sample is not a whole-checkpoint compression ratio. No source weight
bytes enter Git. Native receipts, hashes, exact GPU commands, memory observations,
all nine timing rounds and both process observations are preserved under
`runs/fp8-gpu-artifacts/` and summarized in `fp8/gpu-results.json`.

The authorized FP8 window `dgx1-1793809624` completed its six-run matrix, restored
Candidate 36 at 16:44:47 UTC on 2026-11-04 and released the hold. Restoration and
subsequent live doctor passed with the pre-existing wireless-management warnings.
A different experiment holds the cluster during follow-on source preparation.
No promoted configuration, immutable baseline or reference run is changed.

`runs/follow-on-audit.json` preserves preparation errors and their resolutions.
