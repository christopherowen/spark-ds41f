# Coarse score tile geometry

Base `9da97a31`, same C36 image, native coarse arithmetic, compact80 mean cache,
budgets, producer and fine selection. One intended variable: score64 means per
CTA instead of16/32, so each CTA covers multiple complete physical mean pages.
The last partial tile clamps only inactive page-table addresses to the last
visible mean; its loads/stores remain masked. This prevents reading unwritten
page-table suffixes when a tile crosses the last complete page. Full tiles keep
their affine positions.

This is an out-of-band screen, not a serving recipe. Qualify the same 12 saved
batches against exact score bits and native fine arithmetic, strict sanitizer
fixtures and six lifecycle/graph cases before timing two fresh processes. All
old tiles, physical IDs and replay hashes remain unchanged. No model response
or held-out outcome informs the tile choice. Preserve every run/error/sample,
memory observation and sp8 lease release in `runs/`. No model-quality or request
performance claim can follow from this screen alone.

The first screen at `5e4eb543` stops at Docker's duplicate-mount validation:
the predecessor's unused probe and the new probe both target `/tmp/probe.py`.
No container or kernel starts. Its original commands/error and released lease
remain in `runs/screen-01`. The retry removes only the obsolete probe mount;
every bundle's mount destinations are checked for uniqueness before publishing.
The score factory, arithmetic and frozen operands remain unchanged.

The corrected screen at `7b631ce1` passes all12 actual batches, four strict
initialization fixtures with zero sanitizer errors and six graph/lifecycle
cases. Two fresh complete-selector timing processes give native/LSA ratios of
1.76–1.83x dense8, 2.68–2.70x dense64, 1.61x source8 and1.92x source64.
Single rows remain slower and dense single-row variation is large. The small
dense improvements and slight source regressions relative to the preceding
compact screen do not establish a uniformly better tile choice. No serving
recipe adopts this factory; the running model cohort retains its qualified
16/32-mean tiles.

Every sample, allocator observation and conditional two-process interval is in
`runs/screen-02`. A missing interval means a small-sample time bound crosses
zero, not a pass. The sp8 lease releases at19:47:51 UTC, with no remaining
containers; an independent read confirms no hold or pending request. A later
monitor-only SSH banner timeout resolves on a bounded read; it does not rerun
or change any GPU job. No request-speed or final-answer result is claimed.
