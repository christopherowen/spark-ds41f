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
