# Exact raw wide coarse selection

Base `f48423d50b97878f5c74bc2637ec60dbd7da8d5f`. One intended variable relative to the source1024/dense64/native-generation recipe: replace the all-entry4096-key segment pass and following select-all8192-key merge with a raw16384-key to8192-key selection. Remaining exact wide merges are unchanged. The first useful selection sees the same entries and retains the same BF16 thresholds, score bits and ascending-entry ties. The full-width coarse route removes two launches and reduces its ping-pong scratch by four; these are structural counts, not measured request-speed gains.

The parent answer comparison is still running. This optimization cannot repair a parent selection-quality loss. Retention, cache geometry, score arithmetic, phase policy, weights and final-answer protocol remain unchanged. Independent prior-route/stable-sort, strict initialization/race, actual native-score, declared-call and graph checks must pass from a clean published commit. Two fresh component timing processes and independent model request/answer gates remain required before any speed or quality claim.

Fresh twelve-patch application equals all affected overlays. Commands and source hashes remain in `runs/local`; preparation uses one CPU/4 GiB and one worker remotely. Complete image-build inputs are unavailable. All reference calculations remain outside serving, raw operands remain private on nodes, and no production promotion is claimed.

Adversarial checks cover24 batches at8193/16384/32768/131072 widths, each with12 lengths crossing4096/8192/16384 boundaries, ties/mixed scores and active-key caps. Both the prior generic route and an independent stable sort must match indices and BF16 score bits. Unwritten suffixes remain unwritten under initialization checking; six quick batches also run under race checking. Coarse scratch bytes are checked independently at full width.

Two fresh timing processes use every frozen source capture at1/8/64 rows. Identical coarse score matrices are generated once outside timers; the generic route brackets the raw route with CUDA graph samples. Output comparisons, score preparation, hashing and mean construction are outside timers. These measure only coarse selection on captured prefill queries, not actual decode queries, whole attention, maintenance or request performance.

Component checks and two fresh timing processes now pass; see `results.md`. Full-request and final-answer qualification remain separate.

The repeated request screen preserves26/26 answers on both candidate boots; see `model-results.md`. Its stock-fan timings are exploratory. The documented performance fan curve is now mandatory for future timed clients and recorded-control reuse. No promotion.
