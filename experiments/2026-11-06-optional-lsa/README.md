# Optional non-quantized BF16 TileLang LSA across 32K-1M

Base `bdb4a2fbb80ce41b8ecda995dac8da6e7b53fa3c`. Makes TileLang LSA a strict opt-in choice via `VLLM_DS41_LSA=1` with zero overhead on the default path (`state_content_bytes = 68`, zero extra declared kernels or workspace scratch when unset or `0`), removes internal sub-optionality and downward-quantized `compact8` summaries from the runtime path in favor of exact BF16 `block8` summaries (`STATE_BYTES = 100`, FP32 accumulation), upgrades `summarize_blocks` to the 512-thread cooperative kernel with `pdl.chain()`, wires `RawWideCoarseTopK` (`budget = 1024`) and `DirectCoarseTopK` (`budget = 512`) plus full PDL chaining into `QueryReusePlan`, and enables unified `score_width` routing across `32K-1M` prefill and multi-row decode:

1. **Default-off bit-identity (`VLLM_DS41_LSA` unset or `"0"`)**:
   - Admits `state_content_bytes = 68` and builds only `IndexerV41Plan` (`native_index = {}`, `source_short = None`, `_page_summary = None`).
2. **No downward quantization in opt-in LSA (`VLLM_DS41_LSA="1"`)**:
   - Single clean BF16 `block8` summary representation (`STATE_BYTES = 100`): 8-key BF16 means accumulated in FP32 and scored via `QueryReusePlan` (`prepare_queries` + `prepared_scores`) in BF16 with FP32 accumulation.
3. **Unified `32K-1M` prefill and multi-row decode coverage**:
   - Dense compressed layers (`2, 8, 14`, `compress_ratio = 2`): engages `QueryReusePlan(16384, 128, budget=64)` on `prefill_short` (`16K-32K` tokens, `4x` key reduction at `32K`) and `QueryReusePlan(524288, 128, budget=64)` on `64K-1M` tokens (`8x-128x` key reduction) across both prefill and multi-row decode (`rows >= 2`).
   - Candidate source layer (`20`, `compress_ratio = 1`): routes `<= 32K` (`score_width <= 33,023`, where all keys survive) to `native_index`, engages `source_short = QueryReusePlan(1048576, 256, candidate_blocks=True, budget=512)` (`DirectCoarseTopK`, `2x` key reduction) at `64K` (`33,024 <= score_width <= 65,791`), and engages `QueryReusePlan(1048576, 256, candidate_blocks=True, budget=1024)` (`RawWideCoarseTopK`, `2x-16x` key reduction) at `131K-1M` across both prefill and multi-row decode (`rows >= 2`).
   - Single-row decode (`rows == 1`): stays on `native_index`.

Fresh 14-patch source application (`runs/local/source-check.json`), CPU routing/AST verification (`runs/local/cpu-routing-check.json`), all 7 standalone GPU correctness gates (`sp8-summary-initcheck`, `sp8-summary-racecheck`, `sp8-summary-timing`, `sp8-registration`, `sp8-actual`, `sp8-initcheck`, `sp8-graph`), and 2-boot benchmark runs (`sp8-bench-boot1`, `sp8-bench-boot2`) are recorded in `runs/gpu-gates-01/`, `summary.json`, and `results.md`.
