# Default `CopyDrafter` for DSpark Greedy Adaptive Verification (2026-11-01)

## Motivation ("One True Way")

Previously, `SpeculativeConfig.copy_drafts` defaulted to `None`, requiring an explicit `"copy_drafts": {"min_match": 16, "initial_drafts": 8, "max_drafts": 16}` entry in `--speculative-config` plus manual widening of `--max-num-batched-tokens`, `--max-cudagraph-capture-size`, and `cudagraph_capture_sizes` in a separate overlay arm (`tp3-copy.json` / `copy-tp4.json`).

In Candidate 18 (`rc18`) qualification (`49/49` sessions passed), `CopyDrafter` (`min_match=16, initial_drafts=8, max_drafts=16`) demonstrated:
- **100% Bit-Exact:** Identical SHA-256 digests across all greedy, logprob, 16K SP, Vision, and seeded-sampling gates.
- **No Regression on Non-Copy Traffic:**
  - Engine decode grid: `-0.41% ±0.16%` faster
  - `c8` paired decode: `+0.43%` tok/s
  - `c1` paired decode: `+0.19% ±0.16%` faster (`+0.27%` overall)
  - `sampled c1` paired decode: `+0.23% ±0.06%` faster
  - `sampled c8` TPOT p99: `-2.89%`
- **Up to `6.70x` Higher Acceptance on Copy / Edit / Tool Workloads (`copy_probe.py`):**
  - Aggregate `tokens/step`: `2.350 -> 9.185` (`3.91x`)
  - `copy`: `2.39 -> 16.029 tok/step` (`6.70x`)
  - `edit`: `2.38 -> 14.249 tok/step` (`5.99x`)
  - `tool`: `2.15 -> 6.709 tok/step` (`3.12x`)

Following the **"One True Way"** design principle, `CopyDrafter` is now enabled by default whenever DSpark runs with greedy adaptive verification on the async V2 model runner, with automatic decode-row and CUDA-graph ladder widening.

## Design

1. **Default `CopyDraftsConfig` in `SpeculativeConfig.__post_init__` (`vllm/config/speculative.py`):**
   - When `self.copy_drafts is None` and `self.method == "dspark"` and `self.enable_adaptive_verification` and `self.draft_sample_method == "greedy"`, default `self.copy_drafts` to `CopyDraftsConfig(max_drafts=max(CopyDraftsConfig().max_drafts, self.num_speculative_tokens))` (`min_match=16, initial_drafts=8, max_drafts=16`).
   - In `VllmConfig.__post_init__` (`vllm/config/vllm.py`), if `copy_drafts` was defaulted and a runtime requirement (`async_scheduling`, `use_v2_model_runner`, `pipeline_parallel_size == 1`, `not use_request_boundary_checkpoints`) is not met, clear `self.speculative_config.copy_drafts = None`.
2. **Automatic Decode-Row & CUDA-Graph Envelope Widening in `VllmConfig` (`vllm/config/vllm.py`):**
   - In `_set_max_num_scheduled_tokens`, when `self.max_decode_rows > base_decode_rows` (`max_num_seqs * uniform_decode_query_len`) and `align_prefill_chunks` is enabled with `long_prefill_token_threshold` sized for `base_decode_rows`, automatically widen `max_num_batched_tokens = long_prefill_token_threshold + self.max_decode_rows` (`6144 -> 6155` on TP3, `16384 -> 16395` on TP4) while preserving `long_prefill_token_threshold` (`6096` / `16288`) so aligned prefill chunk boundaries remain 100% unchanged.
   - In `_set_cudagraph_sizes`, when `self.max_decode_rows > base_decode_rows` and `max_cudagraph_capture_size == base_decode_rows`, automatically widen `max_cudagraph_capture_size = self.max_decode_rows` and extend `cudagraph_capture_sizes` up to `self.max_decode_rows` (`[..., 48] -> [..., 48, 56, 59]` on TP3; `[..., 96] -> [..., 96, 104, 107]` on TP4).
