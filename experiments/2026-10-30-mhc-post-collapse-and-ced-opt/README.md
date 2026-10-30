# Fused `TileKernelsMHC.post_collapse` & CED Boundary Gather Deduplication

## Summary

1. **Fused `post_collapse_streams` (`TileKernelsMHC.post_collapse`):**
   - For non-final auxiliary hidden-state layers on the target model (`idx + 1 in self.aux_hidden_state_layers and idx + 1 != self.end_layer`, i.e., layers 2 and 30), the target model previously ran `aux_recon = layer.mhc.post(hidden_states, residual, post_mix, res_mix)` to materialize a transient `[T, 4, 5120]` BF16 tensor (40 KB/token), followed immediately by `layer.mhc.collapse(aux_recon, out=aux_out)` to collapse the 4 streams with the constant `0.25` mean mix into `[T, 5120]` BF16, and then discarded `aux_recon`.
   - Likewise, on every DSpark draft model forward (`DSparkDeepseekV4Model.forward`) and on the target model's final layer when `end_layer` is not in `aux_hidden_state_layers` and sink column extraction is disabled, `layer.mhc.post(...)` was followed immediately by `layer.mhc.collapse(..., pre_mix)`.
   - `post_collapse_streams` (`mhc_post_collapse`) fuses `post_streams` and `collapse_streams` into a single PDL-chained TileLang kernel that rounds each stream to BF16 in registers using `post_streams`'s exact FMA sequence and collapses the streams with `mix` in registers using `collapse_streams`'s exact FMA sequence, eliminating 80 KB/token of HBM write+read traffic and 1 kernel launch per call while remaining bit-for-bit identical (`torch.equal`) to `collapse(post(...))`.
2. **CED Boundary Gather Deduplication:**
   - At the CED encoder-to-decoder boundary (`idx == self.ced_decoder_start`), `DeepseekV4DecoderLayer.forward` gathered `positions` and `input_ids` after `prepare_global_kv`, and `DeepseekV4Model.forward` gathered `positions` and `input_ids` a second time right after `layer(...)` returned.
   - Pre-gathering `ced_positions` and `ced_input_ids` once in `DeepseekV4Model.forward` and passing them to `DeepseekV4DecoderLayer.forward` eliminates 2 redundant `gather_rows` launches per CED step.

## Verification (`dgx4`, `spark-ds41f:1030`)

- `pytest -p tests.kernels.deepseek_v41_pdl_sources tests/kernels/test_deepseek_v41_tilelang_mhc.py tests/models/test_deepseek_v4_1_ced_model.py`: **114 passed**, `PDL sources: 25 kernels launched, 18 with PDL, 0 violations`.
- Bit-for-bit exact (`torch.equal`) against `collapse(post(...))` and `mhc_pre_apply_mix_fwd(mhc_post_fwd(...))` across `tokens = 1..16384`.

## Benchmark Results (`dgx4` GB10)

| Operation | Rows | Before (`µs`) | After (`µs`) | Speedup | Saved |
|---|---:|---:|---:|---:|---:|
| `mhc.post` + `mhc.collapse` → `mhc.post_collapse` | 4 | 36.48 | 22.46 | **1.62×** | 14.02 µs/call (42.06 µs/step across 3 calls) |
| `mhc.post` + `mhc.collapse` → `mhc.post_collapse` | 6 | 36.48 | 22.46 | **1.62×** | 14.02 µs/call (42.06 µs/step across 3 calls) |
| `mhc.post` + `mhc.collapse` → `mhc.post_collapse` | 64 | 36.54 | 22.34 | **1.64×** | 14.20 µs/call (42.60 µs/step across 3 calls) |
| `mhc.post` + `mhc.collapse` → `mhc.post_collapse` | 96 | 36.45 | 22.30 | **1.63×** | 14.15 µs/call (42.45 µs/step across 3 calls) |
| `mhc.post` + `mhc.collapse` → `mhc.post_collapse` | 256 | 72.70 | 24.93 | **2.92×** | 47.77 µs/call (143.31 µs/step) |
| `mhc.post` + `mhc.collapse` → `mhc.post_collapse` | 2048 | 1231.74 | 535.87 | **2.30×** | 695.87 µs/call (2.09 ms/chunk across 3 calls) |
| `mhc.post` + `mhc.collapse` → `mhc.post_collapse` | 4096 | 2435.46 | 1055.65 | **2.31×** | 1379.81 µs/call (4.14 ms/chunk across 3 calls) |
| CED boundary `positions` + `input_ids` gathers | 2048 → 128 | 64.54 | 34.50 | **1.87×** | 30.04 µs/step (4 → 2 launches) |
| CED boundary `positions` + `input_ids` gathers | 4096 → 128 | 63.90 | 34.40 | **1.86×** | 29.50 µs/step (4 → 2 launches) |
