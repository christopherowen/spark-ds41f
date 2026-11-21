# TP3 L2 Prefetch Band (`WO_BANDS`) and `NEXT_MB=15` Verification

- **Date**: 2026-11-21
- **Base commit**: `7458d3248` (`spark-ds41f:candidate-63-r5`)
- **Reference**: `TODO.md` Next Release #6 (`experiments/2026-11-13-prefetch-window-sweep`).

## Two Single-Variable Arms (Tested Independently)

1. **`tp3-noband.json`**: Sets `VLLM_DS41_L2_PREFETCH_WO_BANDS=""` (removing the unconfirmed `17-32=12` band from `tp3-triangle` while keeping `WO_MB=15`, `FFN_MB=12`, `NEXT_MB=12`).
   - **Prediction**: Decode grid `M=17..32` (`4x24`, `8x24`) shifts by `±0.3%` (previous `tp3-switched` sweep measured `0.06%`, within noise); if neutral or faster without the band, remove `17-32=12` from TP1..TP3.
2. **`tp3-next15.json`**: Sets `VLLM_DS41_L2_PREFETCH_NEXT_MB="15"` (vs `12` default).
   - **Prediction**: Decode grid geometric mean improves `~0.2–0.5%` (following the `0.26%`/`0.49%` TP4 lead) with zero change to outputs (`128/128` greedy identical).

## Live Serving Runs (`tp3-triangle`, `dgx1..dgx3`)

All runs cooled `< 49 C` on `spark-ds41f:candidate-63-r5` inside lab window `dgx1-1795263633` (`results/private/lab/2026-11-21-campaign/`). Both arms passed `model` `128/128` greedy bit-exact (`outputs_sha256 = b795e6d4c813815a475ab1516faf277a90a86b73614c38f4700189484a8c532d`, `rank1_fraction = 1.0`, `perplexity = 1.3160824`, `failed = 0`).

| Arm | `grid` `mad5` GM vs `ctrl` | `grid` `fl25` GM vs `ctrl` | `1x3 @4K` (ms) | `8x24 @4K` (ms) | `8x48 @4K` (ms) | `c1` (tok/s) | `c8` (tok/s) | `262K` prefill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ctrl` (2-boot mean) | `0.000%` | `0.000%` | `16.907` | `22.622` | `27.567` | `69.69` | `169.53` | `7,437` |
| `noband-b1` (`WO_BANDS=""`) | `-0.045%` | `-0.017%` | `16.892` | `22.715` | `27.406` | `69.58 ±3.76%` | `169.39 ±1.14%` | `7,462` |
| `next15-b1` (`NEXT_MB=15`) | `-0.012%` | `-0.035%` | `16.845` | `22.686` | `27.705` | `69.66 ±3.81%` | `169.45 ±0.86%` | `7,427` |

### Conclusion

Both `WO_BANDS=""` (`-0.045%` `mad5` / `-0.017%` `fl25`) and `NEXT_MB=15` (`-0.012%` `mad5` / `-0.035%` `fl25`) are **statistically neutral within `±0.09%` boot noise** on `tp3-triangle`. Neither merits Boot 2 or a config change; keep existing TP3 prefetch defaults.
