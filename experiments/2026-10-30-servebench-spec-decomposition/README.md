# Speculative Log-Throughput Decomposition and Per-Position Conditional Acceptance in Servebench

## Motivation

When comparing two speculative decoding runs across concurrency levels (`c1`, `c8`, `c16`), a net change in output throughput (`output tok/s`) or `output tokens/step` conflates three distinct effects:
1. **Acceptance length per draft** ($1 + \tau = 1 + \text{accepted\_per\_draft}$),
2. **Engine step rate** ($\text{steps/s} = 1 / T_{\text{step}}$), and
3. **Decode batch occupancy** ($\text{requests/step} = \frac{\text{output\_tokens/step}}{1 + \tau}$),

as well as whether a change in `output tokens/step` came from verifying more rows ($\text{verified\_rows/step}$) or from higher verification efficiency ($\text{row\_yield} = \frac{\text{tokens/step}}{\text{verified\_rows/step}}$).

Similarly, at the per-position level ($k = 0 \dots K-1$), unconditional survival $S_k$ and verification coverage $C_k$ are already recorded, but separating **conditional acceptance given survival through $k-1$** ($S_k / S_{k-1}$, with $S_{-1} = 1$) from **verified-row yield at position $k$** ($S_k / C_k$) directly reveals whether tail positions are limited by drafter accuracy or by verification cutoff coverage.

## Design

- **`servebench/report.py`:**
  - Extended `rows_a_step(summary)` to include `acceptance_length` ($1 + \tau$) and, for each draft position $k$, `conditional_acceptance` ($S_k / S_{k-1}$, with $S_{-1} = 1$) and `verified_yield` ($S_k / C_k$).
  - Added `spec_decomposition(base_summary, candidate_summary)` computing the exact additive log-throughput decomposition:
    $$\Delta \log(\text{server tok/s}) = \Delta \log(1 + \tau) + \Delta \log(\text{steps/s}) + \Delta \log(\text{requests/step})$$
    and the exact log-tokens-per-step decomposition:
    $$\Delta \log(\text{tokens/step}) = \Delta \log(\text{verified\_rows/step}) + \Delta \log(\text{row\_yield})$$
    along with per-position deltas (`survival_delta`, `coverage_delta`, `conditional_acceptance_delta`, `verified_yield_delta`).
  - Attached `spec_decomposition` to `report.compare(base, candidate)` when speculative counters are present on both runs and rendered a `speculative log-tps:` summary line in `report.format_comparison`.
- **`servebench/suite.py`:**
  - Extended `format_accounting(by_label)` to display `[cond/yield k:cond/yield ...]` alongside `survival/coverage` per draft position while preserving existing output substrings.

## Verification

- `python3 -m unittest -v tests/test_servebench.py`: all tests pass, including `test_rows_a_step` and `test_spec_decomposition_identities_and_formatting`.
- `python3 bin/spark doctor`: `configuration OK`.
