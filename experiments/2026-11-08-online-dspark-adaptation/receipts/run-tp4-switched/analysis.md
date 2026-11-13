## Acceptance length per turn (coding, pooled over the four conversations)

| Turn | adapt-on | adapt-on-again | adapt-off | adapt-off-again | on mean | off mean | on vs off |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.403 | 4.403 | 4.540 | 4.540 | 4.403 | 4.540 | -3.01% |
| 2 | 4.651 | 4.651 | 4.742 | 4.742 | 4.651 | 4.742 | -1.92% |
| 3 | 5.786 | 5.786 | 5.815 | 5.815 | 5.786 | 5.815 | -0.51% |
| 4 | 5.746 | 5.746 | 5.704 | 5.704 | 5.746 | 5.704 | +0.75% |
| 5 | 5.023 | 5.023 | 5.141 | 5.141 | 5.023 | 5.141 | -2.29% |
| 6 | 6.287 | 6.287 | 6.448 | 6.448 | 6.287 | 6.448 | -2.49% |

## Totals

| | adapt-on | adapt-on-again | adapt-off | adapt-off-again | on mean | off mean | on vs off |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| coding acceptance length | 5.231 | 5.231 | 5.319 | 5.319 | 5.231 | 5.319 | -1.66% |
| held-out acceptance length | 2.219 | 2.219 | 2.311 | 2.311 | 2.219 | 2.311 | -4.01% |
| coding c1 decode tok/s | 152.13 | 152.01 | 157.98 | 158.31 | 152.07 | 158.14 | -3.84% |
| held-out c1 decode tok/s | 80.51 | 80.44 | 85.16 | 85.22 | 80.48 | 85.19 | -5.53% |

## Draft steps (pooled over requests)

| | adapt-on | adapt-on-again | adapt-off | adapt-off-again | on mean | off mean | on vs off |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| coding draft steps | 1769 | 1769 | 1741 | 1741 | 1769 | 1741 | +1.61% |
| coding verified drafts per step | 5.179 | 5.179 | 5.387 | 5.387 | 5.179 | 5.387 | -3.85% |
| coding accepted / verified | 0.817 | 0.817 | 0.802 | 0.802 | 0.817 | 0.802 | +1.88% |
| coding decode ms per step | 35.283 | 35.317 | 34.466 | 34.435 | 35.300 | 34.450 | +2.47% |
| held-out draft steps | 1819 | 1819 | 1766 | 1766 | 1819 | 1766 | +3.00% |
| held-out verified drafts per step | 2.224 | 2.224 | 2.672 | 2.672 | 2.224 | 2.672 | -16.76% |
| held-out accepted / verified | 0.548 | 0.548 | 0.491 | 0.491 | 0.548 | 0.491 | +11.65% |
| held-out decode ms per step | 28.292 | 28.317 | 27.489 | 27.448 | 28.305 | 27.468 | +3.04% |

## Text, gates and rank state

| Arm | workload text | gate (log-probs) ref / long | gate (text) ref / long | identical | ranks' digests |
| --- | --- | --- | --- | --- | --- |
| adapt-on | `9df935a06bc6` | `18d61961` / `d6568649` | `0a2dacdd` / `12fd8257` | True / True | equal at 46 update counts, last 11520 |
| adapt-on-again | `9df935a06bc6` | `18d61961` / `d6568649` | `0a2dacdd` / `12fd8257` | True / True | equal at 46 update counts, last 11520 |
| adapt-off | `9df935a06bc6` | `18d61961` / `d6568649` | `0a2dacdd` / `12fd8257` | True / True | none logged |
| adapt-off-again | `9df935a06bc6` | `18d61961` / `d6568649` | `0a2dacdd` / `12fd8257` | True / True | none logged |

## adapt-on against adapt-on-again (rank 0's state at each logged update count)

equal through update 4352; first differs at 4608; 28 of 46 logged counts differ
- update 0: conf=9657efa61acd0000 w1=95bad0b7fce0b532 (max row drift 0.0) / conf=9657efa61acd0000 w1=95bad0b7fce0b532 (max row drift 0.0)
- update 4608: conf=3b6671cdc4b2d349 w1=95baefddeb7139f6 (max row drift 0.0056786) / conf=3b6671d3d676ed38 w1=95baefddeb7139f6 (max row drift 0.0056786)
- update 11520: conf=ae7809f4c6600b89 w1=95bc36877c5398f0 (max row drift 0.0239736) / conf=ae6138c4a0101f5e w1=95bc4facc489b0b0 (max row drift 0.0240193)
