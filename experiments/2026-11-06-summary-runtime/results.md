# Cooperative-summary integration runtime qualification

Source `e0577c15a344be49fc05601cda9981cd07bcafa7` (13-patch stack over C36 prepared upstream); one exact summary-construction variable relative to the raw-wide source1024/dense64/native-generation parent. Untimed standalone GPU correctness gate on `sp8` passes all six stages:

| Stage | Scope | Result | Seconds |
| --- | --- | --- | ---: |
| `summary-initcheck` | 4 rank-0 actual captures + synthetic pages 128/256 (`[-100, 100]` exponents, zero/tie/sign, skip/recycle) vs serial oracle under `initcheck` | Full-pool byte equality, 0 init errors | 9.08 |
| `summary-racecheck` | Same actual + synthetic summary oracle under `racecheck` | Full-pool byte equality, 0 race hazards | 21.01 |
| `registration` | Frozen kernel declaration & bind-after-freeze check (8 shapes, 74 declared calls) | Passed | 9.19 |
| `actual` | 12 captured runtime prefill/decode batches at `source_budget=1024` | Passed | 10.92 |
| `initcheck` | 4 strict uninitialized-suffix lifecycle cases under `initcheck` | Passed, 0 init errors | 12.24 |
| `graph` | 72 graph-capture dispatch routes (`approximate_source_prefill`, `source_budget=1024`) under `initcheck` | Passed, 0 init errors | 52.10 |

No component or request timing is claimed in this untimed gate. Own `sp8` reservation and container were removed immediately in `finally:` and independently verified absent (`runs/runtime-gates-01/independent-release.json`). Repeated model request and answer qualification under documented performance cooling remain required before promotion.
