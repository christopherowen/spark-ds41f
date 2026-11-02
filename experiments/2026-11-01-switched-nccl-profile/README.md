# sparknet's switched NCCL profile on tp4-switched

## Summary

`tp4-switched` extends `tp4-ring`. The switched-shape rules drop the ring traffic controls
(`NCCL_ALGO`, `NCCL_MIN_NCHANNELS`, …), but four values still come from the ring. The shape's
notes say its NCCL settings are sparknet's `oneshot-switched` profile, yet sparknet's own switched
recipe (`recipes/switched.json`, sparknet 0.5.0) differs in all four:

| Variable | `tp4-switched` today (from `tp4-ring`) | sparknet switched | Arm |
|---|---:|---:|---:|
| `NCCL_MAX_NCHANNELS` | 4 | 8 | 8 |
| `NCCL_BUFFSIZE` | 4194304 | 1048576 | 1048576 |
| `NCCL_CUMEM_ENABLE` | 1 | 0 | 0 |
| `SPARKNET_ROCE_ALLGATHER_MAX_BYTES` | 2097152 | 4194304 | 4194304 |

`tp4-switched.json` is the recipe's `tp4-switched` with those four values set to sparknet's.
Nothing else changes: the arm resolves to the shape with exactly these four differences, and local
`bin/spark doctor` reports `configuration OK` with a switched node map. It sets no
`NCCL_MIN_NCHANNELS`, which switched transports refuse. sparknet's renderer replaces only routing
keys (HCAs, GID, traffic class, topology, socket interfaces), so the arm's values are the ones that
run. This was checked on one rank's container.

**Result:**
- **No gain at the recipe's 16,384-token budget on candidate-36.** Prefill -0.17% to +0.13% against
  the direct exchange alone, two boots each. ECN-marked packets roughly quadruple, with no RDMA
  errors.
- On spark-ds41f-1030 it added +1.0% prefill with 32,768-token prefill chunks, on a stack that also
  carried three other levers.
- At 16,384 on 1030 it read +0.6%, which two boots do not resolve.

Decode and outputs are unchanged everywhere. As a recipe change for `tp4-switched` at its own
budget, these boots do not support it.

## Measurement on candidate-36 at the recipe's budget

With `experiments/2026-11-01-direct-exchange-switched` (same window, image, suite and stock
reference; its README has the setup). Both arms run the direct exchange on candidate-36 at
16,384-token chunks. Only the four values differ. The `direct-nccl` arm is this directory's
`tp4-switched.json` values over the direct arm.

| Boot | Prefill 32K / 128K / 256K tok/s | Prefill vs stock | c16 paired | c1 paired | Grid | RDMA errors | ECN marks (4 ports) |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct | 7,505 / 7,553 / 7,197 | +3.67% | -0.11% | +0.09% | +0.08% | 0 | 28,479 |
| direct, repeat | 7,486 / 7,525 / 7,176 | +3.36% | -0.40% | -0.11% | -0.22% | 0 | 36,083 |
| direct + profile | 7,471 / 7,493 / 7,152 | +3.02% | -0.03% | +0.05% | +1.04% | 0 | 125,151 |
| direct + profile, repeat | 7,539 / 7,558 / 7,196 | +3.85% | -0.12% | +0.27% | +0.07% | 0 | 122,842 |

- **Prefill:** the profile changes it by +0.13% at 32K, -0.19% at 128K and -0.17% at 256K
  (two-boot means): nothing beyond boot noise.
- **Decode:** flat. The grid's +1.04% on one boot is not repeated (+0.62% ±5.13 over boots).
- **Outputs and logprobs:** identical to stock.
- **Congestion:** ECN-marked packets over the four switch ports rose about 3.8 times (about 124,000
  against 32,300 a boot), and about ten times on the head's port (59,266 against 5,762 in the first
  boots). No RDMA error counter grew. Each port appears as two PCIe functions with identical
  counters, counted once here; `metrics.json`'s `hca_ecn_marks` sums both functions.

Receipts: `candidate-36/runs/` (the two profile boots; the direct and stock boots are in the direct
exchange's `candidate-36/runs/`).

## Measurement on spark-ds41f-1030

Same site, session and suite as `experiments/2026-11-01-direct-exchange-switched` (its README has the
hardware, the suite and the stock reference). Every arm here runs the direct exchange
(`spark-ds41f:1030-direct`). Changes are against that experiment's first stock 1030 boot. Prefill is
the geometric mean over 32K, 128K and 256K; decode is paired per request.

**Ablation on a fuller stack.** These boots also carry three other config levers not in this PR: the
drafter's NVFP4 vocabulary head, adaptive verification row cost 0.01, and a 32,768-token prefill
budget. Only the NCCL profile differs between the two rows.

| Arm | Boots | Prefill 32K / 128K / 256K tok/s (mean) | Prefill | c16 | c1 | Grid |
|---|---|---:|---:|---:|---:|---:|
| with the profile | `112359Z`, `114513Z`, `145317Z` | 7,311 / 7,385 / 7,076 | +7.70 / +7.94 / +7.76% | +1.00 / +1.45 / +1.31% | +2.02 / +2.22 / +2.15% | -4.21 / -4.17 / -3.74% |
| without it | `120646Z`, `122809Z` | 7,249 / 7,307 / 7,002 | +6.65 / +6.81% | +1.14 / +1.25% | +1.97 / +2.10% | -4.60 / -3.89% |

The profile adds +0.9% at 32K, +1.1% at 128K and +1.1% at 256K, **+1.0%** overall. The two arms'
boots do not overlap: 7.70-7.94% against 6.65-6.81%. Decode and the grid are the same within boot
noise.

**At the recipe's 16,384 budget, direct exchange alone.** Boots `072604Z` and `074719Z` read prefill
+3.81% and +3.07% against the direct exchange's +3.10% and +2.55%: +0.6%, inside the 0.6-0.7%
spread between boots of one arm. Decode was unchanged. With 16,384-token chunks the gain is not
resolved on two boots. With 32,768-token chunks it is.

**Outputs:** every boot's greedy token ids are identical to stock 1030.

Not measured:
- channel counts between 8 and 16 (one boot at 16 read +0.4% over 8, within noise);
- the four values one at a time;
- `tp3-switched` and `tp5..tp8-switched`;
- any image after candidate-36, or a 32,768-token budget on candidate-36.

## Taking it into a release

Not proposed for release on these results. If a later budget or image shows a gain, the four values
would go into the recipe's `tp4-switched` `environment`, with a new benchmark reference for the
shape. `tp5..tp8-switched` extend `tp4-switched` and were qualified with `tp4-ring`'s values, so they
would need the four values pinned in their own `environment`, or a measurement at those sizes.
`tp3-switched` extends `tp3-triangle` and is unaffected.

## Receipts

`runs/<run>/` holds each boot's suite outputs: `suite.json`, `conditions.json`, `metrics.json`,
`grid.json`, `c16.json`, `c1.json` and `prefill.json`, plus the `arm.json` that ran, which matches the
bench's recorded `cluster_config_sha256`. The arms extend that site's configuration.
