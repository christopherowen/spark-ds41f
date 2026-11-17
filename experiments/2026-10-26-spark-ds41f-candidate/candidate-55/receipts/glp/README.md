# GLP on candidate-55's fourth build, 2026-11-12 and 2026-11-13

Image `31b056d3` (vLLM tree `8aaae3df`, #738's `0805`) on every testbed. GLP-39 (Spec v1) is every measured arm's
vector. GLP-27 (Spec v3, `hc_fold_pre_engram`, layers 11..37, alpha 3.5, revision `4d570926`, sha256 `24e97d41`)
runs on the qualification arms only, through [glp27-tp4.json](../../glp27-tp4.json),
[glp27-tp4-switched.json](../../glp27-tp4-switched.json) and [glp27-tp3-switched.json](../../glp27-tp3-switched.json).

| Testbed | Run | Queue entry | Window (UTC) | Lab checkout | Boots: time to ready |
| --- | --- | --- | --- | --- | --- |
| `tp4-ring` | `candidate-55-glp39-steered-tp4-b4` ([glp39-steered-tp4.json](../../glp39-steered-tp4.json)) | `20261112T135036` | 2026-11-12 15:13-15:20 | `998b21a0` | 110.2 s, 117.6 s |
| `tp4-ring` | `candidate-55-glp27-qualify-tp4-b4` ([qualify-glp27-tp4.json](../../qualify-glp27-tp4.json)) | `20261112T135039` | 2026-11-12 15:20-16:12 | `998b21a0` | 120.8 s, 115.5 s |
| `tp4-switched` | `candidate-55-glp27-qualify-tp4-switched-b4` ([qualify-glp27-tp4-switched.json](../../qualify-glp27-tp4-switched.json)) | `20261113T110833` | 2026-11-13 11:08-12:02 | `85ae0055` | 137.8 s, 133.2 s |
| `tp3-switched` | `candidate-55-glp27-qualify-tp3-switched-b4` ([qualify-glp27-tp3-switched.json](../../qualify-glp27-tp3-switched.json)) | `20261113T110017` | 2026-11-13 11:00-12:23 | `85ae0055` | 131.9 s, 130.5 s |

Each GLP-27 run boots its profile twice and runs, on each boot, the temperature-0 gate on the stock name
(`token_determinism_record.py`, with and without log probabilities),
[dual_probe.py](../../../../2026-10-28-glp-dual-serving/dual_probe.py) `--speed`, 16 distinct streams
(`distinct_streams.py`) and [advbench.py](../../../../2026-10-28-glp-dual-serving/advbench.py). The reply digests
are [dual_replies.py](../../dual_replies.py)'s, over each name's replies to the probe's 8 prompts (token IDs and log
probabilities).

## Results

| | `tp4-ring` | `tp4-switched` | `tp3-switched` |
| --- | --- | --- | --- |
| Gate on the stock name, GLP-27 configured | 2 of 2 boots, 8 of 8 runs identical; candidate-54's TP4 hashes `18d61961`/`d6568649`, text `0a2dacdd`/`12fd8257` | the same | 2 of 2 boots, 8 of 8 runs identical; candidate-55's TP3 hashes `5c22ce41`/`bce6092e`, text `26a739a9`/`12cfee9b` ([third-build](../third-build/README.md)) |
| `dual_probe.py` | every check passes on both boots; steering moves all 8 prompts | the same | the same |
| Replies: stock / steered | `9872053e` / `4cc693cd` on both boots | `9872053e` / `4cc693cd` on both boots | `6cf416a8` / `f2245c80` on both boots |
| c1 tok/s: stock / steered | 75.82, 76.31 / 76.80, 76.38 | 75.06, 76.54 / 76.36, 77.80 | 56.29, 56.39 / 57.45, 57.36 |
| c16 tok/s (dual probe): stock / steered | 303.01, 303.06 / 298.24, 299.04 | 299.16, 298.93 / 295.65, 294.62 | 164.65, 164.37 / 163.90, 163.52 |
| 16 distinct streams, tok/s | 273.93, 273.99 | 270.10, 269.86 | 155.99, 156.92 |
| AdvBench, 520 prompts, stock: no refusal phrase / judge yes / both | 1.35% / 0.96% / 0.77% | the same | 1.73% / 0.96% / 0.77% |
| AdvBench, steered: no refusal phrase / judge yes / both | 63.65% / 66.54% / 41.73% | the same | 68.85% / 62.69% / 43.08% |

- **The stock name is unchanged with GLP-27 configured** on every testbed: each gate gives the hashes its shape's
  measured arms give with GLP-39.
- **Steering is deterministic and topology-independent at TP4:** both boots of each run give the same steered
  replies, and `tp4-ring` and `tp4-switched` give the same ones (`4cc693cd`) and the same AdvBench rates. TP3's replies differ from TP4's, as its stock replies and gate hashes do.
- **Steering costs nothing measurable at c1** (steered 0.1% to 2.1% faster than stock on every boot, within the
  probe's spread) and 0.5% to 1.6% slower at c16. These rates are recorded, not gated.
- **GLP-39 (Spec v1) steers as before:** `candidate-55-glp39-steered-tp4-b4`'s replies are stock `9872053e`,
  steered `739b8abc` on both boots, the third build's (`-b3`, [third-build](../third-build/README.md)).

## Files

- `<testbed>/lab/`: each run's table, window log and overlay list, and `<run>-records.sha256`, the SHA-256 of the
  gate's `record-*.json` files, which stay on the testbed's head under
  `results/private/determinism/<run>/` (checked against the head's copies when copied).
- `<testbed>/determinism/<run>/`: each extra's output and telemetry. `advbench.py` prints every reply; its output
  stays on the testbed's head (dgx1, sp1, sp5) under `results/private/determinism/<run>/`, and only each boot's
  summary line is kept, in the table.
