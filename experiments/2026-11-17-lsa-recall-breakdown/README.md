# LSA recall breakdown

Date: 2026-11-17. Base: `cc20ed18b` (this repository). LSA sources: the
`2026-11-06-optional-lsa` overlay set at `a351a5ba0a3ac9b19e894cf6fd414883a4e5a28e`,
the commit its recorded benchmark boots ran from. Image: C36,
`sha256:3795255f485ea5c1a9db54c742921dd9b2917daad0799361e5c6f56a70119713`.

## Question

Opt-in LSA (`VLLM_DS41_LSA=1`, patch `0741`) keeps 78–98% of the native
indexer's top-512 keys on the captured operands
(`../2026-11-06-optional-lsa/results.md`). Which native keys does it drop, and
which of three changes would recover them: an always-scored recent window, a
larger shortlist budget, or a better block summary?

This is an untimed, out-of-band selection proxy. It changes no runtime file and
makes no speed or answer-quality claim. The one variable is the shortlist rule
applied to fixed operands.

## Method

Inputs are the 12 rank-0 operand captures of `2026-11-05-selector-fitting`
(three documents, each with dense layers 2, 8, 14 and source layer 20, 64 query
rows per capture), verified against the replay receipt's hashes. Dense rows see
33,624–33,656 keys and the source rows 122,726–131,180. Tensors stay on sp8.

`breakdown.py` runs the real kernels for both selections (`IndexerV41Plan` and
`QueryReusePlan` with dense budget 64 and source budget 1024), then uses Torch
for per-key native scores and for the counterfactual shortlists. Two checks
tie the Torch arithmetic to the kernels on every row:

- Torch's native top-512 equals the native kernel's selection: 100% on all 768 rows.
- Torch's emulated LSA selection equals the LSA kernel's: 100% on all 768 rows.

Kernel recall also equals the recorded `bench-boot1` values on all 12 captures.

`run_sp8.py` takes the sp8 lease (`~/spark-hold.json`, exclusive create,
30-second heartbeat), exports the LSA sources at the named commit from the sp8
checkout with `git archive`, and runs one container with the mounts and limits
of the recorded `sp8-bench-boot1` job, replacing only the timed script. Both
scripts were copied to `~/spark-lab/scratch/lsa-recall-breakdown/` on sp8; their
hashes are in each run's `command.json`.

```sh
python3 run_sp8.py --checkout ~/spark-lab/checkouts/lsa-development \
  --source-commit a351a5ba0a3ac9b19e894cf6fd414883a4e5a28e \
  --script breakdown.py --out ~/spark-lab/results/lsa-recall-breakdown/run-03
```

## Runs

| Run | Result | Note |
| --- | --- | --- |
| `runs/run-01` | failed, exit 1, 9.3 s | Mounted the sp8 checkout's current head (`13a9a0eb2`), whose overlay set no longer matches the recorded job: `ImportError: shortlist_prefix`. No kernel ran. The runner now exports the sources at a named commit. |
| `runs/run-02` | passed, 14.4 s | Included a score-mass metric that is invalid because indexer scores are signed; its values are not used. |
| `runs/run-03` | passed, 121.0 s | Fresh container, score-mass metric removed. Every count and recall equals `run-02`. |

`run-02` and `run-03` overlapped with two containers of other work that started
on sp8 at about 09:20 UTC without an sp8 hold file. Nothing here is timed and
the two runs agree exactly; `run-03`'s longer wall time is that shared GPU.
Torch peaks at 380,140,032 allocated and 744,488,960 reserved bytes
(standalone tensor allocator only). Each run's lease, command, output, verdict,
memory snapshots and release are retained; every lease was released.

## Results

Figures are over the three documents (192 rows and 98,304 native picks per
layer). Adjacent query rows do not add independent documents.

### What LSA keeps

| Layer | Fine keys scored | Top-512 recall, mean / min | Top-128, mean | Top-64, mean / min |
| --- | ---: | ---: | ---: | ---: |
| 2 | 4,096 | 80.94% / 65.43% | 90.72% | 93.43% / 75.00% |
| 8 | 4,096 | 88.98% / 68.55% | 96.62% | 98.11% / 89.06% |
| 14 | 4,096 | 91.82% / 76.76% | 97.54% | 98.84% / 90.62% |
| 20 (source) | 65,536 | 97.92% / 94.34% | 97.58% | 97.46% / 89.06% |

### Miss rate by native rank

| Layer | 1–32 | 33–64 | 65–128 | 129–256 | 257–384 | 385–512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 4.4% | 8.7% | 12.0% | 17.6% | 22.4% | 27.0% |
| 8 | 0.9% | 2.8% | 4.9% | 8.7% | 13.8% | 18.2% |
| 14 | 0.4% | 1.9% | 3.8% | 6.3% | 10.0% | 13.9% |
| 20 (source) | 1.3% | 3.8% | 2.3% | 1.9% | 1.9% | 2.1% |

The keys LSA selects in their place are the next-best native keys: 82.0%,
97.6%, 98.4% and 100% of replacements (layers 2, 8, 14, 20) have native rank
513–640, and the rest of the dense ones 641–768 apart from under 1%.

### Miss rate by how many native picks share the 8-key block

| Layer | Lone pick: share of picks | Lone pick: miss rate | Lone pick: share of all misses | 2 picks: miss rate | 3 or more: miss rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 43.8% | 39.6% | 90.9% | 8.1% | 0.3% or less |
| 8 | 34.4% | 29.1% | 91.0% | 4.6% | 0.3% or less |
| 14 | 27.3% | 27.9% | 93.1% | 3.3% | 0.2% or less |
| 20 (source) | 54.1% | 3.8% | 99.9% | 0.0% | 0.0% |

### Miss rate by distance from the row's end (keys)

| Layer | 0–511 | 512–2,047 | 2,048–8,191 | 8,192–32,767 | 32,768 and further |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 3.6% | 20.6% | 28.9% | 24.9% | 11.4% |
| 8 | 0.8% | 3.4% | 10.8% | 21.6% | 17.6% |
| 14 | 0.4% | 2.8% | 7.5% | 24.8% | 42.8% |
| 20 (source) | 0.0% | 0.4% | 0.8% | 1.6% | 4.0% |

The most recent 512 keys hold 18.8–27.7% of native picks and carry 0.0–4.6% of
the misses. In the source layer 78.2% of misses lie 32,768 keys or further back.
No miss is in the open page.

### Counterfactual shortlists (top-512 recall, mean / min)

| Shortlist | Layer 2 | Layer 8 | Layer 14 | Layer 20 |
| --- | ---: | ---: | ---: | ---: |
| Current (equals the kernel) | 80.94 / 65.43 | 88.98 / 68.55 | 91.82 / 76.76 | 97.92 / 94.34 |
| + recent window, 512 keys | 81.82 / 66.99 | 89.16 / 69.53 | 91.92 / 76.76 | 97.93 / 94.34 |
| + recent window, 2,048 keys | 84.49 / 71.09 | 89.78 / 72.27 | 92.60 / 78.12 | 97.95 / 94.53 |
| + recent window, 8,192 keys | 89.33 / 77.73 | 92.38 / 81.84 | 94.69 / 84.18 | 98.08 / 94.73 |
| Budget ×2 | 91.61 / 79.69 | 96.15 / 83.40 | 96.62 / 87.70 | 100 / 100 |
| Budget ×4 | 98.13 / 93.55 | 99.33 / 95.31 | 99.06 / 94.73 | 100 / 100 |
| Blocks ranked by the mean of their keys' native scores | 83.08 / 68.95 | 91.11 / 71.09 | 93.06 / 78.32 | 98.57 / 96.68 |

Source budget ×2 covers every visible key at these lengths, so it is the native
selection, not a shortlist. The last row is an oracle: it ranks blocks with the
exact per-key scores, which no summary written at cache time can do. Dense
budget ×4 scores 16,384 of about 33,600 keys; top-64 recall there is
99.15–99.91% mean and 89.06–96.88% minimum.

Of the missed dense keys, 55.9–64.9% sit in blocks ranked within twice the block
budget and the rest within eight times; every missed source key's block is
within twice the budget.

## Conclusion

- The dropped keys are lone picks: 91–100% of misses are the only native pick
  in their 8-key block, and blocks holding three or more picks are almost never
  dropped. An 8-key mean cannot represent one strong key among seven others.
- A better mean does not fix this. Ranking blocks by the exact mean of their
  keys' native scores recovers 1.2–2.1 points in the dense layers and 0.65 in
  the source layer.
- A recent window is not the fix either. Recent keys are already kept; 512 keys
  adds 0.0–0.9 points.
- In the dense layers most misses are near the selection threshold and are
  replaced by the next-ranked keys, but the top 32 are not safe: 0.4–4.4% are
  dropped. In the source layer the miss rate does not fall with rank (1.3% of
  the top 32, 3.8% of ranks 33–64).
- Budget is the only tested lever that moves recall. Dense ×4 reaches 98.1–99.3%
  mean with minimum rows at 93.6–95.3%, and no tested shortlist reaches the
  native selection in the dense layers. `2026-11-05-dense-retention` measured
  the speed cost of dense ×2: single-row selection slower than native.

No runtime configuration changes. This proxy covers three source-code documents
on rank 0, at about 33,600 visible keys in the dense layers and 123,000–131,000
in the source layer; it does not test longer contexts, other content, decode
rows, or final answers.
