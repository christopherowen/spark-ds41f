# 2026-11-16-c60-ring-prefill-sweep

What does candidate-60's cold prefill curve look like on the switchless TP4 ring, from 512 to
1M tokens? Candidate-60's prefill benches ran only on `tp4-switched`, at three sizes: 32K, 131K
and 262K. There it reached 10,000.1 tok/s at 131K in Step 1 and 9,924.4 in its final build. On
the ring, the last prefill sweep was candidate-58's, at the same three sizes: 9,408 / 9,755 /
9,620 tok/s.

- **Base commit**: `7db478260`.
- **Image**: `spark-ds41f:candidate-60`, built on dgx by the lab job `20261116T171622-build`.
- **Profile**: [tp4-dev.json](tp4-dev.json). It copies
  `experiments/2026-11-16-c60-prefill/tp4.json` (image, expected vLLM tree label, upstream lock)
  and adds development RPC, as `tp4-dev.json` there does. Because it requires the image's tree
  label, a rebuilt `candidate-60` tag refuses to boot instead of being measured.
- **Testbed**: `tp4-ring` (dgx1..dgx4, switchless cable ring, context 1,048,576).
- **One variable**: prompt length. Every size runs in the same boots.

## Workload

`bin/spark bench --compare none --suites prefill --prefill-text source --prefill-sizes 512,1024,2048,4096,8192,16384,32768,131072,262144,524288,1000000 --prefill-repeats 2`.

The prompts are cold, seeded Python standard library source; the rate is actual prompt tokens
divided by client-side time to first token. There are 2 boots (`c60-ring-sweep-b1`, `-b2`).

Spec: [bench-prefill-sweep-tp4.json](bench-prefill-sweep-tp4.json).

```sh
bin/spark candidate queue tp4-ring experiments/2026-11-16-c60-ring-prefill-sweep/bench-prefill-sweep-tp4.json --apply
```

## References

These are recorded runs, not measured again:

- **Candidate-58 on `tp4-ring`, source text:** 9,408 / 9,755 / 9,620 tok/s at 32K / 131K /
  262K (`experiments/2026-11-14-c58-prefill-10k`).
- **Candidate-59 on `tp4-switched`, source text, 512 to 1M:**
  `experiments/2026-11-16-c59-long-prefill`.

## Runs

Queue entry `20261116T191719-bench-prefill-sweep-tp4` ran on dgx1..dgx4 from 19:17Z to 19:35Z
on 2026-11-16, at deployment commit `6f1911271`, from the pinned profile on
`spark-ds41f:candidate-60`. Both boots exited 0, and every sample completed.

| Size label | Actual tokens | b1 tok/s (TTFT) | b2 tok/s (TTFT) | Mean tok/s | Mean TTFT | Boot spread | µs per token |
|---|---|---|---|---|---|---|---|
| 512 | 523 | 1,812.9 (0.273 s) | 1,872.7 (0.264 s) | 1,842.8 | 0.269 s | 3.24% | 542.7 |
| 1,024 | 926 | 2,735.3 (0.327 s) | 2,842.0 (0.314 s) | 2,788.6 | 0.321 s | 3.83% | 358.6 |
| 2,048 | 1,641 | 4,289.2 (0.420 s) | 4,294.8 (0.419 s) | 4,292.0 | 0.419 s | 0.13% | 233.0 |
| 4,096 | 4,204 | 6,495.8 (0.595 s) | 6,527.0 (0.592 s) | 6,511.4 | 0.593 s | 0.48% | 153.6 |
| 8,192 | 7,204 | 8,022.3 (0.901 s) | 8,005.9 (0.903 s) | 8,014.1 | 0.902 s | 0.20% | 124.8 |
| 16,384 | 14,493 | 9,335.9 (1.583 s) | 9,249.1 (1.599 s) | 9,292.5 | 1.591 s | 0.93% | 107.6 |
| 32,768 | 29,387 | 9,727.5 (3.168 s) | 9,453.2 (3.259 s) | 9,590.4 | 3.213 s | 2.86% | 104.3 |
| 131,072 | 121,796 | 9,901.5 (12.250 s) | 9,927.9 (12.217 s) | 9,914.7 | 12.233 s | 0.27% | 100.9 |
| 262,144 | 236,689 | 9,803.7 (24.086 s) | 9,813.5 (24.062 s) | 9,808.6 | 24.074 s | 0.10% | 102.0 |
| 524,288 | 469,970 | 9,553.1 (49.556 s) | 9,583.3 (49.401 s) | 9,568.2 | 49.478 s | 0.32% | 104.5 |
| 1,000,000 | 910,469 | 9,055.3 (101.042 s) | 9,071.9 (100.858 s) | 9,063.6 | 100.950 s | 0.18% | 110.3 |

- **Memory:** the lowest `MemAvailable` was 21.97-22.09 GiB on dgx1 and 23.95-25.06 GiB on
  dgx2..dgx4, with no swap growth.
- **Thermals:** the GPU peaked at 68-76 °C, at least 11 °C below its limit, with 0 ms of thermal
  slowdown and power at most 75.9 W.

Receipts: [receipts/tp4-ring/](receipts/tp4-ring/) has both `bench.json` files, their telemetry
and the window log.

## Conclusion

On the switchless ring, candidate-60 peaks at 9,915 tok/s at 121,796 tokens and keeps 91.4% of
that at 910,469 tokens: 9,064 tok/s, 101 s. Extending the 524K-1M slope to a full 1,000,000
tokens gives about 8,970 tok/s.

The sizes split into three regimes:

- **Up to 4K:** time to first token sets the rate. A 523-token prompt answers in 0.27 s.
- **16K to 262K:** the rate stays within 1.4% of the peak.
- **Past 262K:** each later position costs more, about 8 µs per token more at 910K than at 237K.

Against recorded runs:

- **Candidate-58 on this ring, source text:** 9,408 / 9,755 / 9,620 tok/s at 32K / 131K / 262K.
  Candidate-60 reads 1.9% / 1.6% / 2.0% higher. The 131K prompt differs: 121,796 tokens here,
  113,805 there.
- **Candidate-59 on `tp4-switched`** (`experiments/2026-11-16-c59-long-prefill`): candidate-60 on
  the ring is 1.3% / 1.8% / 3.6% higher at 262K / 524K / 1M. That crosses topology, prompts and
  build.
- **Candidate-60 on `tp4-switched`:** 9,924 tok/s at 131K. The ring matches it within 0.1%, so
  this build is as fast switchless as switched.
- **spark-ds41f-1107 on this ring:** 6,012 tok/s at 1M with servebench's default text. This is
  not a matched comparison.

10,000 tok/s is not reached at any size. The closest is 9,928 in one boot at 131K.
