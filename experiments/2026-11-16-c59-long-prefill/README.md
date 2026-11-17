# 2026-11-16-c59-long-prefill

Does candidate-59 keep its ~9.8K tok/s cold prefill at 524K and 1M?
Candidate-59's prefill benches stopped at the 262K size label (240,637 actual tokens), and no
candidate since spark-ds41f-1107 has measured above 262K.

- **Base commit**: `625450b87`.
- **Image**: `spark-ds41f:candidate-55`, with the candidate-59 overlay mounted.
- **Profile**: [tp4-switched-dev.json](tp4-switched-dev.json). It copies
  `experiments/2026-11-15-c59-prefill/tp4-switched-c59-dev.json` at `266bd1e19`, the profile
  candidate-59's prefill benches ran with. Only the notes differ.
  #1218 later added an `l2_prefetch.py` overlay mount to candidate-59's profiles. That overlay
  is not in this profile, so these runs extend the recorded curve.
- **Testbed**: `tp4-switched` (sp1..sp4, switched RoCE TP4, context 1,048,576).
- **One variable**: prompt length beyond 262K. The 262K label runs in the same boots as an
  anchor to candidate-59's recorded 262K.

## Workload

`bin/spark bench --compare none --suites prefill --prefill-text source --prefill-sizes 262144,524288,1000000 --prefill-repeats 2`.

The prompts are cold, seeded Python standard library source; the rate is actual prompt tokens
divided by client-side time to first token. There are 2 boots (`c59-long-prefill-b1`, `-b2`).

Spec: [bench-prefill-long-tp4-switched.json](bench-prefill-long-tp4-switched.json).

```sh
bin/spark candidate queue tp4-switched experiments/2026-11-16-c59-long-prefill/bench-prefill-long-tp4-switched.json --apply
```

## Prediction

In spark-ds41f-1107 on `tp4-ring` (default text), the time per prompt token rises with
position:

| Size label | tok/s | µs per token |
|---|---|---|
| 262K | 8,261 | 121.0 |
| 524K | 7,460 | 134.0 |
| 1M | 6,012 | 166.3 |

Candidate-59's prefill work (SP overlap, Engram prefetch, MoE SwiGLU fusion) is mostly
per-token cost. Its paired indexer reduction is the only part that scales with position
(1.05-1.06x on large rows). Adding 1107's position cost to candidate-59's 102.5 µs per token
at 240K predicts about 6,700-6,900 tok/s at 1M. That projection crosses topology and prompt
text, so it is a rough expectation, not a reference.

## Runs

Queue entry `20261116T182404-bench-prefill-long-tp4-switched` ran on sp1..sp4 from 18:24Z to
18:42Z on 2026-11-16, at deployment commit `ac768604d`, from the pinned profile above. Both
boots exited 0, and every sample completed.

The first entry, `20261116T164422`, was queued at 16:44Z behind candidate-60's screen. Its
pending file was removed from sp1's queue at 17:08:28Z by something other than the queue runner,
so it never ran. It was re-queued at 18:24Z.

| Size label | Actual tokens | b1 tok/s (TTFT) | b2 tok/s (TTFT) | Mean tok/s | µs per token |
|---|---|---|---|---|---|
| 262,144 | 237,746 | 9,708.3 (24.59 s) | 9,664.8 (24.70 s) | 9,686.6 | 103.2 |
| 524,288 | 471,725 | 9,402.2 (50.35 s) | 9,394.1 (50.40 s) | 9,398.2 | 106.4 |
| 1,000,000 | 916,234 | 8,754.9 (104.56 s) | 8,740.0 (104.74 s) | 8,747.5 | 114.3 |

The two boots differ by 0.45% at 262K, 0.09% at 524K and 0.17% at 1M.

- **Anchor:** candidate-59's recorded 262K was 9,757.3 tok/s at 240,637 tokens. Today's 262K is
  0.72% below it. The size list seeds the prompts in order, so today's 262K prompt is a
  different 237,746-token text.
- **Memory:** the lowest `MemAvailable` was 16.67 / 16.59 GiB on sp1 and 19.63-19.80 GiB on
  sp2..sp4, with no swap growth.
- **Thermals:** the GPU peaked at 78-82 °C, 8-13 °C below its limit, with 0 ms of thermal
  slowdown and power at most 75.2 W.
- **Conditions:** the bench reported the spaceport kernel warnings (4 KiB pages, no memory-saver
  module) that every switched run on these nodes reports.

Receipts: [receipts/tp4-switched/](receipts/tp4-switched/) has both `bench.json` files, their
telemetry and the window log.

## Conclusion

Candidate-59 does not hold 10K at 1M, but it comes much closer than predicted.

- **At 1M:** it reads 916,234 tokens at 8,747 tok/s (104.6 s), 90.3% of its 262K rate.
  Extending the 524K-1M slope to a full 1,000,000 tokens gives about 8,630 tok/s.
- **Against 1107:** spark-ds41f-1107 on `tp4-ring` kept 72.8% (6,012 against 8,261 tok/s).
  spark-ds41f-1030 on this testbed kept 78.1% (5,471.8 against 7,009.5 tok/s, in
  `experiments/2026-10-26-spark-ds41f-candidate/1107/receipts/switched-sp1/tp4/servebench/release/1030/`).

The prediction of 6,700-6,900 tok/s was wrong. It assumed 1107's position cost: about 45 µs per
token from 262K to 1M. Candidate-59 adds about 11 µs per token from 238K to 916K.

The cause is not attributed:
- no candidate between 1107 and candidate-59 was measured above 262K;
- the 1107 and 1030 records come from servebench's default text, not `bin/spark bench` source
  text.

So this is not a matched gain over a release. A matched figure needs a release reference
measured with this bench and text at 1M on this testbed.

## Short prompts

The second run completes the curve below 32K, with the same profile, bench, text and repeats.
The only change is the size list: 512, 1K, 2K, 4K, 8K and 16K, plus 32K as an anchor to
candidate-59's recorded 32K (9,548.5 tok/s at 28,907 tokens). At these sizes, time to first
token is mostly fixed per-request cost, so read the TTFT column first.

Spec: [bench-prefill-short-tp4-switched.json](bench-prefill-short-tp4-switched.json).

```sh
bin/spark candidate queue tp4-switched experiments/2026-11-16-c59-long-prefill/bench-prefill-short-tp4-switched.json --apply
```

### Short-prompt runs

Queue entry `20261116T190116-bench-prefill-short-tp4-switched` ran on sp1..sp4 from 19:01Z to
19:07Z on 2026-11-16, at deployment commit `7db478260`. Both boots exited 0.

| Size label | Actual tokens | b1 tok/s (TTFT) | b2 tok/s (TTFT) | Mean tok/s | Mean TTFT |
|---|---|---|---|---|---|
| 512 | 464 | 1,815.7 (0.247 s) | 1,806.1 (0.248 s) | 1,810.9 | 0.248 s |
| 1,024 | 1,041 | 3,155.7 (0.319 s) | 3,179.5 (0.316 s) | 3,167.6 | 0.318 s |
| 2,048 | 1,882 | 4,815.9 (0.415 s) | 4,791.9 (0.417 s) | 4,803.9 | 0.416 s |
| 4,096 | 4,572 | 6,958.4 (0.620 s) | 6,922.7 (0.623 s) | 6,940.6 | 0.622 s |
| 8,192 | 7,644 | 8,127.3 (0.956 s) | 8,198.6 (0.948 s) | 8,163.0 | 0.952 s |
| 16,384 | 14,900 | 9,279.1 (1.608 s) | 9,244.8 (1.614 s) | 9,262.0 | 1.611 s |
| 32,768 | 28,805 | 9,598.9 (3.011 s) | 9,617.0 (3.005 s) | 9,608.0 | 3.008 s |

- **Anchor:** the 32K anchor reads 0.6% above candidate-59's recorded 9,548.5 tok/s at 28,907
  tokens.
- **Memory:** the lowest `MemAvailable` was 17.89-17.9 GiB on sp1 and 20.35-20.47 GiB on
  sp2..sp4, with no swap growth.
- **Thermals:** the GPU peaked at 52 °C.

Receipts: [receipts/tp4-switched-short/](receipts/tp4-switched-short/).

Below about 4K tokens, time to first token sets the rate. A 464-token prompt answers in 0.25 s,
and up to 4K each extra 1K tokens adds only about 0.09 s. Assuming a straight line through
these points, the fixed per-request cost is roughly 0.2 s.
