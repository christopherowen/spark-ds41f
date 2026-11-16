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

Pending.

## Conclusion

Pending.
