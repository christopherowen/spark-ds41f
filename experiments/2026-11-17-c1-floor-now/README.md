# The single-stream step against its byte floor, today

Date: 2026-11-17. Base commit: `d4916ca97` (main). A recalculation: no boot, no kernel run.

## Question

How much of today's single-stream step (candidate-61 with the top-2 margin, tp4-ring) is weight
bytes at the attainable rate, and how much is left for step-time work?

## What was already measured

[../2026-10-26-c1-100tps](../2026-10-26-c1-100tps/README.md) measured the rate and built the
floor, for spark-ds41f-1022:

- GB10 streams expert weights at 266.6 GB/s at the margin (FC1's marginal rate at six rows on
  dgx4; FC2 265, a plain `torch.sum` 248 to 254, the packed LM head 240.7 in the traces; the
  specification is 273);
- a rank reads 5.71 GB a step, 21.4 ms at that rate, of a 30.55 ms step: 8.7 ms was not bytes.

Nothing here measures the rate again.

## One variable

The step the model is applied to: today's, in place of 1022's.

## Method

[floor_now.py](floor_now.py) imports that record's `model.py` and `floor.json` and takes the step,
its tokens and its verified drafts from the two boots of
[the margin's integrated screen](../2026-11-17-draft-calibration-sizing/README.md#integration)
(86.23 tok/s, 27.55 ms a step, 2.385 tokens a step, 2.54 drafts verified a step):

```
step = overhead + (dense + drafter_routed + 40 * N(live) * expert) / rate + dead * row
```

The live rows are the anchor and the verified drafts (the dead-row cut is on the device now);
the overhead is what is left. The block's other rows are counted as dead rows (0.37 ms each) or
as free, which brackets it.

```sh
python3 -I floor_now.py <c1.json> <c1.json> > receipts/floor-now.json
```

## Runs

[receipts/floor-now.json](receipts/floor-now.json).

| | spark-ds41f-1022 (the record) | Today |
| --- | ---: | ---: |
| tok/s | 74.3 | 86.23 |
| Step | 30.55 ms | 27.55 ms |
| Tokens a step | 2.271 | 2.385 |
| Live rows (anchor and verified drafts) | 2.89 | 3.54 |
| Distinct experts a layer | 13.8 | 16.1 |
| Bytes a rank reads | 5,708 MB | 6,128 MB |
| Byte floor at 266.6 GB/s | 21.41 ms (70% of the step) | 22.99 ms (83.5%) |
| Not bytes | 8.70 ms | 3.66 to 4.56 ms (13 to 17%) |
| A live row | 2.9 ms | 2.37 ms |

What the remaining overhead is worth, at today's tokens a step:

| Of the overhead removed | Step | tok/s | Tokens a step that 100 tok/s then needs |
| --- | ---: | ---: | ---: |
| none (today) | 27.55 ms | 86.2 | 2.77 (+16%) |
| a quarter | 26.4 to 26.6 ms | 89.2 to 89.9 | 2.65 to 2.67 |
| half | 25.3 to 25.7 ms | 92.4 to 94.0 | 2.54 to 2.58 (+6 to +8%) |
| two thirds | 24.5 to 25.1 ms | 94.6 to 96.9 | 2.46 to 2.52 |
| all of it | 23.0 to 23.9 ms | 99.4 to 103.3 | 2.31 to 2.40 |

## Conclusion

- **The step is 83% bytes.** More than half of 1022's 8.7 ms of overhead is gone; 3.7 to 4.6 ms
  is left.
- **Step-time work alone does not reach 100.** Every remaining millisecond that is not a byte
  would have to go for 99 to 103 tok/s. Half of it, the record's "credible" measure, is 92 to
  94.
- **The floor moved up, not down.** Today's step verifies more live rows (3.54 against 2.89), so
  it reads 0.42 GB more a rank. Tokens a step paid for that; the bytes a token cost barely moved
  (2,513 MB then, 2,569 MB now).
- **So 100 needs tokens a step or fewer bytes a token**: 2.54 to 2.58 tokens a step with half the
  overhead gone (+6 to +8%), 2.77 at today's step (+16%). The first is a more accurate drafter.
  The second is rows that reach fewer new experts, or fewer bytes read for the same rows.

## Limits

- 1022's byte inventory and routes: candidate-61 reads the drafter's LM head through an 8-bit
  screen for single-stream drafts, which the inventory still counts at 12 bits (up to 83 MB,
  0.3 ms, would move from the floor to the overhead).
- `N(r)` is from eight consecutive greedy tokens of the reference forward, interpolated at 3.54
  rows.
- The overhead is a remainder, not a trace: what it is made of today (collectives, launch gaps,
  streaming below the rate) needs a traced step of candidate-61, as `floor.py` read for 1022.
- Prose single stream on tp4-ring; agent traffic adds 0.3 to 0.5 ms of KV and index-key reads a
  step at 64-100K contexts (the record's estimate).
