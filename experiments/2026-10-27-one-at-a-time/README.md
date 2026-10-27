# Candidate 9, one fix at a time (2026-10-27)

Candidate 9 against 1025 measured about 46 vLLM changes as one bundle (c16 +0.8%, c8 +1.5%,
c4 +1.8%, temperature-0 text identical). A sum hides a change that costs. Each fix here is
measured alone: candidate 9's image with only that fix taken out, beside candidate 9 itself,
in one session.

- **Base:** candidate 9, `spark-ds41f:19f2c20ed4d6-next-9` (vLLM e2b2b0b59, tree 2b0931a0),
  arm `experiments/2026-10-26-spark-ds41f-candidate/rc9/rc-tp4.json`.
- **One variable:** the files a single fix changes, mounted as they are with that fix reverted
  at e2b2b0b59 ([make_arms.py](make_arms.py), `overlay/<label>/`). Nothing else differs.
- **PDL** (0520-0525) already has its own A/B: `rc9/pdl-ab-tp4.json`.

## Fixes

| Arm | Fix taken out | Claim to check | Metric |
| --- | --- | --- | --- |
| no-0504 | routed experts at 64-row tiles for prefill (`prefill_block_M=32` instead: 0504 cannot be reverted alone, PDL's late kernels build on it) | faster prefill, same bits | cold 32K / 131K prefill |
| no-0511 | SP prefill MoE: combine writes the exchange slots | faster prefill | cold 32K / 131K prefill |
| no-0505 | two unread budget-sized buffers dropped | more free memory, same speed | startup memory, c8 / c1 |
| no-0506 | attention scratch in its own workspace lane | same bits, same speed | determinism, c8 / c1 |
| no-0526 | CED replays the prompt's last 128 rows | correct CED at any chunking; cost | c8 / c1, prefill |
| no-0528 | block verification ends at the dead-row cut | correct sampled verification; cost | sampled digest, c8 / c1 |

Not isolated yet: 0502 (KV zeroing), 0508 (softmax fix), 0510 (mHC cast) and the live
page tables (0513-0519). Each conflicts with later commits when reverted and needs a hand
revert or a switch.

## Each run (`<label>.json`)

1. Candidate 9, then the arm: the determinism gate with and without logprobs, the seeded
   sampled digest.
2. The lean suite ([lean-suite.json](lean-suite.json), a copy of the cluster program's) on
   candidate 9, the arm, candidate 9 again: model behaviour at 16 streams, the engine grid,
   c8 and c1 closed loops (240 s), cold prefill at 32K and 131K, three repeats.

## Acceptance

- A fix that claims the same bits: the arm's greedy and logprob digests equal candidate 9's.
- A fix that claims speed: candidate 9 beats the arm on its metric by more than the spread
  between candidate 9 and candidate 9 again. Otherwise it is fixed to deliver or dropped.
- A correctness fix: kept; its cost (arm faster than candidate 9) is reported, and a cost
  beyond the bracket spread is a reason to make it cheaper.

## Results

Not run.
