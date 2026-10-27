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
| no-0502 | no per-allocation KV zeroing (`uses_only_written_slots=False` on the three cache specs: the manager zeroes each allocated block again) | same bits, cheaper allocation | determinism, c8 / c1, TTFT |
| nopdl-no-0510 | mHC finalize writes the MXFP8 cast, the projections stop casting | same bits, shorter layers | determinism, c1, prefill |
| nopdl-no-pagetables | sparse-MLA page tables over live pages (0513-0519) | same bits, cheaper metadata at long context | determinism, c1, prefill |

The `nopdl-` arms are measured on candidate 9's no-PDL arm (`rc9/rc-tp4-nopdl.json`) and
compared with it: PDL was written over these fixes (it chains the kernels they changed), so
neither reverts alone with PDL in place. 0510's revert there keeps 0504's tile heights and
0511's deferred combine and takes the MoE input unquantized again; the determinism gate
against the no-PDL arm checks that hand resolution, since 0510 claims the same bits.

Not isolated: 0508 (the softmax fix in the logprob and verification kernels). 0528 was
written over its arithmetic, so a revert means merging two kernel rewrites by hand; its
cost is the kernel's own (`benchmarks/kernels/benchmark_tilelang_rejection.py`), and it is
a correctness fix.

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

## Additions on candidate 9 (`make_bundles.py`)

Two byte levers, each alone on candidate 9's image, built from their uncommitted vLLM
worktrees at e2b2b0b59 by `make_bundles.py BLOCK_SCALES_WORKTREE EXPERT_CODES_WORKTREE`.
Each spec runs kernel-lab bundles before any boot, then the same gates and lean suite as
the removals, with the arm in the middle.

| arm | change | kernel check | claim |
|---|---|---|---|
| block-scales | MXFP8 GEMMs load the 32x32 block-scale grid at `SFB[row // 32]` | `bundles/block32-base` and `bundles/block32-scales` print block-32 GEMM output digests (seeded TP4 shapes, 1-16 decode rows and prefill rows); they must match line for line. `block32-scales-tests`: the branch's GEMM tests | same bits, decode faster |
| expert-codes | 2-bit scale codes for experts whose block scales span at most 4 values | `bundles/expert-codes`: the branch's MoE tests, the mixed 2-bit/4-bit launch against the uncoded GEMM bit for bit | same bits, decode faster |

Kernel bundle outputs stay on the nodes under `~/spark-lab/results/<run>/<bundle>-<node>/`.

## Results

PDL (rc9/pdl-ab-tp4.json, 2026-10-27, no-PDL then PDL then PDL again): same bits (128/128
greedy, teacher-forced KL 0, equal determinism, sampled and control digests). Decode grid
1.3% faster (±0.17%, repeat drift 0.7%); c1 engine steps +1.9%; distinct streams +3.4% at
c4, +1.4% at c8, +0.4% at c16; 262K prefill 0.4-0.7% slower. Kept.

Leave-one-out arms: queued 2026-10-27, not finished.
