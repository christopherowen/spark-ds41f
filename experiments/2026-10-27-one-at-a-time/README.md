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

`x7-gemm.json` (kernel only): `bundles/x7-gemm`, copied from the decode-bytes experiment
with candidate 9's image. The X7 tile load against the production decode GEMM on real
DS4.1-1017 projections: bit-exact outputs at 1-16 rows and per-call times, cold and
L2-warm. A serving arm follows only if it matches and is faster.

Kernel bundle outputs stay on the nodes under `~/spark-lab/results/<run>/<bundle>-<node>/`.

## Results

PDL (rc9/pdl-ab-tp4.json, 2026-10-27, no-PDL then PDL then PDL again): same bits (128/128
greedy, teacher-forced KL 0, equal determinism, sampled and control digests). Decode grid
1.3% faster (±0.17%, repeat drift 0.7%); c1 engine steps +1.9%; distinct streams +3.4% at
c4, +1.4% at c8, +0.4% at c16; 262K prefill 0.4-0.7% slower. Kept.

Leave-one-out arms (queued 2026-10-27; change is candidate 9 against the arm, repeat is
candidate 9 against candidate 9 again):

| fix | bits without it | verdict on its metric | decision |
|---|---|---|---|
| 0504 (MoE prefill block_M 64) | equal (determinism, greedy, sampled) | prefill +4.2% at 32K (±1.3%), +4.4% at 131K (±0.4%), repeat under 0.5%; decode, c8, c1 same | keep |
| 0511 (SP prefill MoE combine writes the exchange slots) | equal | prefill +2.8% at 32K (±1.0%), +2.7% at 131K (±0.3%), repeat under 0.2%; decode grid 1.2% faster (±0.15%, repeat 0.2%); c8, c1 same | keep |
| 0510 (mHC finalize writes the MXFP8 cast; against no-PDL) | equal | decode grid 1.4% faster (±0.16%, repeat 0.16%); prefill 131K +0.85% (±0.4%); 32K, c8, c1 same | keep |
| 0513-0519 (sparse-MLA page tables over live pages; against no-PDL) | equal | nothing beyond the repeat: decode grid 0.4% faster against 0.6% repeat drift, every 64K/256K/1016K row same; prefill, c8, c1 same | unproven: drop, or show the workload it pays on |
| 0502 (no per-allocation KV zeroing) | equal | nothing beyond the repeat: decode grid +0.7% against -0.8% repeat drift; c8, c1, prefill same; c8 TTFT p99 4.3% lower with it (no interval) | unproven: measure allocation-heavy TTFT, else drop |
| 0505 (two unread budget-sized buffers dropped) | equal | speed same as claimed (decode, c8, c1, prefill all within the repeat). Memory: the startup KV pool is pinned (--kv-cache-memory-bytes), so no-0505-memory.json found it equal (11,554,122 tokens on all three boots); the suite's lowest MemAvailable (conditions.json) was lower without 0505 on every node against both candidate 9 boots: 0.4 GiB (dgx1), 0.6 (dgx2), 0.85 (dgx3), 0.4-0.6 (dgx4) | keep: more memory headroom, same bits and speed |
| 0506 (attention scratch in its own workspace lane) | equal | same speed as claimed: decode, c8, c1, 32K same; 131K prefill 0.4% slower with it (±0.1%, repeat 0.1%) | keep; memory: lowest MemAvailable 0.1-0.4 GiB lower without it on dgx1-dgx3 against both candidate 9 boots, dgx4 mixed (weaker than 0505's) |
| 0526 (CED replays the prompt's last 128 rows) | equal | correctness fix; cost 0.5% at 131K prefill (±0.1%, repeat 0.1%); decode, c8, c1, 32K same | keep, cost reported |
| 0528 (block verification ends at the dead-row cut) | greedy and logprob digests equal; the seeded sampled digest differs (the fix: sampled verification) | correctness fix; cost: decode grid 1.2% slower with it (±0.16%, repeat 0.4%); c8, c1, prefill same | keep; the cost is beyond the repeat, so make it cheaper |

Additions:

| change | bits | result | decision |
|---|---|---|---|
| block-scales (SFB[row // 32]) | kernel: block32-base and block32-scales digests equal over 182 cases (7 TP4 shapes x 26 row counts), branch GEMM tests pass; serving: determinism, greedy, sampled digests equal | decode grid 1.3% faster (±0.16%, repeat 0.3%); 131K prefill 1.1% faster (±0.5%, repeat 0.3%); c8, c1, 32K same | promote candidate |
| expert-codes (2-bit scale codes for narrow experts) | kernel: 114 MoE tests pass, the mixed 2-bit/4-bit launch equals the uncoded GEMM bit for bit; serving: determinism, greedy, sampled digests equal | slower: prefill 2.7% at 32K (±1.4%), 2.8% at 131K (±0.1%), c8 1.6% (±1.2%), repeats under 0.3%; decode grid within its 1.8% repeat drift | fix or drop: the expand costs more than the bytes it saves |
| x7-gemm (kernel only) | rerun (2026-10-28, signed test): X7 pack round trip exact on real Q-B weights (8.7% fewer bytes than production), every X7 and X8 GEMM case bit-exact | X8 control 0.97-1.07x production (the harness is fair); X7 1.9-2.0x production's time on Q-B, 3.3x on indexer Q-B, 6.2x on fused Q-A/KV, 6.7x on Engram wkv, 10x on shared gate/up, 1-16 rows | hold back: the in-kernel decode costs far more than 8.7% of the bytes saves; a rework needs another unpack design |

## Are the correctness fixes' costs real? (`make_cost_check.py`, `cost-check.json`)

The costs charged to 0506, 0526 and 0528 above come from one boot per arm, and two things
say they may be boot noise, not cost:

- At 131K prefill every arm on candidate 9 beat its run's candidate 9 boots: 0502 +0.24%,
  0505 +0.25%, 0528 +0.35%, 0506 +0.44%, 0526 +0.47%. Five unrelated fixes cannot all cost
  the same. Candidate 9's 18 boots in these runs spread 0.22% (sd), 0.89% (range).
- 0528's 1.2% was on the engine grid, whose dummy steps (`ModelRunner._dummy_run`) run the
  target and the drafter but never the rejection sampler 0528 changes; and every lean
  workload is temperature 0, where 0528 commits what it committed before.

`cost-check.json` boots candidate 9 three times and null-overlay, no-0526, no-0506 and
no-0528 twice each, interleaved (11 boots), on [cost-suite.json](cost-suite.json) (the lean
suite's heat-soak, grid and prefill). null-overlay mounts candidate 9's own copies of the
11 files those arms replace: it differs from candidate 9 only in being an overlay arm.
`cost_check.py` compares each arm's boots with candidate 9's (Welch over boots). A cost is
real only if the arm beats candidate 9 and null-overlay does not; only then is it reworked.
