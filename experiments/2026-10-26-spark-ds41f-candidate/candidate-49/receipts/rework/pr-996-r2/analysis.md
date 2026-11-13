# PR #996 rework, round 2: views skip the cache

**Change.** On a rank without pad rows, `_local_mask` returns `mask[start:start + local_rows]` before any cache work. That is exactly the view `local_slice(mask[:num_tokens])` returns, made with one slice instead of two, and nothing is kept for it. The tail rank keeps round 1's per-forward cache of its padded tensor, and the `pad=True` fix is unchanged.

**No stale masks.**
- Ranks without pad rows keep nothing, and build their view from the current mask on every call, as 1107 does.
- The tail rank's guarantees are round 1's:
  - entries live on the forward's own context;
  - the entry holds the mask;
  - no version counter is read;
  - masks are not written during a forward;
  - the layout is part of the key.

**Result (sp8, two boots, against the recorded 1107 own-mark boots).**
- **Tail rank:** 5 of 5 cells faster beyond the interval on both timings (round 1: 4 of 5): 1.30x, 1.14x, 1.08x, 1.03x and 1.03x per call.
- **Non-tail, per call:** 6 faster, 15 within, 2 slower: TP3 T4096 r0 by 0.03 µs and TP4 T16384 r0 by 3.0 µs (0.9%).
- **Non-tail, pass:** 3 slower: TP3 T4096 r0 +0.20 µs, TP4 T6144 r0 +0.004 µs and TP4 T16384 r0 +2.5 µs.
- **Holds:** bit-exact (1680/1680 layers), inference mode, and 279 tests per boot. One round 1 assertion changed: only the tail rank now keeps masks.

**Round 1's attribution does not hold.**
- **Host cost on sp8:** one tensor slice costs about 0.7 µs, and the dict hit about 0.1 µs.
- **Per routing layer** on a rank without pad rows, under inference mode, the masks cost 5.2 µs in 1107, 2.0 µs in round 1 and 3.0 µs in round 2.
- **So:** round 1's lookup was already cheaper than 1107's view. The non-tail GPU work is identical in all three, and the host work overlaps the gate GEMM.

**The remaining slower cells are not the patch.**
- **In-process A/B:** round 2's `_local_mask` against 1107's body, interleaved per step, on two boots. All three cells are within on both timings. At TP4 T16384 r0, 1107's own mask logic measures 323.0 / 325.7 µs, a spread 16 times its recorded interval of 323.2 / 323.4.
- **Unchanged code, re-run:** round 1's tree, re-run in this session (ABBA with round 2), is faster than its own recorded boots in 4 non-tail cells and slower in 5. Against 1107 it is slower in 4 cells, not round 1's 6, with only 2 in common.
- **Round 2 against round 1, same session:** non-tail per call 3 faster, 19 within, 1 slower (TP4 T768 r0); pass 0 slower.

**Caveats.**
- The round 1 control's first cell (TP3 T768 r0) measured 133 µs on both control boots. It is unexplained, and it is not in the verdict.
- sp8 was shared, with every GPU command under the lock.
- The harness uses a stand-in ForwardContext and does not run the gather collective. A tp3-switched boot of this stack is still to be done.

## Coordinator's verdict: meets the mark on the controlled measurement

The sub-agent's literal verdict was `corrected-misses-mark`. It compared against 1107 boots recorded in another session, and the session-to-session spread of these GPU-bound cells is larger than the recorded interval. The same-session interleaved A/B shows no harm, and re-running unchanged code reproduces "slower" cells in other places.

Judged on the measurement that isolates the patch, #996 round 2 meets its mark:
- it fixes candidate-49's boot failure;
- the tail rank is faster in 5/5 cells;
- no cell is slower in the same-session A/B;
- it is bit-exact.

It joins the next candidate for fresh qualification, including a real TP3 boot.

The lesson for the workflow: the baseline of a microbenchmark A/B must run in the same session, interleaved (ABBA), never be taken from a recorded run of another session.

The analysis is the rework sub-agent's; the coordinator saved it and wrote the verdict.
