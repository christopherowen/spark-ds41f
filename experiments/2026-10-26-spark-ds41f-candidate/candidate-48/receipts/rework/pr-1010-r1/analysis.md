# PR #1010 rework, round 1: why the stress probe fell below 1107

Base: candidate-48 image (sha256:d6ffd842…), with 0763 at PR head ad613458e559. Line numbers refer to the image's `/opt/spark-ds41f/candidate/vllm/vllm/v1/core/` files, with 0763 applied.

## The miss
The candidate gains 256 tokens on every non-evicted turn and loses 1,536 to 5,120 on turns after an eviction.
- tp3: 1107 = 1,124,352 (99.6759%), candidate = 1,121,024 (99.3809%).
- tp4: 1107 = 1,122,816 (99.5397%), candidate = 1,116,416 (98.9723%).
- The probe's ceiling, with every turn at its last 256-token boundary, is 1,126,400 (99.8574%).

## What the eviction takes
Before turns 4, 6 and 8 the probe takes free blocks from the LRU end of `free_block_queue` until it has 18 cached ones. `diag/diag_stress.py` logs which they are.

**Pre-0763 code on the same image** (diagnostic run; per-turn hits equal 1107's receipts):
- tp3, turn 4: eswa:a0@7680 … @512, @15872, @15360, @14848.
- tp4, turn 4: eswa@8192 … @57344 for both agents, then a0@69888 and a0@65536.

Every one is an SWA checkpoint that no hit has read since turn 1. A sliding-window hit reads only the window ending at the hit (`find_longest_cache_hit` scans right to left and stops at the first cached window, `single_type_kv_cache_manager.py:1303-1325`). These checkpoints therefore drift to the LRU end, and evicting them costs the next turn nothing.

**candidate-48:**
- tp3, turn 4: gmla, imla and eswa @75264, @75008, … down to gmla/imla @73728 are taken; agent 0 hits 73216 (−2048).
- tp4, turn 4: same tail; agent 0 hits 70400 (−4864). The next surviving SWA checkpoint is turn 2's endpoint.
- Totals: tp3 −2048, −1536, −1792; tp4 −4864, 0, −5120. Turn 6 on tp4 was absorbed by orphaned duplicate blocks left by the turn-4 re-prefill.

## Root cause
Three additions in 0763 move the old checkpoints behind the live tails:
1. `add_local_computed_blocks` (`:1478-1509`) follows `_block_prefix_chain` (built in `cache_blocks`, `:1654-1689`). `_refresh_free_retained_blocks_desc` (`:1452-1476`) then re-appends every retained checkpoint of the matched prefix to the most-recently-used end on each hit, though the hit did not read them.
2. The `remove_skipped_blocks` override (`:1869-1895`) re-appends all retained checkpoints whenever the window passes a cached block. The `free` override (`:1902-1906`) does it again.
3. `KVCacheCoordinator.free` (`kv_cache_coordinator.py:440-517`) moves the request's already-free skipped checkpoints, together with its live blocks, into one tail-to-root run at the recently-used end.

The decode-frontier recycler also un-hashes obsolete frontier blocks (`:1600-1635`); that part is correct and kept. With all four in place, no old cached block remains ahead of the conversations. Each conversation becomes one tail-to-root run, and the oldest-finished conversation's newest positions are evicted first. The PR's premise ("evicting an early SWA checkpoint collapsed find_longest_cache_hit … even when all MLA blocks remained cached") is false for extension hits: in the pre-0763 diagnostic, 18 root checkpoints are evicted and the turn-4 hit stays at 75264.

## Why the mark cannot be met as written
Zero inversions requires all of a conversation's cached blocks to form one run that never rises in token position. Then the 18-block eviction must take agent 0's newest positions. With at most 3 blocks per 256-token position, that is at least 6 positions, so turn 4 alone loses at least 1,536 tokens. The whole tp3 margin over 1107 is 2,048 tokens for three evictions. The inversion check measures the very policy that causes the loss.

## The correction
- **Kept:** decode-frontier retention and recycling; the stopping-step caching in `Scheduler` and `AsyncScheduler`; and `pop_blocks_for_free` sorted by token position across groups, with `free` releasing `reversed(pop_blocks_for_free(...))`. Without that sort, freeing is group-major: an eviction that reaches a live conversation takes one group's blocks position by position, and the other groups' blocks at those positions are wasted.
- **Removed:** the checkpoint refresh (`_req_retained_blocks`, `_block_prefix_chain`, `_refresh_free_retained_blocks_desc`, the three overrides, and the re-sort in `free`). This also drops a per-block chain list that grew with context length, and a sort on each cache step.
- **Tests:** the LRU-order test now asserts the corrected order, and the stress test asserts an exact boundary hit on every turn.

## Result
On both container runs:
- Boundary matrix 1404/1404/567; spec-decode 60/60.
- LRU inversions 6/6/6/6: five among the aged checkpoints, one where they meet the finished request's run, none inside that run.
- Stress 99.8574% on tp3 and on tp4 (1107: 99.6759% / 99.5397%).
- Test list: 2679 passed on each run.

## Trade-off left open
Under pool pressure, old root checkpoints are evicted first, as in 1107. A later request that shares only a root prefix and then branches falls back further. The probe doesn't measure this. Plain LRU limits it, because a hit that reads a checkpoint refreshes it.

(Written by the rework sub-agent; saved by the coordinator because the harness blocks sub-agents from writing `.md` files.)
