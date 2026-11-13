# PR #1010 rework, round 2: no defensible correction; dropped from this release

**Verdict:** `no-defensible-correction`. No design meets the whole mark: its two eviction clauses exclude each other on this probe. The inversion definition measures what the PR says it wants, so it was not changed. Round 2 delivers no patch. Under the owner's rule (a second miss drops the PR from the release), #1010 is dropped from this release. Its record is kept here and in `../pr-1010-r1`.

## Method
- sp8 only, CPU-only, under `flock`, with no hold file present. No build, no code change.
- A harness (`diag/oracle_bound.py`) replays the probe's LRU scenario and stress scenario, with round 1's core files mounted on candidate-48. At each eviction it chooses which cached blocks go by an ordering rule, and measures each hit with the real `get_computed_blocks`.
- Two container runs gave identical numbers.

## Numbers (both runs identical)

| Design | Inversions (0/512/8192/None) | Stress tp3 | Stress tp4 |
|---|---|---|---|
| 1107 (receipts) | 6/6/6/8 | 1,124,352 (99.6759%) | 1,122,816 (99.5397%) |
| Original 0763 (candidate-48) | 0/0/0/0 | 99.3809% | 98.9723% |
| Round 1 | 6/6/6/6 | 99.8574% (ceiling) | 99.8574% (ceiling) |
| floor1: aged checkpoints first, sorted by position | 1/1/1/1 | 99.8574% | 99.8574% |
| oracle0: best any 0-inversion order can do | 0 | 1,121,536 (99.4262%, −2,816 vs 1107) | 1,120,512 (99.3355%, −2,304) |
| oracle0 with dense SWA retention | 0 | 1,121,792 (99.4489%, exactly the bound below) | same |

- The harness's natural 0-inversion order reproduces candidate-48's two stress numbers exactly, which cross-checks the harness.
- floor1 and oracle0 are orderings the harness applies, not code.
- The boundary-matrix, spec-decode and test-list clauses pass for both existing corrections: 1404/1404/567, 60/60, 2679 passed ×2.

**At 36 blocks per eviction** (the count the PR's own test comment names; the code uses 18):
- Round 1 on tp4 drops to 93.1852%: agent 0's turn-4 hit falls to 0.
- The pre-0763 code, which reproduces 1107, also collapses, to 92.8901%. That was a single diagnostic run.
- The best 0-inversion order keeps 98.6546%.

## 1. The inversion definition measures the PR's stated purpose
- The PR's test docstring says partial eviction "always evicts the newest tail blocks first and preserves the root prefix blocks".
- Its README puts root checkpoints (512, 8,192) at the end of the queue that is evicted last.
- The LRU scenario targets turn 1's skipped SWA checkpoints, which are blocks outside the finished request's own run.

Narrowing the scope doesn't hold up. Within the request's own run, round 1 scores 0, but that drops the PR's stated target. Per KV group, round 1 still scores 6, all in the SWA group.

The check also guards a real property. In the LRU state at retention 512:
- With round 1's order, the extension hit stays at 25,088 while the 52 aged checkpoints are evicted. On the 53rd eviction it drops to 0, with every MLA block still cached.
- With the monotone order it falls in steps of 256 or 512 tokens, one step every 3–5 blocks. A branch hit at 8,256 keeps its 8,192 checkpoint until 170 evictions, against 17 under round 1's order.

## 2. Bound
- With 0 inversions, each conversation's cached blocks leave the queue in non-increasing position order, MLA before SWA at a tie.
- So the first cached block in the queue is some conversation's top block, which its next hit needs.
- Every block the stress eviction takes is therefore either one of at most 3 cached blocks at a lost 256-token position (measured maximum: 3 at every eviction), or a redundant copy, which exists only for a position already lost and re-prefilled (at most 3 per position).
- The three evictions take 54 cached blocks, so at least 18 positions (4,608 tokens) are lost, and the hit is at most 1,121,792 (99.4489%).
- 1107 sits only 8 positions below the ceiling on tp3 and 14 on tp4, so no 0-inversion order can reach it.
- The dense-retention oracle reaches the bound exactly, so the bound is tight.

## 3. The one assumption
The bound assumes the queue rule is the same in both scenarios. The LRU scenario runs retention 512 and 8192 through the same `cache_blocks`, `remove_skipped_blocks` and `free` paths as the stress profiles. The scenarios differ only in the number of conversations, chunked prefill and pool size. A rule keyed to any of those would game the probe, not fix the code.

## 4. The floor is 1 inversion, not 0
Meeting the stress clause requires aged checkpoints to come before some live top. floor1 reaches 1/1/1/1 at the ceiling; round 1 has 6. Getting from 6 to 1 in real code needs a segmented free queue in `kv_cache_utils.py`, a hot-path change that still misses the clause.

## 5. For a later release (the owner's decision)
The two existing corrections each miss exactly one clause:
- **Original 0763:** degrades gracefully and protects branch checkpoints; its stress share is 3,328 / 6,400 tokens below 1107.
- **Round 1:** stress share at the ceiling, but it inherits 1107's collapse under deeper pressure.

A later design could evict aged checkpoints first but keep a short ladder of fallback checkpoints per conversation. It would still count at least 1 inversion, so it would need a mark that names the property (for example "no hit collapses below the last retained checkpoint") instead of strict ordering.

## Recorded error
The first screening revision misclassified each turn's endpoint SWA checkpoint as "aged", because that block slides out of the window before free. floor1 then fell to 98.45% (tp3) and 91.01% (tp4). It was fixed by classifying blocks against the real lookup's hit path, and the error is kept in `commands.txt`.

(The analysis is the rework sub-agent's; the coordinator saved it.)
