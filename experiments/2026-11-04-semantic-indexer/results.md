# Exact semantic search screen

Two fresh C36 processes agree on every diagnostic observation. All48 native
top512 checks and12 source block/count/reindex checks pass. These are twelve
snapshots from one prompt; source queries repeat across ranks. No new full
model serving or request-speed qualification is claimed.

| Snapshot | Visible keys | Semantic ideal retained | Practical scored once | Practical pruning |
| --- | ---: | ---: | ---: | ---: |
| rank-0-dense-0.pt | 33656 | 15402 | 19608 | 41.74% |
| rank-0-dense-1.pt | 33656 | 11385 | 13586 | 59.63% |
| rank-0-dense-2.pt | 33656 | 8836 | 9967 | 70.39% |
| rank-0-source-0.pt | 36576 | 33571 | 36521 | 0.15% |
| rank-1-dense-0.pt | 34672 | 13274 | 17539 | 49.41% |
| rank-1-dense-1.pt | 34672 | 14455 | 17770 | 48.75% |
| rank-1-dense-2.pt | 34672 | 10013 | 11490 | 66.86% |
| rank-1-source-0.pt | 36576 | 33571 | 36521 | 0.15% |
| rank-2-dense-0.pt | 32896 | 10718 | 14330 | 56.44% |
| rank-2-dense-1.pt | 32896 | 8648 | 9643 | 70.69% |
| rank-2-dense-2.pt | 32896 | 4809 | 5064 | 84.61% |
| rank-2-source-0.pt | 36576 | 33571 | 36521 | 0.15% |

The practical column includes every exact seed score and reuses it. It is work
accounting from an untimed oracle, not a kernel timing. Native eight-key
expansion loses additional sparsity; the executable native candidate API
has not been changed to gather arbitrary individual keys.

Cheap key-PCA maps through32 dimensions retain every key. Rank64 saves at
most0.42% of keys; rank96 saves62.48% on one dense sample but its optimistic
BF16 coefficients/norm already cost196 bytes/key versus68 native packed
bytes. Other rank96 samples retain96.59–100% of keys. Query-fitted maps
through64 dimensions also retain every key on distinct dense queries.
The initial positive source result was training/evaluation overlap and is
excluded: its query is identical across ranks. Both the flawed initial
screen and the corrected exclusion remain retained.

Small shape ranks lose pruning. Full seven-vector shapes keep useful dense
pruning but require a128-dimensional center, seven128-dimensional factors,
residual/rounding metadata and query-dependent slope optimization. Even an
optimistic FP32 center/factor representation is4096 bytes per eight-key
group before other metadata, versus544 native packed-key bytes. This is
a cost obstacle, not a measured impossibility result for every exact method.

Keep the native runtime. A new kernel/metadata design must demonstrate lower
total cost before another request comparison. No latency confidence interval
is fabricated from deterministic retention counts. C37 was restored on all
three nodes, with independent live/image/source/lease verification.
