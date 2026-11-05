# Alternatives and the remaining bound

These CPU experiments use the same native cache manager, source base and test
image recorded in `base.json`. The source PR's implementation remains unchanged
after this audit. Reproduction scripts and exact commands, generated-source hashes,
all output and failures are under `runs/commands` and `runs/*cost*`/`alternative-runs`.
They are supplemental tests, not GPU timings. Every run stayed within two CPUs and
4 GiB without CUDA initialization. No serving cluster was touched.

| Alternative | Native result | Decision |
| --- | --- | --- |
| Original resend anchor | 28 of the expanded 48 endpoint cases fail | Loses a complete encoder tail under sparse retention |
| Keep only the complete endpoint | 22 failed, 28 passed | Cannot preserve both identical resend and extension/cancellation behavior |
| Keep resend and endpoint minus one token | 13 failed, 37 passed | Discards valid complete pages at aligned prompt lengths |
| Apply two endpoints to ordinary replay too | 3 unrelated retention regressions, alongside 4 unchanged HiSparse failures | Rejected; restrict to multi-token replay |
| Dense retention at each hit-page boundary, original coordinator | 26 passed | Correct for the tested geometry, but retains many more pages |
| Selected two endpoints with sparse retention | 50 new regression cases pass, 277 total selected suite tests pass | Smallest tested retention change that serves both endpoints |

The first dense attempt changed only the 256-token fixture and left the separate
128-token cancellation fixture sparse. It had 25 passes and one failure. That run
is retained; the corrected dense experiment chooses the hit-page spacing in both
fixtures and passes all 26. This is a test setup correction, not evidence against
dense retention. None of these results rehabilitates the four unrelated HiSparse
failures; those remain documented and deselected from the final full CPU suite.

## Native retention cost

The following counts come from the actual cached encoder blocks in a one-encoder
group native fixture, after freeing the producer. Global block counts are equal
between alternatives, and private decoder/drafter hashed counts are zero throughout.
Each extension adds 129 tokens.

| Prompt tokens | Original sparse: encoder blocks / hit | Selected sparse: encoder blocks / hit | Dense: encoder blocks / hit |
| --- | --- | --- | --- |
| 65,856 | 128 / 65,536 | 129 / 65,792 | 257 / 65,792 |
| 262,464 | 512 / 262,144 | 513 / 262,400 | 1,025 / 262,400 |
| 1,000,256 | 1,953 / 999,936 | 1,954 / 1,000,192 | 3,907 / 1,000,192 |

The selected policy uses one more encoder block for the complete endpoint at all
three contexts. Dense retention uses the same endpoint but increasingly many
intermediate pages. At the recorded seven encoder groups and 227,840-byte block,
the selected policy's extra endpoint corresponds to 1.52 MiB per rank. At 65,856
tokens it retains about 194.7 MiB fewer pages per rank than dense retention. These
are extrapolations of native block counts to recorded geometry, not measurements
of GPU allocation or physical memory savings; all entries occupy the same fixed pool.

## Why two endpoints are sufficient for this geometry

Let N be the completed producer prompt, A the appended tokens, and B=256 the
global hit-page size. With private replay R=128, no request can legally hit past

    max(0, floor(min(N, N + A - R) / B) * B)

even if every encoder block is retained. For A >= 0 and R < B, this upper bound
can take only the resend endpoint or the final complete producer endpoint.
The two anchors retain those positions. Removing either fails native tests; extra
intermediate anchors cannot improve an appended request's hit beyond this bound.

The supplemental sweep checks every prompt length 4,096..4,607 (all 256-token
page offsets over two pages), extensions 0/1/127/128/129/256 and retention
0/256/512/4096. All 12,288 combinations reach that upper bound; two completion and
eviction checks also pass, for 12,290 total. The original 50 cases separately cover
chunked prefill and a still-live producer/concurrent consumer. Peak RSS for the
sweep was 1,173,800 KiB; CUDA remained uninitialized.

The missing-tail improvement is at most one 256-token page for this geometry.
The 128-row private replay and partial-page remainder remain. This bounds reusable
tokens, not the time saved: kernels, scheduling, transfer, cache pressure and the
rest of prefill may dominate an actual request.

## What is not proved

Dense retention can help a fork or edit in the middle of a prompt, where the two
tail anchors are not the relevant state. Whether that benefit justifies its larger
cache footprint needs actual distinct-agent traffic and eviction measurements.
The CPU results select sparse endpoint retention for append/resend workloads; they
do not establish a universally best retention policy for all branching workloads.

Saving/restoring decoder, drafter and compressor working state could conceivably
avoid the remaining 128-row replay. That is a different state representation and
ownership design, with partial-page, cancellation, batch invariance and memory
costs not proved here. No such implementation is imported or proposed by this patch.
Adaptive age/turn-dependent retention similarly needs a workload-weighted eviction
study; changing the existing LRU policy is not necessary to reach the append bound.

The live serving gate is still pending: native token/logprob equality under cache
reuse and concurrency, actual cancellation, multi-turn TTFT, throughput, memory and
eviction under two independent boots. Source arithmetic is unchanged, but that
alone does not prove batch-invariant model outputs. An absolute optimum cannot be
established by these finite CPU tests. This is a focused implementation supported
by cache correctness and a mechanism bound, not a fully qualified performance result.
