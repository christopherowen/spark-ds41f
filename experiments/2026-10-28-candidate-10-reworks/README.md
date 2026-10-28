# Candidate 10 reworks

Changes reworked on candidate 10 (rc10: candidate 9 plus the block-scale grid reads), one
per spec, by the one-at-a-time procedure (`../2026-10-27-one-at-a-time/README.md`): kernel
bundles first, then the bit gates (determinism with and without logprobs, the seeded sampled
digest) on the base and the arm, then the lean suite on the base, the arm and the base again.
`make_reworks.py` writes the bundles, arms and specs from two vLLM commits.

## coded-arrive (`coded-arrive.json`, against candidate 10)

The grouped MoE GEMM copies each stage's scale codes with cp.async from a few threads, and
every thread then expands one row's codes into the scale words the MMA reads. The expand runs
after `ptx_wait_group` (which waits for the thread's own copies only) and the wait on the
stage's mbarrier, whose arrivals (`T.barrier_arrive`, a plain `mbarrier.arrive`) happen when
the copies are issued, not when they land. So in candidate 9 and 10 a thread can read codes
another thread's copy has not yet written; the next `sync_threads` comes after the expand.
No run has shown wrong bits (the copy was issued a stage earlier), but nothing orders it.

The fix arrives with `cp.async.mbarrier.arrive.noinc` (`cp_async_barrier_noinc`): each
thread's arrival waits for its copies, so the stage's barrier publishes every row. Claim:
same bits, same speed. Bundles: `moe-c10` (candidate 10's MoE tests, the control) on dgx1,
`coded-arrive` (the same tests with the fix) on dgx2.

## expert-codes-v2 (`expert-codes-v2.json`, against coded-arrive)

The first expert-codes arm (one-at-a-time) was bit-exact but 2.7-2.8% slower at prefill. It
had worked around the race above by making each copying thread expand its own codes: 16
threads expanding 8 rows each (16 for a narrow expert), with the row bases in shared memory,
on every K block's critical path, for every expert, not only narrow ones. On coded-arrive
every thread expands its own row again, from the 2-bit or the 4-bit codes (one expert per
CTA, so the branch is uniform), with the base in a register, as candidate 9 does.

Ceiling: 4-bit codes were bounded at 1.3-1.9% of a decode step
(`../2026-10-18-scale-codes/README.md`); 2-bit codes halve the code bytes again for narrow
experts only, so under about 0.9%, likely 0.3-0.5%. Claim: same bits, prefill no slower than
coded-arrive, decode no slower. Bundle: `expert-codes-v2` (the MoE and scale-code tests) on
dgx3.

## API prompt semantics (`api-semantics.json`, kernel lab only)

The DeepSeek API work (effort aliases; response_format, tool_choice none and the assistant
prefix in the V4.1 prompt) ported onto candidate 10's head: its oracle passed on the
laptop, but its tokenizer tests need vLLM's dependencies, which only the image has.
`make_api_bundles.py` runs them there: `api-c10` (candidate 10's own copies, the control)
on dgx1, `api-semantics` (the branch's tests and sources) on dgx2.

## Markov projection in persisting L2 (`markov-l2.json`, kernel lab only)

Byte lever 7 of the decode-bytes study: the DSpark Markov projection (a TP4 rank's 16.5 MB
BF16 shard) is read by each of a step's five draft positions, and serving's per-call times
(40-68 us against 71 cold) say most re-reads miss L2. Bound: +0.5% of a single-stream step.
Placement only, so the bits cannot change. `bundles/markov-l2` times one step's five calls
(the production `bf16_gemm` at the 16-row tile, a base-logits read and an argmax between
them) as a CUDA graph on cold L2, with and without a persisting access-policy window on the
weight, checks both give the same bits, and reports the device's persisting-L2 limits. A
vLLM change follows only if the window saves time here.
