# Decision

Hold candidate-38 TP4 release qualification. The owner-approved window produced
one valid 8-session measurement, then stopped safely at the sp2 thermal-margin
guard during 16-session warmup. Both repeat points remain unrun. Candidate-37
was restored and independently verified on all four ranks; no hold remains.
No serving, hardware or promoted configuration change is proposed.

Before release, address these in order:

1. Qualify the site's supported cooling under sustained cold prefill, especially
   sp2. The guard reached 5 C thermal margin; auxiliary sampling saw a maximum
   83 C core temperature. No thermal-slowdown counter growth was recorded, so
   this is a guard-limited window, not proof of actual throttling or failed
   16-session decode. Keep the limits; cooling or hardware changes need their
   own authorization.
2. Investigate the preserved 327,905-token warmup request that returned no
   content. Its native completion timestamp, 02:16:01.543 UTC, precedes the
   thermal cancellation at 02:17:15.549. The shutdown's later warmup-failure
   message does not explain that earlier error. No response text/wire capture
   is available, and the cause remains unresolved.
3. Review seven shortened long outputs associated with repetition containment in the valid
   replay. Each shortened output matches a server repetition-counter event;
   they are separate from three deadline cancellations. A valid AgentX
   submission does not establish that those outputs were useful or that the
   containment has no false positives. Retain the scored long-context loop,
   copy and compacted tool-session canaries.
4. After the above review and any required authorization, complete a new full
   matrix with a fresh boot and repeat at both session counts, preserving this
   failed window. Complete the outstanding TP4 inference, deterministic/KL,
   long-context, mixed-load and release/single-stream/sampled gates. Compare
   with immutable recorded baselines; never remeasure the released build.

The single valid snapshot delivered 157.52 output tokens/s and 24.92 median
tokens/s/user, with TTFT 1.136 s median and 7.870 s p99. Cache reads were 96.62%
against 96.98% theoretical reuse; profiling KV peak was 19.81%, with zero
preemptions. Minimum head MemAvailable was 15.61 GiB, with no recorded swap
allocation growth. Those observations do not support prioritizing larger KV
capacity or cache-retention tuning for this sample. Accepted draft tokens
averaged 1.65 of 2.74 verified per round; cost profiling is needed before a
speculative-policy experiment can claim a benefit.

There is no repeat spread, confidence interval or proven gain. No direct
comparison with the earlier TP3 replay is valid. AgentX replays request shapes
and does not score useful task completion or replace the remaining correctness
and release gates. The existing passing image tests and two-boot TP3 focused
reproductions do not supply the missing TP4 evidence.
