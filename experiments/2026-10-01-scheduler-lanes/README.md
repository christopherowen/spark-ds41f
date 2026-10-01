# Prefill lanes and compute sharing (2026-10-01)

## Hypothesis

With one prefill lane, a short request that arrives while a long prompt prefills waits for that
prompt's whole prefill (about 15 s behind a 64K prompt). A second lane
(`--max-parallel-prefills 2`), or compute sharing between prefill and decode
(`--prefill-compute-share auto`), cuts that wait sharply. Neither should give up more than about
3% on single-stream decode, eight-stream decode, prefill throughput or mixed-traffic latency.
On 2026-09-27, eight lanes made four simultaneous 64K prompts wait 67 s on average instead of
41 s. Two lanes should give up much less of that, because two long prompts still mostly run in
turn.

## Intended delta

- Overlay `sched1`: r5o's `vllm/v1/core/sched/scheduler.py` plus two upstream liveness fixes:
  - `upstream-6f4b64421-…`: prefill lanes go to queued requests only while they are admissible;
  - `upstream-0b1bccb85-…`: decodes may preempt in a prefill turn that scheduled nothing.

  Both fix generation stopping with idle GPUs. The first applies with more than one lane, the
  second with compute sharing. Without either flag, neither code path runs.
- One flag per arm (`make_arms.py`): `control`, `lanes2`, `share`, and `lanes2-share` (the interaction).

## Quality and safety gates

Text must not change under the same batch (scheduling only). Memory: the steady memguard must
not fire, and dgx1's MemAvailable must stay at least 6 GiB. Rollback: the window closes and
restores r5o on any failed boot or job.

## Workloads

`scripts/lab_specs/lab4-scheduler-lanes.json`, run by `scripts/lab.py`, one boot per arm, with
control measured again at the end:
- `hol_latency.py --rounds 2`:
  - short requests at 2, 4, 6 and 8 s behind a cold 64K prompt;
  - four cold 32K prompts at once.
- `mixed_latency.py --rounds 3`.
- `c8_distinct.py --samples 2`.
- `c1_distinct.py --prompts 12`.

## Acceptance criteria

Adopt a flag only if the median TTFT of short requests behind a long prompt falls by at least
half, and four simultaneous long prompts' mean TTFT rises by no more than 10%. The other
workloads must stay within 3% of control, allowing for the control's own drift between its two
measurements.
