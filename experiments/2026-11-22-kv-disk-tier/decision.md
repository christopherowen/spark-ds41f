# Decision

Status: **retain for more evidence**

The status is one of the four outcomes that docs/methodology.md allows for an experiment: promote,
reject, retain for more evidence, or inconclusive.

During the planned outage from 23:44Z on 2026-11-24 to 01:00Z on 2026-11-25, the tier passed every
hardware check that ran. The equivalence check passed on its second call, after the first call failed on
a fault of the check itself (README.md, "Runs that failed"). The capacity check, in which many stored
sessions return, and the hung-read check did not run. The lab then kept the profile
`tp3-4k-b6144-m1m-kv35-nvme200` on as its serving profile, and the lab's traffic returned to the engine
at 00:59Z. In the first 12 hours of that traffic the tier served 125 disk loads, and its counters of load
failures, I/O errors, and failed stores stayed at 0 (`runs/kv-soak-12h.tsv`). Every disk hit that the
checks compared returned the same tokens and the same top-5 token sets as a pool hit, with its logprobs
within the check's tolerance. That tolerance was 0 for the seven long prompts of the equivalence check
and was not recorded for the other four comparisons (`results.json`, `runs/tier-gates.jsonl`).

The lab's decision to keep the tier on covers one lab, so this experiment does not ask for promotion. In
this repository a candidate is promoted only when the owner accepts it and the same commit updates the
configuration, the documentation, and a new immutable baseline (AGENTS.md, "Change discipline").
Promotion would first need an image built from `vllm/series`, so that no file is mounted over the image,
and a run, measured on two boots, on the promoted `tp3-triangle-4k` or another qualified shape. It would
also need runs of the paths that the hardware did not exercise: several restores at once, a queue of
loads, one load event that serves more than one request, a load past its time bound, a hung read, a
preemption, a rejection forced right after a disk hit, and a tier of 200 GiB on each rank that has filled
up and reuses its slots. Finally, the question about the sink refusal of KV transfer (README.md, "Sink
refusal of KV transfer") needs an answer, because the answer decides whether patch 9046 is right as it
is.

Against a pool hit of the same session, the disk hit made the 500,000-token request longer by 1.067 s
with 2 decoding streams and by 1.135 s with 4, and the load itself, as the gauge reports it, took 1.034 s
and 1.070 s. One run of each cannot separate a small cost of the tier to the decoding streams from the
spread between runs. With that limit, the streams' longest gap between tokens was 0.369 s and 0.378 s
with the disk hit and 0.412 s and 0.380 s with the pool hit, and their decode rate during the extra time
was 1.04 and 0.94 of the rate with no session sent. Against that baseline, the 500,000-token request
itself did cost the streams something, whether its prefix came from disk or from the pool. Two such costs
were measured for the disk hit: the decode rate while the request was in flight fell to 0.89 and 0.83 of
the baseline, and the longest gap between tokens rose from 0.092 s and 0.104 s to 0.369 s and 0.378 s.
