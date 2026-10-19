# CED prefill cost on r6x

Base: e0f962a7, promoted r6x image (sha256:6339bda1824733cba3c35f648f952e40c119e4fb3d8ce8c0d40a3c74db4224d8), DeepSeek-V4.1-Flash-1017 on the three-node triangle at 1M context.

Measure one cold 131,072-token real-text prefill after warming the kernels with a separately salted request, then a one-stream decode window. The sole runtime variable is torch profiling. Keep all rank traces and separate the encoder from the CED decoder, including the full-resolution KV-source boundary.

Use the production recipe for restoration. No weights, sampling, sizing, clock settings or promoted configuration change. The CED work proceeds if its measured ceiling is at least 2% of the 128K prefill. A single profile estimates the ceiling; it is not a performance acceptance comparison and has no independent-run 95% interval.

Run: `scripts/lab.py run experiments/2026-10-19-prefill-ceiling/profile.json`. The lab restores production when the run ends. TP4 remains unmeasured until the site is cabled as the ring.

First attempt: the full-request, shape-recording profiler completed the cold request but crossed the steady memory floor while exporting at 06:41 UTC. The guard stopped the profiling container at 2.58 GiB available (3 GiB floor); no trace survived. The lab restored production. Keep this failed attempt. The bounded retry records three non-final iterations after 20 wait and one warm-up iteration, with shape retention and aggregate CUDA tables disabled. Sizing and memory guards stay unchanged.

The bounded run saved all three prefill traces (130,567 prompt tokens, warm 29.515 s, profiled 31.267 s). Its decode stopped at EOS after 107 tokens, before the 20-step wait and active window, and produced no decode trace. The v2 run forces 512 decode tokens and adds a tested one-row raw-token workload for the decoder's one-row comparison. Missing rank traces now fail the lab step.

The one-row probe itself completed, but a second scheduled profiling window on the same r6x boot produced no trace: the worker reuses a torch profiler whose one-shot schedule has already finished. `profile-short.json` boots once per short workload so each schedule is fresh. This avoids changing profiler runtime code in the ceiling comparison. Keep the v2 prefill traces and missing second-window outcome.

## TP3 cost gate

A second bounded cold prefill and fresh-boot single-token target forward produced traces on all ranks. These are profiling costs, not an A/B throughput result. `servebench.profile.ced_moe_costs` checks the projection geometry and decoder grouping before attribution. The routed-FFN subtotal excludes the boundary's unchanged full-row KV construction.

| Rank | Decoder routed FFNs at 128 rows, ms | At 1 row, ms | 128-row decoder body span, ms |
|---|---:|---:|---:|
| DGX1 | 80.58 ± 10.16 | 3.56 | 110.92 ± 8.75 |
| DGX2 | 80.33 ± 10.19 | 3.44 | 110.90 ± 8.72 |
| DGX3 | 80.00 ± 9.08 | 3.66 | 110.98 ± 8.79 |

Intervals are 95% t intervals over four decoder groups within one profile, not independent boots. The one-row trace contains one target forward, so its interval is unavailable. The body span runs from the first routed projection through the last; it includes intervening decoder work, excluding the first attention preparation and final tail.

At 130,567 tokens and threshold 4,048, 32 nonfinal chunks have at least 199 following tokens. The routed subtotal alone gives roughly 2.4–2.5 s removable work against the approximately 30 s cold prefill: an 8–9% modeled throughput opportunity. Proceed with the skip after the pure ports pass; do not call this an observed speedup. TP4 remains unmeasured.
