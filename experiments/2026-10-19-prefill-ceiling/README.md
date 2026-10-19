# CED prefill cost on r6x

Base: e0f962a7, promoted r6x image (sha256:6339bda1824733cba3c35f648f952e40c119e4fb3d8ce8c0d40a3c74db4224d8), DeepSeek-V4.1-Flash-1017 on the three-node triangle at 1M context.

Measure one cold 131,072-token real-text prefill after warming the kernels with a separately salted request, then a one-stream decode window. The sole runtime variable is torch profiling. Keep all rank traces and separate the encoder from the CED decoder, including the full-resolution KV-source boundary.

Use the production recipe for restoration. No weights, sampling, sizing, clock settings or promoted configuration change. The CED work proceeds if its measured ceiling is at least 2% of the 128K prefill. A single profile estimates the ceiling; it is not a performance acceptance comparison and has no independent-run 95% interval.

Run: `scripts/lab.py run experiments/2026-10-19-prefill-ceiling/profile.json`. The lab restores production when the run ends. TP4 remains unmeasured until the site is cabled as the ring.

First attempt: the full-request, shape-recording profiler completed the cold request but crossed the steady memory floor while exporting at 06:41 UTC. The guard stopped the profiling container at 2.58 GiB available (3 GiB floor); no trace survived. The lab restored production. Keep this failed attempt. The bounded retry records three non-final iterations after 20 wait and one warm-up iteration, with shape retention and aggregate CUDA tables disabled. Sizing and memory guards stay unchanged.
