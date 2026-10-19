# CED prefill cost on r6x

Base: e0f962a7, promoted r6x image (sha256:6339bda1824733cba3c35f648f952e40c119e4fb3d8ce8c0d40a3c74db4224d8), DeepSeek-V4.1-Flash-1017 on the three-node triangle at 1M context.

Measure one cold 131,072-token real-text prefill after warming the kernels with a separately salted request, then a one-stream decode window. The sole runtime variable is torch profiling. Keep all rank traces and separate the encoder from the CED decoder, including the full-resolution KV-source boundary.

Use the production recipe for restoration. No weights, sampling, sizing, clock settings or promoted configuration change. The CED work proceeds if its measured ceiling is at least 2% of the 128K prefill. A single profile estimates the ceiling; it is not a performance acceptance comparison and has no independent-run 95% interval.

Run: `scripts/lab.py run experiments/2026-10-19-prefill-ceiling/profile.json`. The lab restores production when the run ends. TP4 remains unmeasured until the site is cabled as the ring.
