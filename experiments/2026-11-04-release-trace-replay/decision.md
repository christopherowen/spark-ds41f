# Decision

Pending measurements and explicit authorization for the coordinated test window.
No promotion or tuning change is proposed.

Before release: complete candidate-38 TP4 inference and performance qualification
on a boot and repeat boot; retain the long-context loop canary and a scored long
compacted tool session; verify stable host memory through replay and export.

The candidate image already has a passing release-test gate and two-boot TP3
reproduction evidence. Those results do not establish TP4 throughput, task success
or the absence of spontaneous loops in longer sessions.

Only after valid replay results should cache retention, KV capacity, prefill
stalls or speculative acceptance become individual tuning experiments.
