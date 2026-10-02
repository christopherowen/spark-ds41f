# Driver and CUDA refresh qualification

User-authorized evaluation on all three Sparks, from the exact base in
`base.json`. Production uses R580, 64 KiB pages, memory-saver 0.2.0, r5o,
3.5 GiB KV per rank and 524,288-token context. Keep those capacities and the
5/3 GiB startup/steady guards throughout.

Compare the current container on R580 and R610 first, then a separate CUDA
13.3 container on R610. Freeze DSpark costs across the arms. The R610 allocator
port changes surrounding source context only; its hardware qualification is
part of the driver arm. The candidate kernel policy belongs to this experiment;
`host.kernel_policy` lets doctor and start validate it without changing defaults.

Before mutation, cache the exact replaced R580 packages, simulate the R610
transaction, and check the enrolled signing key. Use Ubuntu precompiled module
metapackages for both kernels; do not introduce NVIDIA DKMS beside memory-saver.
Stop the entire cluster before changing drivers. Build/install the pinned port,
reboot all nodes, check loaded identities, and run allocation/readback, copy,
matmul and graph smoke on each node before serving. Record failures as well as
successes. A failed management path requires diagnosis before another start.

Screen quality, one/eight-stream prose and code decode, and real-text prefill
at 4K, 16K and 64K. Compare step time, draft acceptance and TPS separately.
For small effects repeat the control. Promotion additionally needs long-context
and admission checks, matching node inventories, boot-log review, docs and a
new immutable baseline. Until then, restore the current R580 production stack
and close the hold. Results and decision are pending.
