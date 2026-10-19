# Within-process memory placement

Base: r6x image on nonserving DGX4. Allocate twenty 2 GiB CUDA buffers in one process, after reserving the 3 GiB guard plus 2 GiB workspace. Stop if the reserve or guard would be breached. No clocks, allocator settings or system state change.

Read every buffer contiguously and gather six fixed distinct 18.8 MB chunks per buffer. Check partial-tile checksums before the run, warm all buffers, then measure 100 repetitions in a seeded shuffled buffer order with CUDA events. Keep all samples and 95% t intervals; report buffer spread separately for the two patterns. Physical contiguity and GPU page sizes are unavailable through this allocator API, so virtual addresses cannot establish either.

Run through the kernel lab on the spare node. No production or candidate promotion is implied.
