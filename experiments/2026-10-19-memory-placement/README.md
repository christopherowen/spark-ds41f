# Within-process memory placement

Base: r6x image on nonserving DGX4. Allocate twenty 2 GiB CUDA buffers in one process, after reserving the 3 GiB guard plus 2 GiB workspace. Stop if the reserve or guard would be breached. No clocks, allocator settings or system state change.

Read every buffer contiguously and gather six fixed distinct 18.8 MB chunks per buffer. Check partial-tile checksums before the run, warm all buffers, then measure 100 repetitions in a seeded shuffled buffer order with CUDA events. Keep all samples and 95% t intervals; report buffer spread separately for the two patterns. Physical contiguity and GPU page sizes are unavailable through this allocator API, so virtual addresses cannot establish either.

Run through the kernel lab on the spare node. No production or candidate promotion is implied.

## First DGX4 run

Both checksum modes pass; minimum MemAvailable is 76.70 GiB. The contiguous read has a 1.14% range across buffer means. Buffer 13 takes 8.1068 ± 0.0080 ms, while buffer 16 takes 8.1998 ± 0.0109 ms (95% intervals, 100 repetitions); their intervals are separated. The six-expert gather has a 0.60% range, but its individual intervals overlap, so that pattern is inconclusive. All observations and samples are retained under results/private/lab/postlaunch/memory-placement on DGX4.

Recommendation: memory placement remains a supported suspect. Repeat within-process allocations and attribute per-rank kernels across boots before choosing an allocator intervention. Virtual addresses do not identify the underlying physical mapping. No allocator, huge-page or affinity changes are promoted.
