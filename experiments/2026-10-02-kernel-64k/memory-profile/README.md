# Matched memory profiles (in progress)

The owner requested attribution of the 64 KiB memory overhead and recovery of
reported memory/performance gains, rather than stopping at aggregate figures.
Default promotion remains undecided; the verified 4 KiB fallback stays normal
boot while this work runs in the held maintenance window.

`capture.py` reads host memory, slabs, vmalloc, DMA buffers, cgroup statistics
and process smaps without creating a CUDA context. Run as root at the same
startup/serving phases for both kernels. Raw captures live below ignored
`results/private/kernel64k-profile` until summarized and archived.

`nvidia-pages.bt` follows NVIDIA RM system/contiguous page allocations and actual
frees from container startup. It reports outstanding physical bytes/counts by
allocation stack and requested byte count; it does not cover all kernel or UVM
allocations. It validates the `nv_alloc_s` num_pages offset against the creator's
argument on every observed creation. Any bad_layout, map overflow, dropped
probe/event or failed profiler invalidates attribution. Its source assumptions
are from the installed open NVIDIA 580.178.04 module source; both kernels use
that version. Pass the actual OS page size as argument 1. Attach before startup
and terminate gracefully after the matched workload to print remaining maps.

This profiling adds overhead. Never use an instrumented startup or workload as
a throughput comparison. Timing tests must run without the probes and with the
same pinned draft-verification cost table on each boot. No model/dtype/KV budget
change is part of this experiment.

`cluster-pinned.json` differs from r5o only by a read-only cost-table mount and
`SPARK3_DSPARK_COST_DIR`. The two cost files are snapshotted from the existing
r5o experiment table on dgx1; their tracked bytes are identical across arms.
Trace NVIDIA allocations on dgx3, avoiding extra head-node memory pressure;
collect process/kernel inventories on every node.
