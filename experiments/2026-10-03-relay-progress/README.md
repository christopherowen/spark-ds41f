# Relay progress and GPU-initiated networking investigation

Base: `441c97f`. No production changes or promotion. Investigate separately:

1. Time spent posting local sends, waiting for relay readiness, forwarding,
   draining completions, and waiting for queue capacity in the existing proxy.
2. Whether the pinned GPUNetIO one-sided library can build and run on the
   existing Spark kernel/driver/RDMA stack, using explicit CPU-proxy mode and
   CPU/GPU shared buffers where required. Direct GPU doorbells need a separate
   capability test; automatic fallback is not evidence of support.

Build: `bash experiments/2026-10-03-relay-progress/build-gpunetio.sh`.
External source: NVIDIA-DOCA/gpunetio at
`586453728bcab2d4c50574924dc6cf43543c9ed4`; no host package installation.
The isolated build initially uses the upstream samples unchanged. All build
and run failures are evidence and will be retained.

Use the shared cluster window, bounded model-free probes, identical setup on
all four nodes, and restore the idle entry state. A proxy-instrumented run is
not an uninstrumented performance measurement. Retain the existing directional
split, reduction order and two-slot lifetime until a replacement has a tested
ownership protocol.
