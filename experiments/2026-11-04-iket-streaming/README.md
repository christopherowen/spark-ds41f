# Streaming IKET collection

Owner, 2026-11-04: implement a reusable collector whose total capture size is
limited by disk, complete bias/slot-mapping instrumentation, and check sparknet
coverage. Authorized hardware: idle Spaceport sp8 with candidate-38. Both serving
clusters remain running. No production configuration or baseline changes.

Base identities are in base.json. The NVIDIA collector is binary-only here and
has no supported reclamation API. This experiment tests an opt-in collector
using the existing TileLang phase annotation, a fixed CPU/GPU-visible ring,
system-scope publication, and a native background binary writer. Ring slots are
reused after copying into a bounded write batch. Slow or failed storage applies
backpressure or fails collection explicitly; it never silently overwrites data.
This alternate collector does not instrument arbitrary ATen/NCCL SASS.

Variables are screened separately: ring transport correctness/backpressure;
phase collection; previously unsampled runtime loops; sparknet coverage. Every
hardware point boots twice, retaining failures, exact commands, memory and
identity receipts. Full-fabric tests require an idle authorized multi-node window
and cannot use the serving ranks. A one-node test is not fabric qualification.

Acceptance: output bytes exceed ring capacity many times while resident trace
memory stays bounded; no missing/repeated sequence IDs; slow/disk-error cases
behave explicitly; CUDA-graph and multi-stream captures retain identity; sampled
runtime loops have finite event bounds; networking phases appear in manifests
and exported traces without changing ordinary kernels when profiling is off.

Results pending. No performance claim or promotion.
