# Consumed-route capture prototype

Status: CPU contracts only; native integration and the serving capture profile
are not qualified. This is separate experiment instrumentation, not source
PR #5 and not performance evidence.

Source commit: `228e900b90bb549e353958d17420497ca6248d81`; tree: `a7952c73fc3264bc1bdabec0deeeac23cb5c74ba`.
Apply this directory's patch after the two swizzle patches. The named source
branch is `feat/expert-route-capture`; its parent is
`f3bfdf8b8c08870754cc3c756b75faa9156595e0`.

The module records the exact target top-6 IDs and hashes of route weights at
`TileLangExperts.apply`, immediately before `MoeExpertsV41Plan.run`. It uses the
layer's own SWA metadata key, including shared-cache ownership, and its explicit
CED decoder tag. Full metadata buffers are sliced to the consumer extent;
compact decoder rows are joined to the full scheduler by request and position.
GPU query starts remain authoritative under adaptive verification. The first
kept CED draft is never relabelled an anchor just because it is first in a compact
buffer. Capture occurs before sampling, so rejected verification rows are kept.
No request-response accepted-token projection is substituted for actual calls.

The collector refuses graph execution, non-eager configuration, DBO, PCP,
batch sharding, data/pipeline parallelism, multi-anchor steps, unsupported
geometry and ambiguous/missing scheduler coordinates. It snapshots GPU tensors
and drains them after the target forward, before drafting. This alters timing
and may affect batching. It does not mutate IDs, route weights or model tensors.
It is target-only (384 experts/top-6), not a drafter-route collector.

A node-local `SPARK_EXPERT_ROUTE_CAPTURE_CONFIG` JSON file supplies workload and
capture names, exact workload SHA-256, source tree, model revision, source-file
hashes, output directory and selected integer layer IDs. The collector verifies
all source hashes. Default limits are two prefill, four mixed and sixteen decode
steps, 96 complete invocation records, 16,384 rows per selected step and 64 MiB
of JSONL. A too-large invocation fails; byte/record limits stop between complete
records. Every result retains step/layer, physical shard geometry, per-expert
exact scale widths, source/workload identities and a private summary. Files use
mode 0600 in a 0700 output directory. Only anonymous request labels, positions,
IDs and hashes are saved; token IDs and prompt text are excluded.

Ten CPU tests exercise mixed decode/prefill, CED subsets, authoritative adaptive
starts, dead/padded rows, mismatched/duplicate coordinates, invalid IDs and slot
order. New-module ruff and syntax checks pass. The affected source files' ruff
run also reports an existing unused import and long line in `nvidia/model.py`;
its raw output is retained. These checks do not establish real captures or model
invariance. A separate capture arm needs clean published source/deployment
identities, fixed code/prose/distinct-agent workload manifests and the existing
coordinated serving window before execution.
