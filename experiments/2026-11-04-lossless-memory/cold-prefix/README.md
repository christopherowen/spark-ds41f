# Exact cold-prefix storage gate

The CPU store passes namespace, byte identity, malformed-input, quota and atomic
publication tests. It is not connected to vLLM and has no measured TTFT benefit.
Weight-tile compression ratios do not predict this cache state's ratio. Screen
only real complete-state exports before choosing its codec and block size.

The promoted-source compatibility gate is recorded in `compatibility.json`.
The generic recurrent checkpoint adapter excludes `deepseek_v41`; this model
has private decoder/draft windows and compressor circular state. The native
filesystem tier's scheduler reads a host-local `/dev/shm` region, which alone
does not prove that all three hosts' rank shards reach disk. Do not enable that
path merely because its byte codec round trips.

A second possible design stores only complete, prefix-cacheable blocks and
uses the ordinary warm-prefix replay to rebuild private state. That design
needs an explicit selected-group mapping, committed written-span export and
per-rank storage/restore proof. The current prototype implements neither serving
design and remains behind this compatibility gate.

Reproduce the static audit without loading weights or initializing CUDA:

```sh
python3 experiments/2026-11-04-lossless-memory/cold-prefix/check_compatibility.py \
  --repo .work/upstreams/vllm --out INPUTS/cold-prefix-compatibility.json
```

The next measurement is a complete request-boundary export at 32K, 131K and
262K, at the selected shape's unchanged batch and context limits. Record every
cache group's logical positions, dtype, shape and written byte spans, plus the
target/draft auxiliary state and draft-prefix boundary. Pin the original pages
until a capture fence completes. Never read the allocation's unwritten tail or
serialize an entire pool because its stride is convenient.

Use the pinned checkpoint/tokenizer, serving tree, TP rank, cache layout,
steering identity and full token/multimodal prefix hash as the namespace. Store
and restore are new states of one experiment; compare against the matching
recorded cold-prefill workload, not a newly measured released build. Include
store cost and misses, not just successful cache loads.

For the cold-file condition, fsync and advise away only this experiment's cache
file ranges. Do not clear the host's global page cache. Record both cold and
warm file-cache reads, the actual bytes stored, decoded bytes, host-memory peak,
replay tokens, prefix hits, TTFT and all errors.

Force the original KV pages out before the reuse request. Reserve new pages
through the scheduler, restore every required record and auxiliary field, wait
for all rank acknowledgements, then publish the hit. Verify greedy continuations
and teacher-forced distributions against the exact original state, including
short prompts, partial final blocks, long chunks, live neighbours and sampled
requests. A failed restore must not publish any partial checkpoint.

Two candidate serving boots and a production agent workload are required before
release. If storage ratios, complete-state handling or restored-request latency
do not justify the feature, retain the unfavorable evidence and reject it.
