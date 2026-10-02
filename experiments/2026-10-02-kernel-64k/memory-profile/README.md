# Matched page-size profiling

Status: the approximately 3 GiB loaded-model overhead is attributed. An opt-in
UVM recovery candidate is being tested in [uvm-pool](../uvm-pool/README.md).
No default or production serving configuration has changed.

All runs use r5o, unchanged TP3/model/KV/graph budgets and driver 580.178.04.
`cluster-pinned.json` adds the same read-only draft-cost tables to both arms.
Fresh boot each arm; no CUDA smoke before these matched startups. Dynamic
profilers stop before benchmark timing. Host memory controls are those recorded
in [the boot trial](../boot-test/README.md). Thus this compares the tested host
profiles, not an isolated compile-time PAGE_SIZE change.

## Allocation attribution

On dgx3, charged pages whose call stack goes through UVM `phys_mem_allocate`:

| Arm | Retained bytes | GiB |
|---|---:|---:|
| 4 KiB, original allocator | 218,243,072 | 0.20325 |
| 64 KiB, original allocator | 3,491,823,616 | 3.25201 |
| 64 KiB, expandable segments disabled | 3,509,583,872 | 3.26855 |

The first two differ by 3.04876 GiB. Most allocations are 256-byte GPU leaf
page tables. NVIDIA's sysmem allocator obtains `alloc_pages(get_order(size))`;
each occupies at least one CPU page. Approximately 53,000 live tables therefore
cost approximately sixteen times as much with 64 KiB CPU pages. The stack is
`phys_mem_allocate → allocate_directory → uvm_page_tree_get_ptes_async →
uvm_page_table_range_vec_init → uvm_map_external_allocation_on_gpu`.
These are allocation sizes for GPU translation tables, not the GPU data page
size or the model's KV blocks.

The cgroup `kernel` aggregate independently rose from approximately 0.349 GiB
to 3.389 GiB. The NVIDIA RM allocation tracer, by contrast, reports approximately
102.34 GiB on both arms, differing by only 9.35 MiB. CPU anonymous memory also
increased, but is secondary. Driver table backing is the dominant regression.

Turning off PyTorch expandable segments did not recover it. This negative arm
is retained as `cluster-native-allocator.json`; do not promote that setting.

`kernel-charges.bt` traces kernel-page charges and uncharges. It is not a
complete memory accountant: socket/pipe pages can be reused without the
corresponding uncharge probe firing. The corrected tracer records those
replacements and avoids double counting. No replaced-page attribution in the
4 KiB control involves UVM. Older 64 KiB traces have 7–8 duplicate events;
those are small compared with the observed 3 GiB difference. Native logs and
full host captures remain in `results/private/kernel64k-profile/` on dgx3 and
in the local trial worktree; hashes and compact results are in
`matched-summary.json`.

## Performance before recovery

Native reports: `bench-4k.json`, `bench-64k.json`. Five decode samples per point;
three prefill samples. Prose/code reasoning on, concurrency 1 and 8. Both stock
arms used the Mac client. Actual source-text prompt lengths, read from each
native report's samples, are 1,006 / 952 / 1,077 tokens for nominal 1K;
29,569 / 28,442 / 28,265 for nominal 32K; and 56,157 / 59,016 / 57,537 for
nominal 64K. They match between these two stock arms. No swap growth or thermal
slowdown observed.

| Metric | 4 KiB | 64 KiB |
|---|---:|---:|
| Prose c1 tokens/s | 52.50 | 50.70 |
| Prose c8 tokens/s | 169.00 | 168.70 |
| Code c1 tokens/s | 63.16 | 65.90 |
| Code c8 tokens/s | 194.66 | 194.40 |
| Prose c1 step, ms | 41.81 | 40.93 |
| Code c1 step, ms | 45.83 | 46.36 |
| Prefill nominal 32K, tokens/s | 3,827.9 | 3,838.1 |
| Prefill nominal 64K, tokens/s | 3,783.1 | 3,802.1 |

These screens show no clear throughput improvement from the unmodified 64 KiB
profile. They do not establish that memory recovery will or will not improve
performance. The driver candidate must be measured separately, without probes.
The benchmark files also retain TTFT, acceptance and confidence intervals;
generated-output changes make tokens/s noisier than verification-step timing.

## Errors retained

The first replacement-accounting revision failed to compile under bpftrace
0.20 because its map type was inferred after a cast. Explicit map initialization
fixed it; the successful 4 KiB control is `kmem-4k-v2.txt`. A deployment attempt
with the uncommitted candidate patch was refused by the clean-tree guard; no
service was started by that attempt. A transient SSH interruption during sync
recovered before retry. Neither changed serving or memory guard policy.
