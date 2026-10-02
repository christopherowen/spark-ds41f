# Experimental UVM leaf-table packing

Status: candidate prepared; not validated or promoted. Disabled by default.

Base: Ubuntu `nvidia-kernel-source-580-open=580.178.04-0ubuntu0.24.04.1`,
installed source `/usr/src/nvidia-580.178.04`, corresponding upstream tag
`NVIDIA/open-gpu-kernel-modules` `580.178.04`. No upstream submission yet.
The build script checks both modified input files by SHA-256.

The 64 KiB allocation trace attributes approximately 3.25 GiB to small UVM
page tables. The allocator requests 256 bytes for leaf tables, then obtains
one CPU page via `alloc_pages(get_order(size))`. The experiment gives each
256-byte user leaf table its own 4 KiB slot in a 64 KiB page. It deliberately
does not pack root tables or other allocation sizes.

The existing whole-page allocator retains memory-cgroup ownership, DMA address
translation, physical-TLB invalidation and mapping accounting. Pools belong to
one page tree. A leaf spinlock protects slot ownership; no allocation, mapping
or tracker wait occurs under it. The original tracker wait precedes slot reuse.
The final slot releases the original mapping and page. CPU mapping already
includes the physical address's page offset.

The path requires `uvm_pack_sysmem_leaf_tables=1`, 64 KiB CPU pages, a real
integrated GPU with no separate VRAM, coherent DMA, a user page tree, and an
exact 256-byte allocation request. All other paths retain the stock allocator.
It changes address-translation storage, not model arithmetic. Nevertheless,
errors here could corrupt GPU memory, so build success is not validation.

`build.sh` copies the installed source into an ignored, isolated build directory,
applies the patch and builds NVIDIA/UVM against the staged 64 KiB headers.
It never installs, signs or loads a module and never updates initramfs.

Required validation before any promotion: build; bounded CUDA allocation,
read/write, reuse and teardown stress; large pinned transfer and graph replay;
model load; three-rank quality and memory screen; profiler-free matched TPS and
TTFT measurements. Keep packaged modules and 4 KiB boot default available.
Results and exact loaded module hashes must be recorded separately.

## Trial outcome (2026-10-02)

Built against the staged 64 KiB headers with GCC 13, signed using the existing
enrolled fleet key, and loaded temporarily with `insmod`. Only UVM was replaced;
the rebuilt RM module was not loaded. Packaged modules, initramfs and module
configuration were untouched. A normal reboot restores the packaged driver.
All ranks loaded source identity `2A94659FBB6D360FF27D9A9`; signed module SHA-256
and exact parameters are in `trial-results.json`. Release assertions were on,
with assertion failures configured to set a global UVM error.

The direct allocation test passed 3,584 allocations across four threads with
hole reuse and full-buffer readback, on both stock and patched UVM. The patched
module additionally passed the 4.5 GiB pinned transfer, BF16 matmul and CUDA graph
smoke check. The first test-harness attempt failed equally on stock and patched
modules because `torch.cuda.init()` had not bound a current CUDA context; an
explicit small allocation fixes the harness. No patched-module GPU fault or UVM
assertion was observed through the serving screen.

The same three-rank model loaded, all five serving quality checks passed, and
`bench-64k-pool.json` completed without failed requests, swap growth or thermal
slowdown. Profiler-free measurements use the same pinned draft costs and workload
as `../memory-profile/bench-4k.json` and `bench-64k.json` (plus a quality gate).
CUDA smoke preceded this candidate startup; the stock matched arms went directly
from boot to serving. Do not treat those startup times as a controlled comparison.

UVM page-table backing on dgx3 fell from 3.25201 to 0.21771 GiB, a 3.03430 GiB
reduction. The corresponding cgroup kernel aggregate was 0.3464 GiB after startup,
versus approximately 3.389 GiB with stock 64 KiB UVM. This confirms that the opt-in
packing path actually ran; the result is not just a successful no-op module load.
The corrected trace had eight replacement records, none attributed to UVM.

Minimum available memory during serving was 8.55 / 9.56 / 9.55 GiB on ranks
0 / 1 / 2, compared with 6.72 / 7.70 / 7.73 GiB on the matched 4 KiB control.
The net gain over 4 KiB is therefore 1.82–1.86 GiB per node in this screen.
Single-stream steps were 41.55 ms prose / 45.91 ms code, versus 41.81 / 45.83 ms
at 4 KiB. This is a memory recovery, with no established throughput gain.
A fresh 4 KiB repeat checks the apparent TTFT improvement before interpretation.

The first benchmark launch could not create its output directory inside the
root-owned capture directory. It sent no benchmark requests. Creating only that
output directory with the ordinary user's ownership fixed it; the benchmark
itself ran unprivileged.

## Operational boundary

This remains an experimental driver patch, not a persistent installation.
A production package would need an exact kernel/driver build identity, signature
verification, conditional activation only for the matching 64 KiB kernel, and an
unmodified 4 KiB fallback. A global modprobe option is insufficient: the stock
4 KiB UVM module does not recognize the added parameter. No such option was
installed. Driver/kernel upgrades must not silently remove the fix or load an
incompatible module. Prefer an upstream allocator fix over maintaining a fork.
