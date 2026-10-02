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
