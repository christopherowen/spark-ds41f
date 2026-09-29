# KV cache in the display carve-out

**Base:** r5j (`2026-09-28-karmic-kraken-r5j`), image
`vllm-ds41f-kkref:04c30fa98e79-r5j`.

**Variable:** where the KV cache lives and how large it is. The promoted
configuration allocates 1.4 GiB of KV per rank from ordinary memory. The
`carveout` arm takes the whole backing, 2,032 MiB per rank, from the display
carve-out (vLLM patch 0021, `SPARK3_KV_DISPLAY_CARVEOUT=1`), leaving ordinary
memory free.

## The carve-out

The firmware reserves 2.10 GiB at the top of physical memory on every node
(`1fe30cc000-20693fffff`, reserved in `/proc/iomem`). The GPU driver turns
the `DISPLAY_FRM` part of it into a scanout heap
(`memmgrCreateScanoutCarveoutHeap_GB10B`). Its size comes from the firmware;
no module or kernel parameter changes it. Linux and ordinary CUDA allocations
never use it.

nvidia-drm serves DRM dumb buffers from that heap. Measured on 2026-09-29 with
`nvidia_drm modeset=Y fbdev=Y`, no reboot and no host change:

| Node | Display | Largest dumb buffer | MemAvailable change |
|---|---|---:|---:|
| dgx1 | none | 2,040 MiB | none (−0.06 GiB noise) |
| dgx2 | none | 2,040 MiB | none (−0.04 GiB noise) |
| dgx3 | 1920x1080 HDMI console (`/dev/fb0`) | 2,032 MiB | none (−0.03 GiB noise) |

The console framebuffer takes 8 MiB. The arm uses 2,032 MiB on every node, so
dgx3 keeps its console and dgx1 and dgx2 keep 8 MiB for one.

GPU access depends on how the buffer reaches CUDA (`reserve_bw.cu` style
probe on dgx2, 2,032 MiB, streaming uint4 kernels):

| Path | Read | Write | 64 KiB gather |
|---|---:|---:|---:|
| dma-buf export, `cuImportExternalMemory` | 235 GB/s | 204 GB/s | 240 GB/s |
| CPU mapping registered as I/O memory | 164 GB/s | 113 GB/s | 167 GB/s |
| `cudaMalloc` | 254 GB/s | 187 GB/s | 250 GB/s |

Patch 0021 uses the dma-buf import. A GPU fill and verify of all 266M words,
CPU spot reads, a CPU write read by the GPU, and `cudaMemset` all round-trip
correctly. Through torch, a 1 GiB copy into the imported backing runs at
235 GB/s against 236 GB/s for ordinary memory.

## Method

1. `make_arms.py` writes `cluster-carveout.json` from `config/cluster.json`:
   the DRM card device, the overlay mounts, `SPARK3_KV_DISPLAY_CARVEOUT=1`,
   and `--kv-cache-memory-bytes 2130706432`.
2. `overlay.sh` prepares vLLM with patches 0001-0021 (tree `2f8e61c6`) and
   copies the two changed runtime files to every node.
3. `run_arm.sh control control` benchmarks the running r5j service; then
   `run_arm.sh carveout carveout` starts the arm and runs the same protocol:
   quality gate, decode for prose, code and their answer-only cases at 1 and
   8 streams (3 samples), real-text prefill, and four concurrent 64K
   contexts.

**Acceptance:** the quality gate passes; decode, prefill and the 64K
admission rate are within noise of the control; MemAvailable under load rises
by about the old KV size on every node; the reported KV capacity rises by
about 42%.

## Results

Pending.
