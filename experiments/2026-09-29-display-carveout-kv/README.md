# Display carve-out

**Base:** r5j (`2026-09-28-karmic-kraken-r5j`), image
`vllm-ds41f-kkref:04c30fa98e79-r5j`.

**Goal:** use the memory the firmware reserves for a display, which every
node leaves idle, to hold more KV cache, while keeping a text console.

## Prior art

coolbho3k (emihuang) discovered that the headless GB10 display reserve is usable from CUDA and published it first, in [coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark](https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark) `878e0ee` (2026-09-17) and an [NVIDIA forum post](https://forums.developer.nvidia.com/t/383583/1). That approach registers the mapped DRM display buffer with `cuMemHostRegister(DEVICEMAP | IOMEMORY)` to back KV cache. jspark3 v1.8.0 later adopted that code (credited in its third-party notices), and jontaylor/gb10-ram-reclaim hands the region to Linux instead. This experiment measured both access paths and places read-once weights there rather than KV (see below).

## The carve-out

The firmware reserves 2.10 GiB at the top of physical memory on every node
(`1fe30cc000-20693fffff`, reserved in `/proc/iomem`). The GPU driver turns
the `DISPLAY_FRM` part of it into a scanout heap
(`memmgrCreateScanoutCarveoutHeap_GB10B`). Its size comes from the firmware;
no module or kernel parameter changes it (nvidia-drm has only `modeset` and
`fbdev`). Linux and ordinary CUDA allocations never use it.

nvidia-drm serves DRM dumb buffers from that heap. Measured on 2026-09-29 with
`nvidia_drm modeset=Y fbdev=Y`, no reboot and no host change:

| Node | Display | Largest dumb buffer | MemAvailable change |
|---|---|---:|---:|
| dgx1 | none | 2,040 MiB | none (−0.06 GiB noise) |
| dgx2 | none | 2,040 MiB | none (−0.04 GiB noise) |
| dgx3 | 1920x1080 HDMI console (`/dev/fb0`) | 2,032 MiB | none (−0.03 GiB noise) |

The console framebuffer takes 8 MiB.

## How the GPU sees it

A dumb buffer exported as a dma-buf and imported with `cuImportExternalMemory`
round-trips data correctly (GPU fill and verify of 266M words, CPU spot reads,
a CPU write read by the GPU, `cudaMemset`). Its access speed on dgx2
(`reserve_bw.cu`, 2,032 MiB):

| Access | dma-buf import | CPU mapping as I/O memory | `cudaMalloc` |
|---|---:|---:|---:|
| Sequential read | 235 GB/s | 164 GB/s | 235 GB/s |
| Sequential write | 207 GB/s | 113 GB/s | 196 GB/s |
| 64 KiB spans | 244 GB/s | 168 GB/s | 236 GB/s |
| Random 576-byte records | 7 GB/s | 10 GB/s | 221 GB/s |

BF16 matmuls with the weight in the carve-out (`gemm_place.py`), against the
same weight in ordinary memory:

| Shape | Carve-out / ordinary |
|---|---:|
| M 8649, K 1024, N 3072 (vision tower) | 2.31x |
| M 8649, K 1024, N 5632 | 2.23x |
| M 8649, K 2816, N 1024 | 2.36x |
| M 6, K 5120, N 16384 | 0.96x |
| M 48, K 5120, N 16384 | 1.10x |
| M 6, K 5120, N 43136 (output head) | 0.90x |

The mapping streams at full speed but gives no reuse: anything read more than
once per pass, or read at random, pays full memory latency every time.

## Arm 1: KV cache in the carve-out (rejected)

Patch 0021's first version took the whole KV backing, 2,032 MiB per rank,
from the carve-out. Against a same-day r5j control (same lean protocol, the
control on a service that had been up for 25 hours):

| | r5j control | KV in carve-out |
|---|---:|---:|
| KV capacity | 575,304 tokens | 815,471 tokens (+42%) |
| Quality gate | 5/5 | 5/5 |
| Lowest MemAvailable dgx1 / dgx2 / dgx3 | 5.58 / 7.54 / 7.31 GiB | 7.85 / 8.98 / 8.89 GiB |
| Single-stream step, prose / code | 42.8 / 46.2 ms | 42.5 / 48.2 ms |
| 8 streams, prose / code / answers | 164 / 187 / 175 / 232 tok/s | 150 / 172 / 161 / 222 tok/s |
| Real-text prefill 2K / 32K / 64K | 3.13k / 3.72k / 3.74k | 2.86k / 3.44k / 3.32k tok/s |
| Four 64K contexts, per stream | 13.8 tok/s | 12.7 tok/s |

Single-stream decode was unchanged, but everything that gathers or re-reads
more KV lost 4-11%, consistent with the random-access numbers above. Its first
start also failed: startup allocates two small temporary KV caches (0.2 MiB
for profiling, 7.4 MiB for B12X preparation), which each took a 16 MiB
carve-out buffer and left too little for the serving cache.

## Arm 2: embedding and output head in the carve-out

Patch 0021 now moves the two vocabulary-sized tables that are read once per
step, the embedding table (row lookups) and the output head (only multiplied
by decode rows and prefill's sampled positions), into the carve-out right
after loading, before kernel plans and graphs capture them. The DSpark drafter
aliases both, so they move together. That frees 842.5 MiB of ordinary memory
per rank (two 43,136 x 5,120 BF16 tables), and the `weights` arm grows the KV
cache by 0.8 GiB of it, to 2.2 GiB (about 900K tokens); `weights-256k` also
raises the context limit to 262,144 tokens.

## Method

1. `make_arms.py` writes `cluster-weights.json` and `cluster-weights-256k.json`
   from `config/cluster.json`: the DRM card device, the overlay mounts,
   `SPARK3_DISPLAY_CARVEOUT_WEIGHTS=1`, and the larger KV size.
2. `overlay.sh` applies patches 0001-0021 (tree `19270e20`) in a throwaway
   worktree and copies the two changed runtime files to every node.
3. `run_arm.sh ARM LABEL [bench options]` starts an arm under the launcher's
   memory guards and runs the same protocol as the control: quality gate,
   decode for prose, code and their answer-only cases at 1 and 8 streams
   (3 samples), real-text prefill, and four concurrent 64K contexts. The 256K
   arm adds longer prefill and admission sizes and `needle.py`.

**Acceptance:** the quality gate passes; decode, prefill and the 64K admission
rate are within noise of the control; MemAvailable under load is no lower than
the control's; the reported KV capacity rises by about 55%.

## Results

Same protocol as the arm 1 control (r5j), 2026-09-29. Every rank logged
`Display carve-out holds 842.5 MiB of streamed weights (embed_tokens, lm_head);
ordinary memory freed: 934-950 MiB`.

| | r5j control | `weights` (131K) | `weights-256k` |
|---|---:|---:|---:|
| Context limit | 131,072 | 131,072 | 262,144 |
| KV capacity (vLLM) | 575,304 tokens, 4.39x | 904,074 tokens, 6.90x | 1,348,708 tokens, 5.14x |
| Quality gate | 5/5 | 5/5 | 5/5 |
| Single-stream step, prose / code (ms) | 42.8 / 46.2 | 43.6 / 47.4 | 41.9 / 47.6 |
| Single-stream step, answers prose / code (ms) | 44.3 / 49.4 | 44.8 / 49.7 | 44.5 / 49.8 |
| 8 streams prose / code / answers (tok/s) | 164 / 187 / 175 / 232 | 164 / 183 / 175 / 243 | 166 / 184 / 172 / 236 |
| Real-text prefill 2-4K / 32-64K (tok/s) | 3.13k / 3.72-3.74k | 3.13k / 3.73-3.76k | 3.88k / 3.77k |
| Real-text prefill 131K / 200K (tok/s) | | | 3.58k / 3.45k |
| Four 64K contexts, per stream | 13.8 tok/s, KV 32% | 13.4 tok/s, KV 20% | |
| Four 180K contexts, per stream | | | 11.7 tok/s, KV 35% |
| Lowest MemAvailable dgx1 / dgx2 / dgx3 (GiB) | 5.58 / 7.54 / 7.31 | 6.32 / 7.46 / 7.51 | 5.89 / 7.42 / 7.33 |

`needle.py` on `weights-256k` at 177,654 prompt tokens retrieved the phrase at
depths 0.1, 0.5 and 0.9 (3/3; 52, 47 and 25 s).

The `weights` arm meets every acceptance criterion: decode, prefill and
admission are within noise of the control, the quality gate passes, and the
lowest MemAvailable is no lower. The `weights-256k` arm also serves 256K
prompts, with KV for five full windows and four 180K contexts at 35% of the
cache; the longer limit costs dgx1 about 0.4 GiB against `weights`, still
above the control. Candidate for promotion as r5k: patch 0021, the DRM card
device, `SPARK3_DISPLAY_CARVEOUT_WEIGHTS=1`, 2.2 GiB of KV and a 256K limit.
