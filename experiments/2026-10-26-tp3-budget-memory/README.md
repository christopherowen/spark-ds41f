# What limits TP3's long-prefill budget, and what would raise it

Owner, 2026-10-25: why can the ring's long-prefill budget (`max_num_batched_tokens`) go so
much higher than the triangle's, and should the implementation improve?

Base: candidate 6 (vLLM `tilelang-ports` 26382f5cf, image `spark-ds41f:19f2c20ed4d6-next-6`).
- **Sources:**
  - existing runs: the 10-19 TP3 sweep, the 10-23 TP4 budget runs, the 10-23 memory window's
    allocator traces and samplers, the 2026-10-25 TP3 fit check (`candidate-6-prefill-tp3`) and
    request-inventory-5's 15,395-token prefill trace;
  - two read-only code audits of the tree;
  - one single-node kernel check on dgx4 ([bundles/moe-rows](bundles/moe-rows)).
- **Not done:** no cluster time, nothing changed in serving, and no vLLM branch yet. Every lever
  below is bounded, none is built.

## Answer

- **Headroom.** The ring has about three times the triangle's. At its budget, dgx1 keeps
  18.7-20.1 GiB on the ring (16,384) and 6.67 GiB on the triangle (8,192).
  - A TP3 rank holds 94.1 GiB of weights against 73.0 GiB on TP4.
  - The smaller KV pool (4.9 GiB against 10.5) gives back only a quarter of those 21 GiB.
- **Cost per budget token.** It is about the same on both shapes: 0.33-0.48 GiB a rank per
  1,024 tokens on the ring, 0.36-0.41 GiB on the triangle.
  - Sequence-parallel prefill splits only the residual and mHC work.
  - Attention, the indexer, the compressors and the MoE run on all T gathered rows on every rank,
    and their buffers are sized by T.
- **Why 3.3 GiB against 0.7.** The ring's 8,192 → 16,384 step added twice the tokens of the
  triangle's 4,096 → 8,192.
  - It cost 3.4-3.9 GiB on every ring node.
  - The triangle step looked cheap on dgx1 (−0.5 GiB) because:
    - single-run lows on the triangle move by up to 0.55 GiB between identical arms, and dgx2
      lost 2.0 GiB in the same arms;
    - below about 6.7K tokens the MoE's workspace sits under a fixed 0.88 GB floor.
- **12,288 on the triangle.** Expect about 1.4 GiB less at the lows, for +3.3-5.3% prefill:
  - dgx1 6.67 → about 5.2 GiB during 1M prefills;
  - the start-up low falls by the same amount; it is not recorded today, and must be above
    about 6.4 GiB now for 12,288 to clear the 5 GiB guard (the sweep below measures it);
  - the median decode step beside a long prompt 1.44 → about 2.0 s.
- **16,384 on the triangle.** It needs about 2.9 GiB for +5.0-7.2% prefill and a median step
  of about 2.6 s. It does not fit today.
- **The implementation can improve on both axes.**
  - **Memory:** three low-risk cuts free about 1.0 GiB a rank at 8,192 and 1.3 GiB at 12,288,
    about what 12,288 needs. The display carve-out has another 1.27 GiB free on every triangle
    node.
  - **Speed:**
    - Each chunk pays a fixed cost of about 196 ms (TP3) and 155 ms (TP4). About 70% of it is
      the routed MoE's size-independent time: about 70% of one read of every expert in the 20
      full-row layers, which compute does not hide.
    - A 64-row MoE tile for prefill is bit-identical on dgx4 and makes the routed MoE 11-19%
      faster. That is about +3.5-5% prefill on both shapes, with no stall trade and no guard
      change.

## 1. Headroom per shape

| | TP4 ring at 16,384 | TP3 triangle at 8,192 |
| --- | --- | --- |
| weights a rank (`Model loading took`) | 72.79 GiB | 94.12 GiB |
| KV pool | 10.5 GiB | 4.91 GiB |
| display carve-out | unused (the head lost its L2 reuse there, 2026-10-12) | 737.2 MiB of about 2,040 MiB: embedding + packed head |
| graph capture + tail-shape warm-up | reserve +2.84 GiB, MemAvailable −3.3 GiB | "took 1.93 GiB" (engine log) |
| lowest MemAvailable dgx1 / others (budget suite) | 18.66 / 19.93-20.15 GiB (candidate 1) | 6.67 / 7.53, 7.94 GiB (candidate 6) |
| guards | 5 GiB start-up, 3 GiB running, 4 GiB bench floor | same |

On the triangle, dgx1 is the floor by 0.86 GiB (dgx2) and 1.27 GiB (dgx3) at the fit check.
Candidate 6 at 8,192 sits 1.4-2.1 GiB above r6x at the same budget on all three nodes
(10-19: 4.57 / 5.49 / 6.50 GiB). Existing data cannot attribute this.
- glibc pinning's measured effect was dgx1's API server (the 10-23 soak), not the other nodes.
- B12X's removal from serving is the likely source, unverified.
The sweep below measures candidate 6's own per-token cost.

## 2. What grows with the budget

### Measured, per rank (budget suite: 32K-500K prefill, the 131K stall, cold 1M prefills)

| Step | dgx1 | dgx2 | dgx3 | dgx4 | GiB per 1,024 tokens |
| --- | ---: | ---: | ---: | ---: | --- |
| TP4 candidate 1, 8,192 → 16,384 (`next-budget-tp4`, two 8,192 arms) | −3.40 | −3.38 | −3.67 | −3.86 | 0.42-0.48 |
| TP4 candidate 1, 16,384 → 24,576 (`next-budget24-tp4`) | −2.60 | −2.71 | −2.24 | −2.62 | 0.28-0.34 |
| TP3 r6x + skip, 4,096 → 8,192 (`prefill-budget-tp3`, two 4,096 arms) | −0.54 | −2.02 | −0.93 | | 0.13-0.50 |
| TP3 r6x, 4,096 → 8,192 (experiments/2026-10-19-r6x-tp3-1m: serving / 1M low) | −1.63 / −1.44 | | | | 0.36-0.41 |

The two 4,096 arms of the TP3 sweep differ by 0.30 / 0.55 / 0.37 GiB on dgx1 / dgx2 / dgx3, so
single-run TP3 lows resolve no better than about half a GiB.

### The ring's boot, traced (dgx2, 16,384, 2026-10-23 18:33 CET)

From [memory/analyze_sweep.py](memory/analyze_sweep.py) on the memory window's allocator trace
and sampler:

| after | torch allocated / reserved | MemAvailable | driver (GPU) | anon |
| --- | --- | ---: | ---: | ---: |
| weights | 72.99 / 73.27 GiB | 39.30 | 78.24 | 3.36 |
| kernel preparation, the profile run and its release | 77.41 / 77.72 | 33.89 | 82.40 | 4.29 |
| KV pool | 87.94 / 88.14 | 23.35 | 92.86 | 4.29 |
| graph capture and tail-shape warm-up | 88.00 / 90.98 | 20.03 | 95.98 | 4.71 |
| serving: lowest; torch peak | allocated 89.59, reserved 91.04 | 19.95 | | |

What each phase means:
- **Released memory returns.** The profile run's peak (reserved 79.8 GiB) went back to the OS at
  its `empty_cache` (driver −2.0 GiB).
- **The budget-sized forward that stays reserved is the tail-shape warm-up.**
  - Inside `capture_model`, after capture's own `empty_cache`, adaptive verification runs eager
    dummy forwards that double up to the budget
    (`preparation_tail_sizes` / `warm_tail_shapes`, `vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:96-108, 189-196`,
    called at `gpu/model_runner.py:1430`).
  - Their activation peak stays cached: reserved +2.84 GiB with allocated flat.
  - Serving's long prefills then reuse it: reserved +0.06 GiB more over 75 minutes of prompts up
    to 966K tokens.
- **The GPU's memory outside torch is about 1 GiB a rank:** nvidia-smi 92.07 GiB against
  91.04 GiB reserved.

### From the code (per budget token, per rank, TP3 / TP4)

Two audits (model directory; runner, workspace and collectives), spot-checked; file:line at
26382f5cf.

| Allocation | Bytes a token | Kind |
| --- | --- | --- |
| MoE plan in workspace lane 0 (`tilelang/moe.py:1257-1269`): per-pick down outputs 6 × 5,120 BF16 (60 KiB), gate/up output (18 / 13.5 KiB), MXFP8 inputs (about 10 KiB), fold output (10 KiB) | about 100 / 95 KiB, above a fixed 0.88 GB floor (layer 20's indexer chunk scratch, sized by max_model_len) that it exceeds from about 6.7K / 7.2K tokens | persistent; grows in the profile run |
| layer 20's `_candidates` (budget × 16,384 int32) + lengths (`attention.py:721`) | 64 KiB | persistent; written only for rows that reach the CED decoder, except for requests that keep every row (prompt logprobs) |
| compressor scratch, layers 2, 8, 14, 20 (`compressor.py:207-215`) | 17 KiB | persistent |
| DSpark speculator context, encoder `inputs_embeds` (budget × 5,120 BF16 each) | 10 + 10 KiB | persistent |
| Engram staged rows, two tables (`common/engram.py:900`) | 8 / 6 KiB | persistent |
| Engram pinned host buffers (`engram_rows/rows.py:129-131`), every column although a rank writes only its 24/N (`_row_plan.h:257`) | 12.75 KiB, of which 8.5 / 9.6 KiB never written | pinned host |
| `candidate_block_buffer` (budget × 2,048 int32, `nvidia/model.py:456`): the attention takes the argument and never stores it | 8 KiB | persistent, unread |
| DSpark's top-k buffer (budget × 512 int32, `nvidia/dspark.py:405`): the drafter's layers are window-only | 2 KiB | persistent, unread |
| shared top-k, metadata builders, CED and mHC odds | about 3 KiB | persistent |
| **persistent total** | **about 230 / 220 KiB** | |
| activation peak a budget-sized forward leaves cached (aux hidden states, residual carries, MoE partials, gathered rows, q and attention outputs, WO send buffers) | up to 155 / 131 KiB (ring at 16,384: +2.84 GiB reserve at capture = 182 KiB with graph pools) | held after the warm-up |
| **total** | **about 385 / 350 KiB = 0.37 / 0.34 GiB per 1,024 tokens** | matches the measured rates |

**Sized for the full chunk where SP needs a share?**
- Only the residual side runs on local rows, and its buffers are local (the mHC constant mixes,
  32 B a token, are the exception).
- Everything large runs on the gathered rows by design.
- The waste is elsewhere:
  - **CED rows:** budget-sized buffers that only the CED decoder's compacted rows use (layer 20's
    candidates: at most max_num_seqs × 128 rows unless a request keeps every row);
  - two unread buffers;
  - Engram's unwritten columns;
  - **the drafter's workspace lane (a fixed cost):**
    - `WorkspaceManager.reserve_all` (`vllm/v1/worker/workspace.py:224-251`) sizes every slot of
      both lanes to the largest reservation.
    - Every target attention layer reserves its indexer scratch through it
      (`tilelang/attention.py:205-214`).
    - So the DSpark lane holds at least 0.88 GB although its window-only layers borrow about
      0.27 GB at 8,192.
    - The sweep's slot log confirms the size.

## 3. The rank-0 tax

The ring at 16,384 with glibc pinned (soak, 2026-10-23 18:33-19:50 CET; process memory from
[process_memory.py](process_memory.py)):

| Process | dgx1 anon PSS (start → after 75 min) | other nodes |
| --- | --- | --- |
| API server | 1.06 → 1.43 GiB | none |
| engine core | 0.71 → 0.97 GiB | none |
| headless launcher (`vllm serve --headless`, idle) | none | 0.98 GiB, flat |
| worker | 3.30-3.34 GiB | 3.30-3.34 GiB |

The head node's extra:
- 0.79 GiB at the start, 1.42 GiB after 75 minutes of million-token agent traffic;
- on the triangle at the fit check, dgx1 is 0.86 / 1.27 GiB below dgx2 / dgx3.
- humming's nvrtc and ops probes spawn at start-up, but each lives a second or two, before the
  weights load.

What could move:
- **The API server, off the triangle** (vLLM's API-only frontend against a headless engine).
  It pays only until dgx2 becomes the floor: +0.86 GiB at the fit check.
- **The engine core** stays with the mp executor.
- **The display carve-out:**
  - The triangle uses 737.2 MiB of about 2,040 MiB (`docs/current-state.md`,
    experiments/2026-09-29-display-carveout-kv): 1.27 GiB is idle on every node.
  - Only data read once a step belongs there: uncached, small pages, no L2 reuse.
  - The best candidate is the DSpark drafter's routed experts:
    - 3 layers × 128 experts × 2,304, about 0.80 GB a layer a TP3 rank;
    - only draft rows ever reach them (the prefill forward launches no drafter MoE).
  - Their coded scales are read 16 bytes at a time, so they need a decode A/B before anything else.

## 4. What every chunk pays regardless of its size

**TTFT fits** ([fit_chunks.py](fit_chunks.py)): TTFT = K_N + chunks × b, per prompt size, across
a run's budgets.

| Run | b, ms a chunk (32K / 131K / 262-524K / 1M) | worst residual |
| --- | --- | --- |
| TP3 r6x + skip, 4,096 / 6,144 / 8,192 | 192 / 198 / 196-200 / 203 | 0.06-0.45 s |
| TP4 candidate 1, 8,192 / 16,384 | 151 / 157 / 158 / 168 | 0.01-0.36 s |
| TP4 candidate 1, 16,384 / 24,576 | none (two chunks either way) / 139 / 162 / 154 | 0.04-0.37 s |

The fixed cost does not depend on context. It is 13% of a TP3 chunk at 8,192 (1.48 s at 32K)
and 6-7% of a TP4 chunk at 16,384 (2.43 s).

**One forward** ([prefill_trace.py](prefill_trace.py) on request-inventory-5's 15,395-token
prompt, TP4 rank 0): 2,140 ms, GPU busy 2,134 ms (no host gaps).
- **Layers 0-19 run on all rows:** 91-101 ms each; layer 0 is 174 ms with the embedding
  exchange, and the Engram layers are longer.
- **Layers 20-39 run on the CED decoder's 128 compacted rows:** 3.6-5.9 ms each, 87 ms in all,
  52 ms of it routed MoE.
  - A prompt's final chunk pays these.
  - Nonfinal chunks keep one decoder row (the skip): a few milliseconds of launches a layer.

**One layer's routed experts against the chunk size** (dgx4, the release kernels, uniform
routing, [bundles/moe-rows](bundles/moe-rows), [moe-rows.json](bundles/moe-rows/moe-rows.json);
2026-10-25 20:16 CET):

| | TP3 (768 columns) | TP4 (576) |
| --- | --- | --- |
| all 384 experts once (512 tokens) | 9.71 ms (241 GB/s) | 7.52 ms (234 GB/s) |
| fit over 2,048-24,576 tokens, 32-row tiles | 6.93 ms + 2.06 µs a token | 5.39 ms + 1.62 µs a token |
| the fixed part × 20 full-row layers | **139 ms a chunk (71% of b)** | **108 ms (70% of b)** |
| 64-row tiles | 7.69 ms + 1.58 µs a token | 5.80 ms + 1.32 µs a token |
| 64 against 32 rows at 4,096 / 8,192 / 12,288 / 16,384 tokens | −7.3 / −11.0 / −16.2 / −19.1% | −5.6 / −11.4 / −12.4 / −14.8% |
| outputs, 64 against 32 rows | bit-identical at every size | bit-identical |

So about 70% of one full read of each full-row layer's experts is not hidden under compute.
That one cost is most of what a chunk pays regardless of its size.

The remaining 55-65 ms a chunk is not attributed. The candidates:
- the latency of about 170 NCCL exchanges and gathers;
- the decoder's one-row pass;
- small kernels and partial waves.

The first chunk's Engram wait (35-40 ms) and the final chunk's 128-row decoder pass are per
prompt (in K), not per chunk. Per-token communication is the fusion survey's subject
(experiments/2026-10-25-fusion-survey, #583, #593).

The kernel lab marks this bundle FAIL only because its `exit` verdict expects pytest output.
The step exited 0 in 43 s.

## 5. Ceilings: 12,288 and 16,384 on the triangle

The forecast takes candidate 6's fit-check TTFTs at 8,192 and moves the chunk count with
b = 196 ms. It assumes per-token cost does not depend on the chunk size: the TP4 fits held
within 0.5% at a third budget.

| Prompt | 8,192 (measured) | 12,288 | 16,384 |
| --- | ---: | ---: | ---: |
| 32,761 tokens | 5,542 tok/s | 5,733 (+3.4%) | 5,936 (+7.1%) |
| 131,900 | 5,618 | 5,914 (+5.3%) | 6,020 (+7.2%) |
| 262,554 | 5,343 | 5,589 (+4.6%) | 5,708 (+6.8%) |
| 500,373 | 4,828 | 5,028 (+4.1%) | 5,129 (+6.2%) |
| 999,979 | 3,949 | 4,078 (+3.3%) | 4,145 (+5.0%) |
| median decode step beside a 131K prompt | 1.44 s | about 2.0 s (+41%) | about 2.6 s (+81%) |
| memory at the lows, at 0.35-0.40 GiB per 1,024 tokens | 6.67 GiB (dgx1) | about 5.1-5.3 | about 3.5-3.9: below the bench floor |

On the ring's curve at equal rows per rank, 12,288 on the triangle is the ring's 16,384
(4,096 rows a rank). The ring's 2,048 → 4,096 rows gave +5.4-6.3%, the same order as the
forecast. Rows per rank is the wrong axis, though: the fixed cost follows the experts' bytes per
rank and the number of chunks, not SP's row split.

The 64-row MoE tile adds about +3.6% at 8,192 and +5% at 12,288 (−51 / −103 ms a chunk) on top
of these figures.

## Ranked levers

Gains are per triangle rank unless marked. Nothing is built; each lever is one variable for
its own branch.

| # | Lever | Gain | Evidence | Effort | Risk |
| --- | --- | --- | --- | --- | --- |
| 1 | 64-row tiles for the routed MoE in prefill forwards (2,048 rows or more; decode keeps 32) | MoE −11% (8,192) / −16% (12,288) / −19% (16,384); prefill about +3.6% at 8,192 and +5% at 12,288 on TP3, about +4% at 16,384 on TP4; no decode change; costs about 0.16 GiB fixed (12,288 more layout rows) | dgx4 sweep, bit-identical outputs at every size | S-M: a second plan by row count, both declared at start-up | low |
| 2 | Layer 20's candidates as 2,048 block ids a row instead of 16,384 positions | 0.44 / 0.66 / 0.88 GiB at 8,192 / 12,288 / 16,384 | code (`attention.py:721`; the four reindex layers read positions) | M: the producer's top-k writes ids, the consumers expand ×8 on load | low: same candidates in the same order; check outputs bit for bit |
| 3 | The drafter's workspace lane without the target's indexer scratch (reserve by lane) | about 0.5 GiB fixed, every rank, both shapes | code (`workspace.py:224-251`, `tilelang/attention.py:205-214`); the size comes from the sweep's slot log | S | low: the locked workspace refuses an undersized lane at start-up |
| 4 | Drop the two unread buffers | 0.08 / 0.12 / 0.16 GiB | code | S | none |
| 5 | DSpark's routed experts in the display carve-out (about 1.5 of 3 layers fit) | up to 1.27 GiB on every triangle node | experiments/2026-09-29-display-carveout-kv (read-once data streams at full rate, re-read data loses L2); drafter experts see only draft rows | M-L | medium: uncached memory, 16-byte scale reads; needs a decode kernel A/B first |
| 6 | Hide the routed experts' weight stream under compute (prefetch the next experts' slabs, persistent CTAs) | ceiling 139 ms a chunk on TP3 (about 9% at 8,192), 108 ms on TP4 (about 4.5% at 16,384) | dgx4 sweep: the fixed part is 71% of one full weight read | L | medium: new kernel structure; same arithmetic per row |
| 7 | API server off the triangle | up to 0.86 GiB on dgx1, until dgx2 is the floor | soak process table | L: deployment topology | medium |
| 8 | SwiGLU in the gate/up GEMM's epilogue (no BF16 gate/up buffer) | 0.14 / 0.21 / 0.28 GiB, plus about 1% prefill | code; fusion survey row 12 | M-L | medium: the BF16 rounding point must be kept |
| 9 | Engram's pinned buffers for the rank's own columns | 0.07 / 0.10 / 0.13 GiB of pinned host memory | code | S-M | low |
| 10 | One compressor scratch set for the four compressor layers | about 0.09 / 0.13 / 0.18 GiB | code | M | low |

**Not levers:**
- **The MoE's per-pick down outputs** (60 KiB a token): splitting the routed MoE into row
  slices pays the fixed 6.9 ms a layer again for every extra slice.
- **The held activation peak:** every long prefill needs it, and releasing it between prefills
  does not raise the minimum the guard sees.
- **The guards:** never loosened.

**Together:**
- Levers 2-4 free about 1.0 GiB at 8,192 and 1.3 GiB at 12,288: enough for 12,288 at the
  lows the fit check measured.
- With lever 5 as well, 16,384 (about 2.9 GiB) comes within reach.
- Levers 1 and 6 take part of what a larger budget buys without its decode stall.

## Measurements

### Queued: nothing

The cluster is busy with candidate 6's TP3 qualification and release comparison. Queue these
afterwards, with the coordinator's go:

1. **[tp3-budget-memory.json](tp3-budget-memory.json): the TP3 budget sweep with the allocator
   trace.**
   - Arms: candidate 6 at 8,192, 12,288, 6,144, then 8,192 again, on the fit check's suite
     (`experiments/2026-10-19-prefill-budget-tp3/suite.json`). About 2 hours.
   - [make_arms.py](make_arms.py) generates the arms from `rc6/rc-tp3.json`. They differ only in
     the budget and threshold, a diagnostic env, and the overlay mount:
     - `budget-6144-tp3.json` resolves byte-identical to `rc6/rc-tp3-6144.json`'s serve_args;
     - the overlay is [memory/overlays](memory/overlays): the release's `gpu_worker.py` plus an
       allocator sampler thread, phase marks and per-pool summaries; serving behaviour is
       unchanged;
     - the env adds `VLLM_DEBUG_WORKSPACE=1`, which logs only slot resizes.
   - **Order:**
     - 12,288 runs second: if it trips the 5 GiB start-up guard, the boot fails closed and the
       other arms still run.
     - The repeat bounds the session's drift.
   - **Before queueing:**
     1. Merge this PR: the lab syncs the node checkouts to origin/main.
     2. Copy [memory/sampler.py](memory/sampler.py) to `~/spark-lab/tp3-budget-memory/` on dgx1-3.
     3. Start [memory/watch.sh](memory/watch.sh) on dgx1. It arms the samplers and an engine-log
        follower when the job starts, and collects the traces after the restore.
     4. Pass `--production-config experiments/2026-10-23-next-release/1022/cluster.json`
        (production on the triangle is 1022).
   - **Analysis:** [memory/analyze_sweep.py](memory/analyze_sweep.py) on the watcher's output. It
     gives, per arm and node:
     - the phase table above;
     - the start-up low against the guard;
     - the serving low;
     - the workspace slots;
     - the pools;
     - the change per 1,024 tokens per phase.
   - **What it decides:**
     - candidate 6's real cost of a budget token on the triangle;
     - whether 12,288 clears the start-up guard;
     - lever 3's size.
2. **Single-node, dgx4:**
   - **Done:** [bundles/moe-rows](bundles/moe-rows) (the MoE sweep above).
   - **Next, not written:**
     - lever 1's tests and gated benchmark;
     - lever 5's decode A/B: drafter-shaped grouped GEMMs with weights in a dma-buf carve-out
       buffer against ordinary memory.

## Files

- [README.md](README.md): this record.
- [fit_chunks.py](fit_chunks.py): the per-chunk fixed cost from a budget run's TTFTs.
- [prefill_trace.py](prefill_trace.py): one prefill forward split into full-row and compact
  layers.
- [process_memory.py](process_memory.py): a node's memory by component and process across an arm.
- [make_arms.py](make_arms.py): writes `budget-{6144,8192,12288}-tp3.json`.
- [tp3-budget-memory.json](tp3-budget-memory.json): the sweep spec.
- [memory/](memory): the trace overlay, sampler (unchanged from 2026-10-23), watcher and
  analysis.
- [bundles/moe-rows](bundles/moe-rows): the dgx4 kernel bundle and its record.
