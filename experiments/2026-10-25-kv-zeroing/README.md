# KV block zeroing: DS4.1 drops it, as DeepSeek's reference does

2026-10-25, CET. For the release after spark-ds41f-1025; nothing here touches candidate 6.
The single-node GPU checks ran on dgx4 (below: the tests, the indexer's initcheck and its
benchmark gate). Serving has not run: the lab specs are queued for the full cluster. The
gain bound comes from existing traces and the code.

## Question, direction and answer

The owner asked whether removing vLLM's KV block zeroing (`zero_kv_blocks`, 0.20% of
prefill in the kernel audit, spark-ds41f #585) could let old bad memory hurt us. The
direction that followed: work toward DeepSeek's reference (REF, `inference/model.py` in
the checkpoint snapshot).

REF zeroes each cache once, when it is built, and never between sequences. It is correct
because no kernel reads a slot the sequence has not written: reads are bounded, or
skipped by a -1 index.

- **One DS4.1 reader diverged from REF: DSpark's draft windows.** On two kinds of nonfinal
  prefill chunk they read slots that nothing had written in the block's current
  ownership:
  - CED's one-row chunks, which every prompt longer than one chunk plus 143 rows gets;
  - a first chunk shorter than 128 rows after a prefix hit.

  Zeroing made those slots zero records. Without it they hold another group's bytes, and
  as E4M3 or UE8M0 those can decode to NaN. The drafter's MoE router then refuses the row
  with a device assert (`tilelang/router.py:460`), which aborts the engine. The drafts
  were discarded (`scheduler.py:3705`), so outputs and acceptance never depended on them.
  **0501** bounds those windows at CED's replay start, as REF bounds its windows.
- **The indexer read slightly wider than REF.** It stored no score past the visible
  positions, but its last 64-row tile loaded the rows past them, which REF never reads,
  then discarded their results. **0503** predicates those rows' key and scale loads off,
  so no key or scale is read past the visible positions. Outputs are bit-identical, and on
  dgx4 the bounded kernel is faster than the release's, not slower (below).
- **Every other reader already matches REF**:
  - the sparse MLA skips -1 entries without loading them;
  - the compressor pools only slots the request wrote;
  - the window and compressed reads stop at the written positions.

  The table below has the details. No kernel masks an unwritten slot arithmetically, and
  after 0501 and 0503 no difference from REF remains.
- **0502 removes per-allocation zeroing for DS4.1**, as its own patch. Every DS4.1 cache
  declares `AttentionSpec.uses_only_written_slots`: no value of an unwritten slot reaches
  a result. `needs_kv_cache_zeroing` skips mixed-precision caches whose layers all
  declare it. For DS4.1 the scheduler then records no block ids, the runner builds no
  zeroer, and no `zero_kv_blocks` kernel is declared or launched. The pool is still
  allocated zeroed, once, like REF's `torch.zeros` buffers.
- **0500, `VLLM_KV_CACHE_POISON`**, stays as the debug lab check that proves the bounded
  reads, now and for every future kernel. It fills every new block with NaN bytes,
  compressor rings included.
- **Gain:** at most 0.20% of prefill, about 0 in decode. It was never the reason; REF's
  semantics are.

## REF and DS4.1, reader by reader

REF line numbers are of the snapshot's `inference/model.py` (dba1be0a). DS4.1 paths are
under `vllm/models/deepseek_v4_1/` in candidate 6's vLLM 3bdf9558c unless noted.

| Reader | REF | DS4.1 serving (TileLang) | Match |
| --- | --- | --- | --- |
| Cache buffers | `register_buffer(torch.zeros(...))` once: index `k_cache` L520-523, `window_kv_cache` L663-666, `compress_kv_cache` L670+; never zeroed between sequences | one pool allocated with `torch.zeros` (`vllm/v1/worker/utils.py:445`); before 0502, also every new block zeroed (`KVBlockZeroer`) | **after 0502** |
| Window, target layers 0-19 | `get_window_topk_idxs` (L409-425): -1 for slots before the sequence started or while the ring fills; `sparse_attn` skips -1 | `_window_macro` (`tilelang/metadata.py:142`) starts at `max(pos + 1 - 128, 0)`; holes and the null block become -1 (:181); prefix hits serve cached window blocks | yes: bounded, -1 skipped |
| Window, CED decoder layers 20-39 | REF computes every row | keeps a chunk's last 128 rows (or one) and starts the window at `replay_start` (:162; `tilelang/ced.py:148`), the first row it computed | yes: reads only rows it wrote |
| Window, DSpark drafter layers 40-42 | REF's draft layers read the same window ring by -1 indices | window `[prefix - 128, prefix + 6)` with no replay bound (:154): reads unwritten slots in nonfinal chunks (below) | **no; 0501 bounds it at `replay_start`** |
| Compressed latents (top-k) | `Indexer.forward` returns -1 for `idxs >= compress_lens` (L577-580); `sparse_attn` skips -1 | top-k pads -1; `gather_idx` selects -1 past `IdxLengths` (`tilelang/sparse_mla.py:220`); a -1 entry's cp.async copies 0 bytes and zero-fills shared memory (:260), with no global read | yes: -1 skipped |
| Window records (sparse MLA) | as above | `gather_swa` selects -1 past the length (:211), staged the same way | yes |
| Index keys, scores | reads `index_k[:bsz, : end_pos // ratio]` (L554); prefill masks positions `>= compress_lens` to -inf (L563-567) | extent `min(visible, Active, width)`, with `visible = (pos + 1) // ratio` (`tilelang/metadata.py:125-137`); scores are stored only for positions below it (`tilelang/indexer.py:202`). Before 0503 the last 64-row tile loaded its rows past the extent (:164) and discarded their results; since 0503 their loads are predicated off | **after 0503**: no key or scale read past the visible positions |
| Index candidates (two-level top-k) | `select_candidate_blocks` (L583-610) on scores already at -inf where unreachable; keeps `top.values > -inf` | candidate scores select `-inf` for -1 candidates (`tilelang/indexer.py:270`, :203); top-k pads -1 | yes |
| Compressor state | `kv_state` zeros and `score_state` -inf at construction; prefill writes the trailing partial group into the state, decode fills slots and pools only a completed group (L458-490) | ring slot per row; the gather reads the previous slot only at an odd first position, under a select (`tilelang/compress.py:79-85`); every ring ownership starts at an even position, so that slot was written by the request's previous step | yes: bounded |
| Writers (window and compressed records, index keys, ring) | slice assignments of what was computed | whole records, pad slots (-1) skipped (`tilelang/cache_writer.py:124`, `tilelang/indexer.py:341`, `tilelang/compress.py:234`) | yes |

With 0501 and 0503 every DS4.1 reader reads only what REF reads. 0502's declaration
states the weaker property that the removal needs: no unwritten slot's value reaches a
result. 0503 makes the stronger one hold for the indexer too: no unwritten slot is read.

### Why a declaration, and not "the FP32 ring doesn't count"

The ring's blocks come from the same pool as every other group. A block a ring frees can
come back as a window or compressed page, and the reverse. The reinterpretation that
vLLM's mixed-precision rule guards against does happen.

What makes it harmless for DS4.1 is that its readers never use an unwritten slot. That is
a property of the model's kernels, so each cache spec states it:
- `AttentionSpec.uses_only_written_slots`, carried through every spec merge;
- the rule zeroes a mixed-precision cache only when some layer does not declare it.

Excluding FP32 rings from the rule would drop zeroing for any model with a ring, whatever
its readers do. Precision-stable pools (freed blocks returned only to groups of the same
layout) are not needed either: with bounded reads, cross-precision reuse is harmless, and
the pool stays one shared pool.

## The gain

From request inventory 5's TP4 traces (rank 0): [zero_share.py](zero_share.py), recorded
in [runs/zero-share.txt](runs/zero-share.txt). It cuts steps the way the kernel audit's
`kernel_shares.py` does.

| Step | Zeroing launches | Zeroing time | Share of busy time |
| --- | ---: | ---: | ---: |
| The 15,395-token prefill | 1 | 4,255 µs | **0.199%** |
| Short prompts' prefill (12 scenarios) | 1 each | 13-21 µs | 0.01-0.03% |
| mixed-c8's prefill / its mixed steps | 1 / 1 | 142 / 14 µs | 0.075% / 0.007% |
| Decode: greedy (11 steps), 11 c1 types (7-29 steps), mixed-c8 (32 steps) | 0 | 0 | **0%** |

- **Prefill:** a 16K chunk allocates about 1,984 blocks (452 MB) and zeroes them in
  4.3 ms. That chunk takes 2.7 s at 6,000 tok/s (spark-ds41f-1022's prefill), so the share
  is 0.16%, and lower at depth.
- **Decode:** a 15-block launch comes about every 55 steps per request, roughly 0.002% at
  one stream and 0.02% at 16.
- **Below resolution:** the release suite resolves prefill to about ±0.5%.

## The drafter finding (0501)

DSpark's context KV is written only for the CED decoder's kept rows. CEDState's indices
pack the drafter's context (`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:619`).
The drafter's three window caches are not prefix-cached (`attention.py:294-302`).

The drafter drafts for every request in the batch, prefill chunks included
(`speculator.py:577`), over `[max(prefix - 128, 0), prefix + 6)`:

1. **CED one-row chunks** (`ced.py:105`, `limits[i] = 1` at :140). A nonfinal aligned
   chunk keeps only its last row for the decoder and the drafter. The window's other 127
   positions were never written.
2. **A short first chunk after a prefix hit at L.** Aligned chunking makes the first chunk
   `T - L % T` rows. On the ring (T = 16,288) at L = 32,512 that is 64 rows; on 1022's
   triangle (T = 4,048) at L = 12,032 it is 112. The window reaches into `[L - 128, L)`,
   a fresh block, because the drafter's groups were not part of the hit.

CED's `replay_start` (`tilelang/ced.py:148`) is `max(prefix_start, the first kept row)`.
That is exactly where a request's draft context begins.

0501 makes the window kernel's draft branch, and its Triton reference, start no earlier
than that. The model state binds its persistent buffer to the drafter's attention layers
(`bind_draft_replay_start`), so captured draft graphs read each step's values, and the
drafter declares only the replay variant.

No decode or final-chunk window changes:
- after a hit, the final chunk keeps at least 128 rows (`kv_cache_manager.py:394`);
- without a hit, `prefix_start` is 0;
- a full chunk's first kept row is its window start already.

## The indexer's read bound (0503)

The score kernel (`tilelang/indexer.py`, `indexer_score_v41`) loads one 64-row tile of
keys and UE8M0 scale words at a time.
- **Before:** the last tile of a row whose extent ends inside it loaded every row of the
  tile. The keys past the extent are slots nobody had written: the rest of the last
  visible position's page. The MMA computed their scores, and the predicated store
  dropped them.
- **Now:** in a dense row, the key and scale loads of rows past the extent are predicated
  off, and those rows score zeros: `T.if_then_else(p0 + p < extent, KPool[...], 0)`, and
  the same for the scale word.
  - Only the loads are predicated. The page read stays unconditional, so a dense tile's
    page read stays uniform and its addresses affine, as in the release.
  - The extent joins the bounds check TileLang already puts on each load (`page` within
    the pool): the generated CUDA is the release's with one more term in that predicate.
  - **Candidate mode is unchanged**: it already read only written slots. There the index
    is into the candidate list, which its source layer writes whole with -1 padding
    (`write_final`). A row past the list's length is padding, so like a masked candidate it
    loads position 0, which is visible, and its score is masked to -inf. The release's
    initcheck errors confirm it (below): all 12,480 are the dense cases' rows past their
    extents.
  - Rows are independent in the block-scaled MMA, and the store drops exactly the rows
    that changed, so **every stored score is bit for bit as before**.

**Cost.** The first form of 0503 clamped the row index instead,
`position(p0, T.min(p, extent - 1 - p0))`. It was bit-identical but 28-64% slower on
dgx4, because it doubled the kernel's registers and halved its occupancy (below). The
predicated form keeps the release's registers (127; the release's 124) and compiles its
load phase straight-line, where nvcc had split the release's into branches. On dgx4 it is
3-24% faster than the release's kernel in every dense case. Candidate mode is the
release's kernel.
- **Gate:** `benchmark_deepseek_v41_indexer_bound.py` measures it. It times the kernel
  against the release's at 6 and 96 decode rows, on the ratio-2 caches' 128-row pages
  (extents up to 524,261 positions) and layer 20's 256-row pages (up to 1,048,549), dense
  and by candidates. Each case first compares the two kernels bit for bit, then times
  eight paired CUDA-graph runs. `--gate` fails if any case is slower beyond its 95%
  interval.

**Tests** (vLLM, `tests/kernels/attention/`):
- `test_indexer_reads_only_visible_rows` runs `deepseek_v41_indexer_reads.py` under
  `compute-sanitizer --tool initcheck` with `PYTORCH_NO_CUDA_MEMORY_CACHING=1`, so every
  tensor is a fresh allocation.
  - Every input is written whole, except that the index cache is written only at each
    row's visible positions.
  - Cases: dense and candidate mode, 128- and 256-row pages, extents ending inside a tile.
  - initcheck reports any read of the unwritten bytes, so the test fails if a row past the
    visible length is read. It skips where no sanitizer is found (`COMPUTE_SANITIZER`
    names one).
- `test_indexer_scores_ignore_rows_past_the_extent` puts NaN bytes in every slot past
  each row's extent (UE8M0 0xFF scales) and checks that every stored score stays
  bit-identical.
- **The [indexer-initcheck](bundles/indexer-initcheck) bundle** (`tp4-gate.json`; run on
  dgx4, below) runs the script under the nodes' host `compute-sanitizer`
  (`/usr/local/cuda-13.0/compute-sanitizer`, as the 2026-09-29 determinism experiment
  mounted it):
  - the bounded kernel must report no error;
  - **the release's kernel must report errors** (initcheck's exit code 86), the positive
    control that the check sees the reads 0503 removes.

## Poison mode (0500)

`VLLM_KV_CACHE_POISON=1` is debug only, off by default and never in the recipe.

- New blocks are filled with all-ones bytes. 0xFF is NaN in FP32, BF16, FP16, E4M3, E5M2
  and UE8M0, at every byte offset of any record layout. E2M1 has neither NaN nor Inf; its
  nibble reads -6, and its scale beside it is NaN.
- The fill runs for every cache, 0502's included, and also for the compressor rings'
  blocks, which zeroing never touched.
- Warmup compiles the fill on the null block with zeros, so block 0 stays as the pool
  starts.

The fill is `fill_kv_blocks_kernel` (`vllm/v1/worker/gpu/tilelang_inputs.py`). It uses the
zeroing kernel's segments, pages and predicated stores, and is declared through
`tilelang_inputs.declare` so the frozen kernel cache accepts it.

With the switch off nothing of it is declared or compiled. Poison is a superset of what
the end state leaves in a block, because every new block is maximally stale.

## vLLM changes

Branch `kv-zeroing-poison` in `~/projects/vllm-kv-zeroing` (local, not pushed), on
candidate 6's vLLM `3bdf9558c` (26382f5cf plus the test patches 0489-0490):

- [0500](vllm/0500-KV-VLLM_KV_CACHE_POISON-new-blocks-filled-with-NaN-b.patch) `341928840`:
  the poison mode.
- [0501](vllm/0501-DS4.1-DSpark-draft-windows-start-at-CED-s-replay-sta.patch) `009ad2699`:
  draft windows start at CED's replay start.
- [0502](vllm/0502-DS4.1-No-per-allocation-KV-zeroing-every-cache-uses-.patch) `8d2842981`:
  DS4.1 without per-allocation zeroing. It adds `uses_only_written_slots`, the rule
  change, and the declaration on the window, compressed, index and ring specs.
- [0503](vllm/0503-DS4.1-Indexer-scores-read-no-key-past-the-visible-po.patch) `169078a4c`:
  the indexer reads no key past the visible positions: a dense row's loads past its
  extent are predicated off. Its tests are the initcheck script and its wrapper and the NaN-tail test; it adds
  the gated benchmark, dense and by candidates.
- [vllm/series](vllm/series): candidate 6's series plus 0500-0503. 0491-0499 are other
  drafts'.

### Tests

CPU, macOS (torch 2.14.1, no TileLang or B12X), [runs/cpu-tests.txt](runs/cpu-tests.txt):

- `tests/v1/core`, `tests/v1/worker` and `tests/test_envs.py`: base 1,674 passed, 0500-0501
  1,689, 0500-0502 **1,692**. All three have the same 159 failing or erroring ids:
  offline hub ids, CPU-build modules, no TileLang. 0503 changes no CPU-run file: its
  kernel and tests need TileLang and a GPU.
- New CPU tests:
  - the zeroer fills where it zeroes;
  - the warmup keeps the null block zero;
  - the declared kernel by mode;
  - the poison bytes are NaN in each float format;
  - the forced fill;
  - DS4.1's FP32 ring is the only mixed precision;
  - the ring manager records its block under the poison;
  - a mixed cache that declares written-slot use skips zeroing, while Mamba, an
    undeclared layer and the poison keep it;
  - every merge keeps the declaration.

GPU and image, in the [poison-tests](bundles/poison-tests) bundle (`tp4-gate.json`, dgx2):

- the TileLang fill writes exactly where the Triton zeroing clears, and launches only its
  declared call;
- every zeroer and runner-input test;
- DS4.1's own window, compressed, index and ring specs declare the property and skip
  zeroing (`tests/models/test_deepseek_v4_1_compressor.py`, which needs the image's B12X
  import);
- the window metadata, TileLang against Triton, including draft rows with a replay start;
- the two drafter cases begin at the hit and at the kept row;
- the TileLang attention kernels, the indexer's NaN-tail test included;
- the indexer benchmark against the release's, bit for bit, then timed and gated.

[indexer-initcheck](bundles/indexer-initcheck) (`tp4-gate.json`, dgx3): the read bound
under initcheck, and the release's kernel as the positive control (0503 above).

`tests/v1/core/test_single_type_kv_cache_manager.py` joins `config/release-tests.json`'s
gate. `scripts/release_tests.py check` reports 0 problems.

## GPU checks on dgx4 (2026-10-25, CET)

The owner opened dgx4 to single-node checks while the triangle serves TP3. Every run below
used `scripts/lab.py kernel-local` on dgx4 and nothing else. Each ran from a checkout of
this branch at `~/spark-lab/agents/kv-zeroing/spark-ds41f`, on the
`spark-ds41f:19f2c20ed4d6-next-6` image (`sha256:82b071b57485…`), and wrote its output
under `~/spark-lab/agents/kv-zeroing/out/` on dgx4:

```sh
python3 scripts/lab.py kernel-local experiments/2026-10-25-kv-zeroing/bundles/BUNDLE --out ~/spark-lab/agents/kv-zeroing/out/OUTPUT
```

| Bundle | Commit | 0503 | Output | Verdict |
| --- | --- | --- | --- | --- |
| indexer-initcheck | `329cebf2` | `57e5e748f`, clamped | `indexer-initcheck` | **pass**, 53 s |
| poison-tests | `6eed8307` | `57e5e748f`, clamped | `poison-tests` | **fail**, 315 s: every test passed, the benchmark gate failed |
| indexer-variants | `b679dbad` | | `indexer-variants` | diagnosis: v1-v3 |
| indexer-variants | `0b41a754` | | `indexer-variants-v4` | diagnosis: v1-v4 |
| indexer-variants | `64a303eb` | | `indexer-variants-v5` | diagnosis: v5 and v4, dense and by candidates |
| poison-tests | `1f180481` | `33b748994`, predicated | `poison-tests-2` | **fail**, 105 s: every test passed; the gate: dense faster in 10 of 10, candidate mode slower in 2 of 4 |
| indexer-initcheck | `1f180481` | `33b748994`, predicated | `indexer-initcheck-2` | **pass**, 38 s |
| **poison-tests** | `93df83d5` | `169078a4c`, dense rows predicated | `poison-tests-3` | **pass**, 83 s: every test passed; the gate: slower in 0 of 14 |
| **indexer-initcheck** | `93df83d5` | `169078a4c`, dense rows predicated | `indexer-initcheck-3` | **pass**, 28 s: 0 errors; the release's kernel 12,480, exit 86 |

**indexer-initcheck** (0503 as first committed, `57e5e748f`):
- the bounded kernel: `ERROR SUMMARY: 0 errors` over the four cases (dense and candidate
  mode, 128- and 256-row pages);
- the release's kernel: 12,480 uninitialized `__global__` reads, exit 86. The positive
  control holds: the check sees the reads 0503 removes.

**poison-tests**, step by step:

| Step | Result |
| --- | --- |
| zeroing, the poison fill and the runner's declared calls | 686 passed, 1 skipped (`test_gpu_tilelang_inputs.py:942`, an existing skip: no budget below the scheduled drafts) |
| the zeroing rule and the declaration's merges | 6 passed |
| the ring's recorded blocks and DS4.1's specs | 28 passed (2 deselected, as the release tests deselect them) |
| window metadata: TileLang against Triton, CED and the draft windows | 131 passed |
| the TileLang attention kernels, the indexer's read bound included | 32 passed, 1 skipped (the in-image initcheck test: the image has no `compute-sanitizer`; the initcheck bundle runs it with the host's), 1 xfailed (the known score-ReLU NaN, strict) |
| the bounded indexer against the release's: bits, then time (gated) | **exit 1**: bits equal in all 10 cases, slower in all 10 |

The gate's timings, µs a call (median of eight paired CUDA-graph runs):

| Case | Pages | Extent | 0503 | Release | Slower |
| --- | --- | --- | --- | --- | --- |
| c1, 6 rows | 128 | 4,133 | 14.9 | 11.7 | 28% |
| c1 | 128 | 65,573 | 182 | 132 | 38% |
| c1 | 128 | 524,261 | 1,604 | 1,069 | 50% |
| c1 | 256 | 8,229 | 25.5 | 19.4 | 32% |
| c1 | 256 | 1,048,549 | 3,516 | 2,260 | 56% |
| c16, 96 rows | 128 | 4,133 | 214 | 146 | 47% |
| c16 | 128 | 65,573 | 3,074 | 2,113 | 45% |
| c16 | 128 | 524,261 | 25,840 | 16,697 | 55% |
| c16 | 256 | 8,229 | 486 | 297 | 64% |
| c16 | 256 | 1,048,549 | 56,467 | 35,672 | 58% |

**Root cause: registers, not instructions.** The cost bound above assumed one integer
`min` per row index. Instead the clamp cost half the kernel's occupancy:
- **Release:** each thread issues 32 one-byte key loads a tile. With 64-aligned tiles that
  never straddle a page, TileLang proves the page read uniform per tile and each load's
  address affine: one base register plus immediate offsets.
- **0503:** `min(p, extent - 1 - p0)` makes every one of the 32 addresses a separate
  64-bit value. Its page index changes per load too. nvcc keeps all 32 loads in flight, so
  it needs all of those addresses live at once.

Register counts, from `cuobjdump -res-usage` on dgx4's compile cache, with blocks of 128
threads an SM out of 64K registers:

| Kernel | Dense | Candidates | Blocks an SM by registers (dense) |
| --- | --- | --- | --- |
| release | 124 | 102 | 4 |
| v1: 0503 as committed | 198 | 174 | 2 |
| v2: v1 with one page read a tile | 202 | 174 | 2 |
| v3: release loads, the whole load guarded by the extent | 127 | 70 | 4 |
| v4: release loads for whole tiles, v1's for the last | 255 | 178 | 2 |
| v5: release addresses, only the key and scale loads predicated | 127 | 96 | 4 |

The [indexer-variants](bundles/indexer-variants) bundle compiled each variant, printed
its CUDA, compared bits and timed it:

| Variant | Bits | c1, 128-row pages, 4,133 | c1, 524,261 | c16, 524,261 | c16, 256-row pages, 1,048,549 |
| --- | --- | --- | --- | --- | --- |
| v1 | equal | +28% | +49% | +56% | +57% |
| v2 | equal | +7% | +10% | +16% | +17% |
| v3 | equal | +87% | +186% | +180% | +212% |
| v4 | equal | +4% | +3% | +11% | +16% |
| **v5** | equal | **-20%** | **-20%** | **-20%** | **-3%** |

- **v3** keeps the release's registers but puts the page read inside the guard. A
  conditional load cannot be hoisted, so every one of the 32 iterations waits on two
  dependent loads.
- **v4**'s whole-tile path compiles to exactly the release's CUDA, and its clamped path
  runs at most once a row. It is still slower: a kernel's registers are allocated for its
  worse path.
- **v5** is the release's CUDA with the extent added to the bounds predicate of each load.
  In SASS the release's load phase has 48 byte loads and 112 byte stores across three
  branches: nvcc unswitched its uniform page-range check. v5's per-row predicate cannot
  be unswitched, so its 32 loads and 32 stores run straight-line (2,169 SASS lines to
  2,361). The load phase is the only place the CUDA differs, so its gain most likely
  comes from there.

By candidates (the reindex layers), from the v5 run, µs a call:

| Variant | c1, 128-row pages, 524,261 | c16, 256-row pages, 1,048,549 |
| --- | --- | --- |
| v4 | 43.8 vs 33.3: +32% | 990 vs 675: +47% |
| **v5** | 34.2 vs 33.3: level (+0.0 ± 2.0) | 676 vs 709: level (-17 ± 30) |

**0503 took v5's form** (`33b748994`), and the benchmark gate gained candidate cases.
Its poison-tests run passed every test. Its gate, µs a call:

| Case | Pages | Extent | 0503 | Release | Change |
| --- | --- | --- | --- | --- | --- |
| c1 | 128 | 4,133 | 9.37 | 11.63 | -19% |
| c1 | 128 | 65,573 | 99.6 | 131.6 | -24% |
| c1 | 128 | 524,261 | 846 | 1,089 | -22% |
| c1 | 256 | 8,229 | 15.0 | 19.3 | -22% |
| c1 | 256 | 1,048,549 | 1,979 | 2,296 | -14% |
| c16 | 128 | 4,133 | 114 | 145 | -22% |
| c16 | 128 | 65,573 | 1,607 | 2,124 | -24% |
| c16 | 128 | 524,261 | 13,504 | 16,745 | -19% |
| c16 | 256 | 8,229 | 252 | 324 | -22% |
| c16 | 256 | 1,048,549 | 35,468 | 36,381 | -3% |
| c1, candidates | 128 | 524,261 | 33.8 | 32.8 | **+1.5 ± 1.2 µs, slower** |
| c1, candidates | 256 | 1,048,549 | 35.0 | 34.1 | **+0.8 ± 0.5 µs, slower** |
| c16, candidates | 128 | 524,261 | 537 | 541 | level (-7 ± 29) |
| c16, candidates | 256 | 1,048,549 | 589 | 613 | level (-28 ± 29) |

**Candidate mode never needed the predicate.** Its CUDA differs from the release's only
by the extent term, which cost about 3% in the latency-bound c1 case. The term bounds
nothing there:
- the source layer's `write_final` writes the whole candidate list, -1 padded;
- a row past the list's length is padding, which loads position 0, a visible slot;
- the release kernel's initcheck errors are all dense. The 12,480 errors equal the dense
  cases' rows past their extents, 65 reads a row (64 key bytes and a scale word):
  (63 + 27 + 0 + 20) rows at 128-row pages plus (28 + 27 + 27) at 256-row pages.
  The candidate cases added none.

So 0503 (`169078a4c`) predicates only dense rows, through a per-mode `loads(p0, p)`
beside the kernel's `position` and `slot_valid`. Candidate mode passes `True`, which TVM
folds away, so its kernel is the release's. The compile cache shows it: in the third
poison-tests run the benchmark's candidate kernels were cache hits on the release's own
entries, and the only new candidate kernel was the tests' 64-row-page one.

**Third run, both bundles pass.**
- poison-tests: every step as in the first run, 686 / 6 / 28 / 131 / 32 passed (the same
  skips and xfail).
- The gate, slower in 0 of 14 cases:

| Case | Pages | Extent | 0503 | Release | Change |
| --- | --- | --- | --- | --- | --- |
| c1 | 128 | 4,133 | 9.41 | 11.62 | -19% |
| c1 | 128 | 65,573 | 99.1 | 130.5 | -24% |
| c1 | 128 | 524,261 | 866 | 1,105 | -22% |
| c1 | 256 | 8,229 | 15.1 | 19.3 | -22% |
| c1 | 256 | 1,048,549 | 1,987 | 2,281 | -13% |
| c16 | 128 | 4,133 | 113 | 145 | -22% |
| c16 | 128 | 65,573 | 1,632 | 2,122 | -23% |
| c16 | 128 | 524,261 | 13,460 | 16,711 | -19% |
| c16 | 256 | 8,229 | 253 | 298 | -15% |
| c16 | 256 | 1,048,549 | 35,432 | 36,383 | -3% |
| c1, candidates | 128 | 524,261 | 32.7 | 32.8 | level (-1.3 ± 3.1) |
| c1, candidates | 256 | 1,048,549 | 34.1 | 34.1 | level (+1.0 ± 2.4) |
| c16, candidates | 128 | 524,261 | 557 | 535 | level (+13 ± 32) |
| c16, candidates | 256 | 1,048,549 | 599 | 640 | level (-15 ± 26) |

- indexer-initcheck: the bounded kernel `ERROR SUMMARY: 0 errors` over the four cases; the
  release's 12,480 errors, exit 86.

## Lab spec: the gate before the removal is promoted

Queue after the release window, once this PR is on main; the lab syncs origin/main.
Generated by [make_bundles.py](make_bundles.py). Every arm counts NaN logits with
`VLLM_COMPUTE_NANS_IN_LOGITS=1`, which no lab run has used before; if it stops a boot,
drop it.

### Arms

Before and after the removal, on candidate 6's image:

| Arm | Over | What a new block holds |
| --- | --- | --- |
| `zero-tp4` | `rc6/rc-tp4.json` + 0500-0501 (9 package files), poison off | zeros: candidate 6's behaviour, the reference |
| `nofill-tp4` | `rc6/rc-tp4.json` + 0500-0503 (11 files), poison off | its previous owner's bytes: **the end state** |
| `poison-tp4` | `nofill-tp4` + `VLLM_KV_CACHE_POISON=1` | NaN bytes in every format: **the gate** |
| `poison-tp4-dev` | `poison-tp4` + `VLLM_SERVER_DEV_MODE=1` | the release suite's engine grid |
| `*-pressure-tp4` | each of the three | the same, with `--max-model-len 262144` and `--kv-cache-memory-bytes 1342177280` (5,890 blocks; one request needs 4,895) |
| `poison-unfixed-tp4` | `rc6/rc-tp4.json` + 0500 alone, poison on | the positive control |
| `*-tp3` | `rc6/rc-tp3.json` (PR #582, 8,192 batched tokens) | the same; the pressure pool is 805,306,368 bytes (3,534 blocks; one request needs 2,975) |

### Specs, in queue order

1. **[tp4-gate.json](tp4-gate.json)** (run `kv-zeroing-tp4`). Sync, the poison-tests
   bundle on dgx2 and the indexer-initcheck bundle on dgx3, then `poison-tp4`,
   `nofill-tp4` and `zero-tp4`. Each runs:
   - the determinism gate with and without logprobs, `--tokens 192 --long-lines 700 --wide`;
   - [prefix_extension.py](prefix_extension.py) `--threshold 16288`: a hit at 32,512 with a
     64-row first chunk, compared with the same prompt computed fresh;
   - `controls_digest.py` and `sampled_digest.py`, as candidate 6's gate ran them;
   - [kv_counters.py](kv_counters.py): NaN-logit requests, preemptions and hits, and each
     node's device asserts and NaN reports.

   About 100 min.
2. **[tp4-pressure.json](tp4-pressure.json)** (run `kv-zeroing-tp4-pressure`). The three
   pressure arms run [pool_pressure.py](pool_pressure.py)
   `--streams 16 --documents 12 --tokens 32000 --max-tokens 2048`:
   - **alone:** the 16 requests one at a time, with 4 documents asked twice, so prefix hits
     occur;
   - **together:** all 16 at once, so pool exhaustion forces preemption, eviction and
     blocks handed between groups;
   - every request's tokens are compared between the two passes;
   - then kv_counters.

   About 60 min.
3. **[tp4-suite.json](tp4-suite.json)** (run `kv-zeroing-tp4-suite`). `poison-tp4-dev`
   runs the `release` suite (1,000,000-token prefill included) and `single-stream`.
   [suite_model.py](suite_model.py) compares the model step's 256 responses, draft
   acceptance by position and teacher-forced scores with candidate 6's release run
   (`candidate-6-release-tp4/candidate-6`). About 60 min.
4. **[tp4-control.json](tp4-control.json)** (run `kv-zeroing-tp4-control`).
   `poison-unfixed-tp4` runs prefix_extension.py (a 52,612-token prompt: CED one-row
   chunks and a short first chunk), then kv_counters. **It is expected to fail**, with
   device asserts or an engine that dies. That proves the gate sees the drafter
   divergence. About 10 min.
5. **TP3, after the re-wire**, with #582 merged and the window's production profile as in
   the TP3 qualification:
   - [tp3-gate.json](tp3-gate.json): `--threshold 8144`, a hit at 24,320 with a 112-row
     first chunk;
   - [tp3-pressure.json](tp3-pressure.json): 8 streams, 6 documents, 48,000 tokens,
     4,096 out;
   - [tp3-suite.json](tp3-suite.json): `release-tp3` and `single-stream`.

   If the triangle qualifies at 6,144, point make_bundles.py's TP3 arm at
   `rc6/rc-tp3-6144.json` and the threshold at 6096, regenerate, and raise the pressure
   prompts to 56,000 tokens.

```sh
.venv/bin/python scripts/lab.py run experiments/2026-10-25-kv-zeroing/tp4-gate.json --dry-run
scripts/lab.py queue add experiments/2026-10-25-kv-zeroing/tp4-gate.json   # then -pressure, -suite, -control
```

### Pass, and then the removal

- `tables_arms.py RUN LABELS` (each spec's table) prints a verdict per arm. `poison-*` and
  `nofill-*` must each show:
  - both determinism records identical, with the release's digests (`3eab5112` /
    `db25fde3`, text `c9cf9dbe` / `c6bb4f4e`; TP3: `62313cb5` / `9742ac2c`, `4d143977` /
    `12fd8257`);
  - the prefix extension identical, with a short first chunk;
  - controls and sampled digests equal to the zero arm's and candidate 6's;
  - pool pressure identical, preempted, and alone-pass outputs equal to the zero arm's;
  - zero NaN-logit requests and zero node reports.
- `suite_model.py kv-zeroing-tp4-suite poison-tp4`: 256/256 identical, acceptance by
  position equal, identical distributions.
- `zero-*` must pass too. Otherwise the difference is not the removal's.
- A missed point makes that part inconclusive, not passed: no preemption, or a hit shorter
  than the chosen point.

When tp4-gate, tp4-pressure and tp4-suite pass, and the TP3 specs after the re-wire:

1. 0500-0503 go into the next candidate's series. 0503's benchmark gate and initcheck
   already passed on dgx4 (above).
2. The recipe needs no change: the removal is the model's declaration, not a flag.
3. The candidate's release comparison then runs on the end state.

The poison specs become the recurring check for any release that changes:
- a KV reader or writer;
- the window or index metadata;
- the compressor;
- CED or DSpark scheduling;
- block allocation.

## Files

| File | What |
| --- | --- |
| [make_bundles.py](make_bundles.py) | bundles, arms and specs from the vLLM branch (`python3 make_bundles.py ~/projects/vllm-kv-zeroing`) |
| [bundles/poison-tests](bundles/poison-tests) | the kernel-lab GPU tests and the indexer benchmark |
| [bundles/indexer-initcheck](bundles/indexer-initcheck) | the indexer's read bound under initcheck, with the positive control |
| [bundles/indexer-variants](bundles/indexer-variants) | the read bound's diagnosis: five bounded variants against the release's kernel (hand-made; v1-v3 from [make_variants.py](bundles/indexer-variants/make_variants.py)) |
| [bundles/serving](bundles/serving), [bundles/serving-zeroing](bundles/serving-zeroing), [bundles/serving-unfixed](bundles/serving-unfixed) | package files the arms mount (0500-0503; 0500-0501; 0500 alone) |
| `zero-*.json`, `nofill-*.json`, `poison-*.json` | arms |
| `tp4-*.json`, `tp3-*.json` | lab specs |
| [prefix_extension.py](prefix_extension.py), [pool_pressure.py](pool_pressure.py), [kv_counters.py](kv_counters.py) | workloads and counters |
| [tables_arms.py](tables_arms.py), [suite_model.py](suite_model.py) | verdicts |
| [zero_share.py](zero_share.py), [runs/zero-share.txt](runs/zero-share.txt) | the gain bound |
| [runs/cpu-tests.txt](runs/cpu-tests.txt), [runs/run-cpu-tests.sh](runs/run-cpu-tests.sh) | the CPU test runs |
| [vllm/](vllm/) | patches 0500-0503 and the series |

## Not done, and overlaps

- **retire-conditionals** (0495) deletes the Triton `_chunk` that 0501 also edits. Whichever
  lands second gives its torch reference the draft branch's replay bound.
- **The initcheck bundle** mounts the nodes' host sanitizer at
  `/usr/local/cuda-13.0/compute-sanitizer`, the path the 2026-09-29 experiment used and
  dgx4 still has. If a host's toolkit has moved, point `SANITIZER` in make_bundles.py at
  it.
- **ruff** reports two problems that already exist at the base, both outside this change:
  `nvidia/model_state.py`'s import block, and `single_type_kv_cache_manager.py:2680`.
- **Upstream:** the declaration is upstreamable as it stands. vLLM's rule otherwise sees
  only dtypes, not page layouts.
