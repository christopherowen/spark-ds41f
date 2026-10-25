# KV block zeroing: can DeepSeek V4.1 serving drop it?

2026-10-25, CET. For the release after spark-ds41f-1025; nothing here touches candidate 6.
**Nothing has run on a GPU yet.** The release freeze holds the cluster, and the laptop has
no docker. The numbers below come from existing traces and the code. The vLLM changes ran
in CPU tests only. The lab specs are queued work for after the release.

## Question and answer

The owner asked whether removing vLLM's KV block zeroing (`zero_kv_blocks`, 0.20% of
prefill in the kernel audit, spark-ds41f #585) could let old bad memory hurt us.

- **Yes, today, through one reader: DSpark's draft windows.** On two kinds of nonfinal
  prefill chunk, the drafter's window metadata marks valid slots that nothing has written
  in the block's current ownership:
  - CED's one-row chunks (every prompt longer than one chunk plus 143 rows);
  - a first chunk shorter than 128 rows after a prefix hit.

  Zeroing makes those slots zero records. Without it they hold another group's bytes, and
  as E4M3 or UE8M0 those can decode to NaN. The drafter's MoE router then refuses the row
  with a device assert (`tilelang/router.py:460`), which aborts the engine. The drafts
  themselves are discarded (`scheduler.py:3705`), so outputs and acceptance never depended
  on them.
- **0501 fixes the reader.** A draft window now starts at CED's replay start, the first
  position where the drafter has context, as the decoder's windows already do. Decode and
  final chunks are unchanged.
- **No arithmetic-masked read of unwritten slots exists in the serving kernels.**
  - The sparse MLA never loads a masked entry from the cache. Its cp.async copies zero
    bytes and zero-fills shared memory.
  - The indexer computes stale rows, but each MMA row stands alone and the store is
    predicated on the extent and a select.
  - The compressor ring reads one slot under a select, and that slot is always written
    first. The ring has never been zeroed.
  - The kernel table below has the details.
- **The gain is at most 0.20% of prefill and about 0 in decode.** That is below the release
  suite's resolution (±0.5% on prefill).
- **Recommendation: keep zeroing.** Land 0500 and 0501, and run the poison check whenever a
  release touches KV readers, writers or block allocation. Details are under
  [Recommendation](#recommendation).

## Why DS4.1 zeroes at all

vLLM zeroes new blocks when `needs_kv_cache_zeroing` holds, which is for Mamba layers or a
mixed-precision cache (`vllm/v1/kv_cache_interface.py:1680`, upstream at our base
19f2c20ed).

DS4.1's attention pages are all bytes, `uint8` with `cache_dtype_str="b12x_dsv41"`
(`attention.py:280`, `:631`):
- window pages are 528 B a token: E4M3 data plus 16 UE8M0 scales;
- compressed latents are 288 B a state: E2M1 data plus E4M3 scales;
- index keys are 68 B a state: E2M1 data plus 4 UE8M0 scales.

Only the compressors' open-group ring is FP32 (`compressor.py:109`, `CircularBufferSpec`).
**The ring alone makes the cache mixed-precision**
(`test_ds41_shaped_cache_zeroing_comes_from_the_fp32_compressor_ring`), and the ring's own
blocks are excluded from zeroing (`single_type_kv_cache_manager.py:196`). The rule is also
blind to layout: the three byte layouts reinterpret each other's bytes just as precisions
would.

All groups draw whole blocks from one pool of 227,840-byte blocks
([kv-capacity](../2026-10-25-kv-capacity/README.md#the-pool)), so any freed block can come
back as any group's page.

The pool is allocated with `torch.zeros` (`vllm/v1/worker/utils.py:445`), so a block's
first use is zeros with or without the zeroer. The null block 0 is never handed out and
never written: writers skip slot -1, and page tables map pages ≤ 0 to -1.

Zeroing covers reads of slots not yet written in a block's current ownership. It does
not cover reads of a block after reallocation; that is #588, the drafter window.

## The gain

From request inventory 5's TP4 traces (rank 0, the build profiled for the kernel audit):
[zero_share.py](zero_share.py), recorded in [runs/zero-share.txt](runs/zero-share.txt). It
cuts steps the way the audit's `kernel_shares.py` does.

| Step | Zeroing launches | Zeroing time | Share of busy time |
| --- | ---: | ---: | ---: |
| The 15,395-token prefill | 1 | 4,255 µs | **0.199%** |
| Short prompts' prefill (12 scenarios) | 1 each | 13-21 µs | 0.01-0.03% |
| mixed-c8's prefill / its mixed steps | 1 / 1 | 142 / 14 µs | 0.075% / 0.007% |
| Decode: greedy (11 steps), 11 c1 types (7-29 steps), mixed-c8 (32 steps) | 0 | 0 | **0%** |

- **Prefill:** a 16K chunk allocates about 1,984 blocks (15 window groups × 128, plus 64
  compressed): 452 MB, zeroed in 4.3 ms. At 6,000 tok/s (spark-ds41f-1022's 32K and 128K
  prefill) that chunk takes 2.7 s, so the share is 0.16%. It is lower at depth, where
  chunks take longer.
- **Decode:** a request needs new blocks only every 128 tokens per window group, about
  every 55 steps at 2.3 tokens a step. Each is a 15-block launch, about 35 µs. That comes
  to about 0.002% at one stream and 0.02% at 16. None fell inside the profiled decode
  windows.
- **Bound: at most 0.2% of prefill time, about 0 in decode.** The release suite resolves
  prefill to about ±0.5% (1022's 32K and 128K: ±0.48%, ±0.56%), so the gain would not
  show.

## Static review: every kernel that reads KV pages

TileLang serving path, candidate 6's vLLM (3bdf9558c). Paths are under
`vllm/models/deepseek_v4_1/` unless they say otherwise. "Unwritten" means not written in
the block's current ownership.

| Kernel | Where | Reads | Can reach unwritten slots | Masking | Verdict |
| --- | --- | --- | --- | --- | --- |
| Window slot lists, target and CED decoder (`_window_macro`) | `tilelang/metadata.py:142` | block table | holes, null block, pad rows (req/pos -1), CED-skipped rows | select: `swa[row, col] = T.if_then_else((page_id > 0) & (logical >= 0) & (col < info[1]), physical, -1)` (:181); CED decoders start at `replay_start` (:162) | safe |
| **Window slot lists, DSpark drafter** (`_window_macro`, `draft`) | `tilelang/metadata.py:154` | block table | **yes**: `info[0] = max(prefix - window, 0)` with no replay bound (`chunk`/`decode_metadata`: `replay = ... and not draft`); see below | none: the slots are valid entries | **at risk; fixed by 0501** |
| Index and main page tables (`metadata_pages`, `decode_metadata` part 1) | `tilelang/metadata.py:300`, `:329` | block table | holes, null block | select `T.if_then_else(page_id > 0, page_id, -1)` | safe |
| Token metadata (`metadata_tokens`) | `tilelang/metadata.py:70` | block table | pad rows | `valid = valid1 & (block > 0)` (:125); `visible = (pos + 1) // ratio` only for valid rows | safe; sets the indexer's extent |
| Sparse MLA gathers (`gather_swa`, `gather_idx`) | `tilelang/sparse_mla.py:211`, `:220` | slot lists, top-k | entries past the lengths, -1, a -1 page | select to slot -1 | safe |
| Sparse MLA staging (`stage_records`) | `tilelang/sparse_mla.py:237` | 528 B window or 288 B compressed records | only masked entries | `T.ptx_cp_async(..., 16, copy_slot[k] >= 0)` (:260): src-size 0 zero-fills 16 B and reads no global memory (TileLang `cp_async_gs_conditional`) | masked entries never touch the cache |
| Sparse MLA dequant (`dequant_swa`, `dequant_idx`) | `tilelang/sparse_mla.py:290`, `:302` | E4M3 × UE8M0 (`shift_left(byte, 23)`: 0xFF is +Inf) / E2M1 × E4M3 | unmasked entries only | none needed: masked rows are zeros | safe **if** the metadata marks only written records valid (the drafter did not) |
| Sparse MLA softmax (`chunk_update`) | `tilelang/sparse_mla.py:345` | shared memory | — | scores `-inf` and P `0` by select (:353, :367); the PV GEMM (:375) multiplies zero rows by P = 0 | safe |
| Indexer scores, dense | `tilelang/indexer.py:133` | 64 B E2M1 + UE8M0 word per position, the last 64-row tile past the extent, holes clamped to block 0 (:166) | yes: rows past the extent, clamped rows | rows are independent in the block-scaled MMA; the store is predicated, `if p0 + p < extent:` (:202), then a select to `-inf` | safe |
| Indexer scores, candidates (reindex layers 24-36) | `tilelang/indexer.py:244` | same | -1 candidates load position 0 | `T.max(Candidates[...], 0)` and `Candidates[...] >= 0` into the same select (:270) | safe |
| Indexer top-k (segment, merge) | `tilelang/indexer.py:598`, `:799` | score scratch, winner lists | stale scratch | selects on `e < n_entries`, `live`, `>= 0`; integer keys (`_key16`, :400) | safe |
| Writers: window and compressed records, index keys | `tilelang/cache_writer.py:124`, `tilelang/indexer.py:341` | — | — | whole records (data and scale), `if (slot >= 0) & (page < pages)` | write-only |
| Compressor ring gather (`compress_gather_partial_state`) | `tilelang/compress.py:54` | one FP32 slot, `(pos - 1) % 8`, of the request's ring | only at an odd first position (:79), and every ring ownership starts at an even one: fresh at 0, prefix hits on 256-token blocks, preemption re-admits | select `T.if_then_else(valid, pool[...], 0.0)` (:85) | safe (the ring is not zeroed today) |
| Compressor (`compress`, ratios 1 and 2) | `tilelang/compress.py:113` | carried state | invalid rows | carry only when `tags[sid] == info[2] - 1` (:183); invalid rows zero by select; whole ring slots written | safe |
| DSpark context slots, CED kernels, Engram lookback, SP prefill | `vllm/v1/worker/gpu/tilelang_inputs.py:2143`, `tilelang/dspark_context.py`, `tilelang/ced.py`, `tilelang/lookback.py`, `sp_prefill.py` | metadata, activations, token ids | — | predicated loads; rejected rows PAD | no KV reads |

Nothing relies on zeroed cache memory for correctness:
- the compressor starts its state from a select-zero carry;
- CED's zero fills act on activations;
- the only zero fill the attention needs is cp.async's, in shared memory.

The B12X and Triton paths are not what serving launches under `VLLM_DS41_KERNEL_BACKEND=tilelang`.
The Triton `_chunk` stays as the reference that the TileLang metadata is tested against.

## The drafter finding (0501)

DSpark's context KV is written only for the CED decoder's kept rows. CEDState's indices
pack the drafter's context (`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:619`:
"discarded decoder rows have no context KV"). The drafter's three window caches are not
prefix-cached (`attention.py:294-302`).

The drafter drafts for every request in the batch, prefill chunks included
(`speculator.py:577`). A draft row's window was `[max(prefix - 128, 0), prefix + 6)`:

1. **CED one-row chunks** (`ced.py:105`, `limits[i] = 1` at :140). A nonfinal aligned
   chunk keeps only its last row for the decoder and the drafter. The window's other 127
   positions are in blocks allocated for that chunk and never written.
2. **A short first chunk after a prefix hit at L.** Aligned chunking makes the first chunk
   `T - L % T` rows. On the ring (T = 16,288) at L = 32,512 that is 64 rows; on 1022's
   triangle (T = 4,048) at L = 12,032 it is 112. The window reaches into `[L - 128, L)`, a fresh block, because the
   drafter's groups were not part of the hit.

Both chunks are nonfinal, so the scheduler drops their drafts. Neither path changes outputs
or acceptance today. What zeroing hides is the abort that stale NaN bytes would cause in
the drafter's routed MoE layers.

CED's `replay_start` (`tilelang/ced.py:148`) is `max(prefix_start, the first kept row)`.
That is exactly where a request's draft context begins.

0501 makes the draft branch of the window kernel, and its Triton reference, start no
earlier than that. The model state binds its persistent buffer to the drafter's attention
layers (`bind_draft_replay_start`), so captured draft graphs read each step's values, and
the drafter declares only the replay variant.

The bound changes no decode or final-chunk window:
- after a hit, the final chunk keeps at least 128 rows
  (`kv_cache_manager.py:394`), so decode starts at least 128 past `prefix_start`;
- without a hit, `prefix_start` is 0;
- a full chunk's first kept row is its window start already.

## Poison mode (0500)

`VLLM_KV_CACHE_POISON=1` is debug only, off by default and never in the recipe.

- New blocks are filled with all-ones bytes instead of zeros. 0xFF is NaN in FP32, BF16,
  FP16, E4M3, E5M2 and UE8M0, at every byte offset of any record layout. E2M1 has neither
  NaN nor Inf; its nibble reads -6, and its UE8M0 or E4M3 scale beside it is NaN.
- The fill runs for every cache, and also for the compressor rings' blocks, which zeroing
  never touches.
- Warmup compiles the fill on the null block with zeros, which leaves block 0 as zeroing
  does.

The fill is `fill_kv_blocks_kernel` (`vllm/v1/worker/gpu/tilelang_inputs.py`). It uses the
zeroing kernel's segments, pages and predicated stores, with a word argument. It is
declared through `tilelang_inputs.declare` in place of zeroing, so the frozen kernel cache
accepts it.

With the switch off, the zeroing kernel, its key and its declaration are byte for byte as
before, and no fill is declared or compiled. The poison is a superset of what dropping
zeroing would leave in a block, because every new block is maximally stale.

A NaN reaches the outputs, or the router's device assert, wherever a kernel reads an
unwritten slot. Comparisons (`fmaxf`, the indexer's integer keys) can absorb a NaN, which
is why the static review covers the selection paths separately.

## vLLM changes

Branch `kv-zeroing-poison` in `~/projects/vllm-kv-zeroing` (local, not pushed), on
candidate 6's vLLM `3bdf9558c` (26382f5cf plus the test patches 0489-0490):

- [0500](vllm/0500-KV-VLLM_KV_CACHE_POISON-new-blocks-filled-with-NaN-b.patch) `341928840`:
  the poison mode and its tests.
- [0501](vllm/0501-DS4.1-DSpark-draft-windows-start-at-CED-s-replay-sta.patch) `009ad2699`:
  draft windows start at CED's replay start, with tests.
- [vllm/series](vllm/series): candidate 6's series plus 0500-0501. 0491-0499 are other
  drafts' (KV capacity, drafter window, retire-conditionals, final names, direct loader).

### Tests

CPU, macOS (torch 2.14.1, no TileLang), [runs/cpu-tests.txt](runs/cpu-tests.txt):

- `tests/v1/core`, `tests/v1/worker` and `tests/test_envs.py`: base 1,674 passed, branch
  **1,689 passed**. Both trees have the same 159 failing or erroring ids: hub ids not in
  the offline cache, CPU-build modules, no TileLang.
- The 15 new CPU tests:
  - the zeroer fills where it zeroes;
  - the warmup keeps the null block zero;
  - the declared kernel by mode;
  - the poison bytes are NaN in each float format, and the E2M1 magnitude;
  - the forced fill;
  - DS4.1's FP32 ring is the only mixed precision;
  - the ring manager records its block under the switch.

GPU, in the [poison-tests](bundles/poison-tests) bundle (`tp4-gate.json`, dgx2):

- the TileLang fill writes exactly where the Triton zeroing clears, and launches only its
  declared call;
- every existing zeroer and runner-input test;
- the window metadata, TileLang against Triton, now including draft rows with a replay
  start;
- the two drafter cases begin at the hit and at the kept row.

`tests/v1/core/test_single_type_kv_cache_manager.py` joins `config/release-tests.json`'s
gate, because the series changes it. `scripts/release_tests.py check` then reports 0
problems.

## Lab spec

Queue after the release window, once this PR is on main. The lab syncs origin/main, so
merge first. Generated by [make_bundles.py](make_bundles.py) from the vLLM branch. Every
arm counts NaN logits with `VLLM_COMPUTE_NANS_IN_LOGITS=1`, which no lab run has used
before; if it stops a boot, drop it from `zero-*` and `poison-*`.

### Arms

| Arm | Over | Difference |
| --- | --- | --- |
| `zero-tp4` | `rc6/rc-tp4.json` (candidate 6's image) | the 9 package files of 0500-0501 mounted, poison off |
| `poison-tp4` | `zero-tp4` | `VLLM_KV_CACHE_POISON=1` |
| `poison-tp4-dev` | `poison-tp4` | `VLLM_SERVER_DEV_MODE=1` (the suite's engine grid) |
| `zero-pressure-tp4`, `poison-pressure-tp4` | `zero-tp4`, `poison-tp4` | `--max-model-len 262144`, `--kv-cache-memory-bytes 1342177280` (5,890 blocks; one request needs 4,895) |
| `poison-unfixed-tp4` | `rc6/rc-tp4.json` | 0500's 5 package files only, poison on: the positive control |
| `*-tp3` | `rc6/rc-tp3.json` (PR #582, 8,192 batched tokens) | the same; the pressure pool is 805,306,368 bytes (3,534 blocks; one request needs 2,975) |

### Specs, in queue order

1. **[tp4-gate.json](tp4-gate.json)** (run `kv-zeroing-tp4`). Sync, the poison-tests
   bundle on dgx2, then `poison-tp4` and `zero-tp4`. Each runs:
   - the determinism gate with and without logprobs, `--tokens 192 --long-lines 700 --wide`;
   - [prefix_extension.py](prefix_extension.py) `--threshold 16288`: a hit at 32,512 with a
     64-row first chunk, compared with the same prompt computed fresh;
   - `controls_digest.py` and `sampled_digest.py`, as candidate 6's gate ran them;
   - [kv_counters.py](kv_counters.py): NaN-logit requests, preemptions and hits, and each
     node's device asserts and NaN reports from the container log.

   About 70 min.
2. **[tp4-pressure.json](tp4-pressure.json)** (run `kv-zeroing-tp4-pressure`). The two
   pressure arms run [pool_pressure.py](pool_pressure.py)
   `--streams 16 --documents 12 --tokens 32000 --max-tokens 2048`:
   - **alone:** the 16 requests one at a time, with 4 documents asked twice, so prefix hits
     occur;
   - **together:** all 16 at once, so pool exhaustion forces preemption, eviction and
     blocks handed between groups;
   - every request's tokens are compared between the two passes;
   - then kv_counters.

   About 40 min.
3. **[tp4-suite.json](tp4-suite.json)** (run `kv-zeroing-tp4-suite`). `poison-tp4-dev`
   runs the `release` suite (1,000,000-token prefill included) and `single-stream`.
   [suite_model.py](suite_model.py) compares the model step's 256 responses, draft
   acceptance by position and teacher-forced scores with candidate 6's own release run
   (`candidate-6-release-tp4/candidate-6`). About 60 min.
4. **[tp4-control.json](tp4-control.json)** (run `kv-zeroing-tp4-control`).
   `poison-unfixed-tp4` runs prefix_extension.py (a 52,612-token prompt: CED one-row
   chunks and a short first chunk), then kv_counters. **It is expected to fail**, with
   device asserts in the node logs or an engine that dies. That proves the check sees the
   drafter hazard. About 10 min.
5. **TP3, after the re-wire**, with #582 merged and the window's production profile as in
   the TP3 qualification:
   - [tp3-gate.json](tp3-gate.json): `--threshold 8144`, a hit at 24,320 with a 112-row
     first chunk;
   - [tp3-pressure.json](tp3-pressure.json): 8 streams, 6 documents, 48,000 tokens,
     4,096 out;
   - [tp3-suite.json](tp3-suite.json): `release-tp3` and `single-stream`.

   If the triangle qualifies at 6,144 instead, change `CANDIDATE`'s TP3 arm to
   `rc6/rc-tp3-6144.json` and the threshold to 6096 in make_bundles.py, regenerate, and
   raise the pressure prompts to 56,000 tokens. The 6,144 budget holds one prompt's
   window in flight in fewer blocks.

```sh
.venv/bin/python scripts/lab.py run experiments/2026-10-25-kv-zeroing/tp4-gate.json --dry-run
scripts/lab.py queue add experiments/2026-10-25-kv-zeroing/tp4-gate.json   # then -pressure, -suite, -control
```

### Pass

- `tables_arms.py RUN LABELS` (each spec's table) prints a verdict per arm.
- For `poison-tp4` and `poison-pressure-tp4`:
  - both determinism records identical, with the release's digests (`3eab5112` /
    `db25fde3`, text `c9cf9dbe` / `c6bb4f4e`; TP3: `62313cb5` / `9742ac2c`, `4d143977` /
    `12fd8257`);
  - the prefix extension identical, with a short first chunk;
  - controls and sampled digests equal to the zero arm's and candidate 6's;
  - pool pressure identical and preempted;
  - zero NaN-logit requests and zero node reports.
- `suite_model.py kv-zeroing-tp4-suite poison-tp4`: 256/256 identical, acceptance by
  position equal, identical distributions.
- The zero arms must pass the same checks. Otherwise the difference is not the poison's:
  for example, preemption recompute that differs bitwise from decode.
- A missed point makes that part inconclusive, not passed: no preemption in the together
  pass, or a hit shorter than the chosen point.

## Recommendation

Keep zeroing. Land 0500 and 0501 for next week's release, and run tp4-gate,
tp4-pressure and tp4-suite in its window.

- **Bounded gain:** at most 0.2% of prefill time, about 0 in decode. That is below what the
  release suite can see. It does not justify a model-specific change to vLLM's core
  zeroing rule.
- **What zeroing buys** is not correctness. 0501 removes the one reader that needed it.
  What it buys is that a future metadata or kernel slip, which reads a slot nobody wrote,
  sees deterministic zero records rather than another group's bytes. Those bytes can
  decode to NaN and abort the engine in the routers' finite check.
- **The poison check is what keeps zeroing honest.** It turns such slips into visible
  failures instead of hidden zero reads. Run the gate, pressure and suite specs whenever a
  release changes the window metadata, a KV writer, the compressor, CED or DSpark
  scheduling, or block allocation.
- **Precision-stable reuse: no.** Returning freed blocks only to groups of the same layout
  would split the pool into window, compressed and ring free lists, while long prefills
  shift thousands of blocks between groups. It also does not make unwritten reads correct:
  same-layout stale records are finite but belong to another request.
- **If the owner wants the 0.2%:** after tp4-gate, tp4-pressure and tp4-suite pass, removal
  is a small opt-out. DS4.1's cache specs would declare that their readers use only
  written slots, so `needs_kv_cache_zeroing` drops them, and the poison check becomes a
  required release step. The ring's FP32 rule would still need a decision upstream: vLLM
  zeroes DS4.1 only because of it.

## Files

| File | What |
| --- | --- |
| [make_bundles.py](make_bundles.py) | bundles, arms and specs from the vLLM branch (`python3 make_bundles.py ~/projects/vllm-kv-zeroing`) |
| [bundles/poison-tests](bundles/poison-tests) | the kernel-lab GPU tests |
| [bundles/serving](bundles/serving), [bundles/serving-unfixed](bundles/serving-unfixed) | package files the arms mount (0500-0501; 0500 alone) |
| `zero-*.json`, `poison-*.json` | arms |
| `tp4-*.json`, `tp3-*.json` | lab specs |
| [prefix_extension.py](prefix_extension.py), [pool_pressure.py](pool_pressure.py), [kv_counters.py](kv_counters.py) | the experiment's workloads and counters |
| [tables_arms.py](tables_arms.py), [suite_model.py](suite_model.py) | verdicts |
| [zero_share.py](zero_share.py), [runs/zero-share.txt](runs/zero-share.txt) | the gain bound from the traces |
| [runs/cpu-tests.txt](runs/cpu-tests.txt), [runs/run-cpu-tests.sh](runs/run-cpu-tests.sh) | the CPU test runs |
| [vllm/](vllm/) | patches 0500-0501 and the series |

## Not done, and overlaps

- **retire-conditionals** (0495) deletes the Triton `_chunk` that 0501 also edits. Whichever
  lands second gives its torch reference the draft branch's replay bound.
- **ruff** reports two existing problems, outside this change:
  - `nvidia/model_state.py`'s import block is unsorted at the base too;
  - `single_type_kv_cache_manager.py:2680` is unformatted at the base too.
- **Upstream:** vLLM's rule zeroes a byte-page cache only when something else in it has
  another dtype. A rule that compared page layouts as well would be the upstream form of
  this finding. Not proposed yet.
