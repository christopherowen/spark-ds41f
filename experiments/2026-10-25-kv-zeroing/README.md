# KV block zeroing: DS4.1 drops it, as DeepSeek's reference does

2026-10-25, CET. For the release after spark-ds41f-1025; nothing here touches candidate 6.
**Nothing has run on a GPU yet.** The release freeze holds the cluster, and the laptop has
no docker. The numbers below come from existing traces and the code. The vLLM changes ran
in CPU tests only. The lab specs are queued work.

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
- **Every other reader already matches REF**:
  - the sparse MLA skips -1 entries without loading them;
  - the compressor pools only slots the request wrote;
  - the window and compressed reads stop at the written positions;
  - the indexer stores no score past the visible positions. It loads its last tile's
    rows past them, which REF never reads, but those rows' results are discarded whole.

  The table below has the details. No kernel masks an unwritten slot arithmetically.
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
| Index keys, scores | reads `index_k[:bsz, : end_pos // ratio]` (L554); prefill masks positions `>= compress_lens` to -inf (L563-567) | extent `min(visible, Active, width)`, with `visible = (pos + 1) // ratio` (`tilelang/metadata.py:125-137`); scores are stored only for positions below it (`tilelang/indexer.py:202`). The last 64-row tile's rows past it (stale or null-block bytes) are loaded and computed (:164), and their results discarded: rows are independent in the block-scaled MMA | values yes; reads wider than REF |
| Index candidates (two-level top-k) | `select_candidate_blocks` (L583-610) on scores already at -inf where unreachable; keeps `top.values > -inf` | candidate scores select `-inf` for -1 candidates (`tilelang/indexer.py:270`, :203); top-k pads -1 | yes |
| Compressor state | `kv_state` zeros and `score_state` -inf at construction; prefill writes the trailing partial group into the state, decode fills slots and pools only a completed group (L458-490) | ring slot per row; the gather reads the previous slot only at an odd first position, under a select (`tilelang/compress.py:79-85`); every ring ownership starts at an even position, so that slot was written by the request's previous step | yes: bounded |
| Writers (window and compressed records, index keys, ring) | slice assignments of what was computed | whole records, pad slots (-1) skipped (`tilelang/cache_writer.py:124`, `tilelang/indexer.py:341`, `tilelang/compress.py:234`) | yes |

The indexer's wider reads touch only bytes whose results are thrown away, so the
declaration in 0502 states the property that matters: no unwritten slot's value reaches
a result. A strict read bound would point those rows at the null block, which stays zero:
one integer select on the page index, values unchanged. It is not needed for the removal,
and is left for the owner.

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
- [vllm/series](vllm/series): candidate 6's series plus 0500-0502. 0491-0499 are other
  drafts'.

### Tests

CPU, macOS (torch 2.14.1, no TileLang or B12X), [runs/cpu-tests.txt](runs/cpu-tests.txt):

- `tests/v1/core`, `tests/v1/worker` and `tests/test_envs.py`: base 1,674 passed, 0500-0501
  1,689, 0500-0502 **1,692**. All three have the same 159 failing or erroring ids:
  offline hub ids, CPU-build modules, no TileLang.
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
- the two drafter cases begin at the hit and at the kept row.

`tests/v1/core/test_single_type_kv_cache_manager.py` joins `config/release-tests.json`'s
gate. `scripts/release_tests.py check` reports 0 problems.

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
| `nofill-tp4` | `rc6/rc-tp4.json` + 0500-0502 (10 files), poison off | its previous owner's bytes: **the end state** |
| `poison-tp4` | `nofill-tp4` + `VLLM_KV_CACHE_POISON=1` | NaN bytes in every format: **the gate** |
| `poison-tp4-dev` | `poison-tp4` + `VLLM_SERVER_DEV_MODE=1` | the release suite's engine grid |
| `*-pressure-tp4` | each of the three | the same, with `--max-model-len 262144` and `--kv-cache-memory-bytes 1342177280` (5,890 blocks; one request needs 4,895) |
| `poison-unfixed-tp4` | `rc6/rc-tp4.json` + 0500 alone, poison on | the positive control |
| `*-tp3` | `rc6/rc-tp3.json` (PR #582, 8,192 batched tokens) | the same; the pressure pool is 805,306,368 bytes (3,534 blocks; one request needs 2,975) |

### Specs, in queue order

1. **[tp4-gate.json](tp4-gate.json)** (run `kv-zeroing-tp4`). Sync, the poison-tests
   bundle on dgx2, then `poison-tp4`, `nofill-tp4` and `zero-tp4`. Each runs:
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

1. 0502 goes into the next candidate's series with 0500 and 0501.
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
| [bundles/poison-tests](bundles/poison-tests) | the kernel-lab GPU tests |
| [bundles/serving](bundles/serving), [bundles/serving-zeroing](bundles/serving-zeroing), [bundles/serving-unfixed](bundles/serving-unfixed) | package files the arms mount (0500-0502; 0500-0501; 0500 alone) |
| `zero-*.json`, `nofill-*.json`, `poison-*.json` | arms |
| `tp4-*.json`, `tp3-*.json` | lab specs |
| [prefix_extension.py](prefix_extension.py), [pool_pressure.py](pool_pressure.py), [kv_counters.py](kv_counters.py) | workloads and counters |
| [tables_arms.py](tables_arms.py), [suite_model.py](suite_model.py) | verdicts |
| [zero_share.py](zero_share.py), [runs/zero-share.txt](runs/zero-share.txt) | the gain bound |
| [runs/cpu-tests.txt](runs/cpu-tests.txt), [runs/run-cpu-tests.sh](runs/run-cpu-tests.sh) | the CPU test runs |
| [vllm/](vllm/) | patches 0500-0502 and the series |

## Not done, and overlaps

- **retire-conditionals** (0495) deletes the Triton `_chunk` that 0501 also edits. Whichever
  lands second gives its torch reference the draft branch's replay bound.
- **The indexer's strict read bound** (null-block page for rows past the extent) is open
  for the owner. Above.
- **ruff** reports two problems that already exist at the base, both outside this change:
  `nvidia/model_state.py`'s import block, and `single_type_kv_cache_manager.py:2680`.
- **Upstream:** the declaration is upstreamable as it stands. vLLM's rule otherwise sees
  only dtypes, not page layouts.
