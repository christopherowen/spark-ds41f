# Decode queries skip RoPE on a TP rank that is dealt no index rows

- **Base commit**: `75fb3e72` (main, `spark-ds41f-1120` current). The measurements were made on two images:
  `spark-ds41f:1120` (`candidate-62`, the release) and the `candidate-69-r2` image built at `945d95d0`, which carried the
  tag `spark-ds41f:1122` before that promotion was reverted. Below, "1120" and "1122" name those two images.
  `vllm/models/deepseek_v4_1/attention.py` is byte-identical in both, and the one diff applies to both.
- **One intended variable**: where the overlapped decode path issues the deferred `_rope(q)` and shared decode metadata in
  `DeepseekV4_1Attention.forward_mqa` ([`decode_rope_empty_index_slice.patch`](decode_rope_empty_index_slice.patch),
  full file in [`overlay/attention.py`](overlay/attention.py)).
- **Site**: eight DGX Sparks on one 200G RoCE switch (the `*-switched` shapes as shipped), kernel `7.0.0-1025-nvidia`
  with 4 KiB pages, base `DeepSeek-V4.1-Flash-1017` weights per node, GLP off, `NCCL_IB_MERGE_NICS=0`,
  `NCCL_CROSS_NIC=0`, port 8002. Nothing else differs from the shapes in `config/cluster.json`.
- **Not a performance experiment.** No release benchmark was run. A small throughput bracket is included only to show
  the fix does not move speed.

## Finding

`tp8-switched` is not batch-invariant on 1120 or 1122. `tp4-switched`, `tp6-switched` and `tp7-switched` are.

1. On layers with an indexer, `_forward_decode_overlapped` (0767) defers `q`'s RoPE and the shared decode metadata to
   `aux_stream()`. They are issued from inside `select_rows`' loop over this rank's index chunks, beside the first
   `_index_chunk`.
2. `_decode_index_split` deals a decode step's index rows across the TP ranks from 24 rows up:
   `SPRows(rows, tp, rank)` gives each rank `ceil(rows / tp)` rows in rank order.
3. When `(tp - 1) * ceil(rows / tp) >= rows` the last rank is dealt nothing. `_index_query` already handles that
   (`if rows > 0`), `select()` never reaches the chunk loop, and the deferred RoPE is never issued.
4. `forward_mqa` then runs sparse MLA on that rank with un-rotated queries: its heads (8 of 64 at TP8), every layer from
   2 up, every row of the step. Rows at position 0 are unaffected because the rotation there is the identity.
5. The same rows in a step of another size get rotated queries on every rank, so a request's output depends on what
   shares its step. The affected step also writes its KV, so the request stays off afterwards.

The overlapped path needs `rows <= DECODE_CHUNK` (64), which bounds the sizes. Whether an empty slice matters depends
on whether that rank owns attention heads. The model has 64 heads in 8 output groups; `config.py` pads the groups to a
multiple of TP and `_trim_padded_groups` drops the all-zero ones, so on worlds that do not divide 8 the trailing ranks
own no heads, and `_rope(q)` is already skipped there (`if self.n_local_heads > 0`):

| TP | Real heads per rank | Row counts (24..64) where the last rank's slice is empty | Affected |
|---:|---|---|---|
| 2, 3, 4 | 32,32 / 24,24,16 / 16 x4 | none | no |
| 5 | 16,16,16,16,0 | none | no |
| 6 | 16,16,16,16,0,0 | 25 | no: rank 5 owns no heads |
| 7 | 16,16,16,16,0,0,0 | 24, 29..30, 36 | no: rank 6 owns no heads |
| 8 | 8 on every rank | 25..28, 33..35, 41..42, 49 | yes |

We first expected TP6 and TP7 to fail at their sizes; they measured clean, and the head layout is why (boot log:
"Padded DeepSeek V4.1 attention for TP7: heads 64 -> 112, output groups 8 -> 14"). TP8 is the only shipped world size
where a rank with an empty slice owns heads.

A step pads up to the next captured graph size, so only sizes a graph was replayed at matter: at TP8 a 34-row step
runs on the 36-row graph, where rank 7 has rows. The sizes seen going wrong at TP8 were 25, 26, 27, 28 and 42.

`_decode_index_split` has a second trigger, an eager decode step past 16,384 tokens of context, which splits from
`rows >= tp`. There smaller worlds would also have an empty last slice (TP3 at 4 rows, TP4 at 5, 6 and 9). It was not
reached here: see "Long context".

## Fix

`issue_pending_aux()` holds the deferred work and runs it once per forward: beside the first index chunk as before, or
right after `select()` when this rank scored nothing. The `wait_stream` join that follows is unchanged. On every step
that was already correct the same kernels are issued on the same streams in the same order.

Unit test: `tests/models/test_deepseek_v4_1_attention.py::test_decode_two_stream_overlap_schedules_disjoint_workspaces`
also runs a rank-split step where this rank is dealt no rows, and one where it is dealt two. In the `spark-ds41f:1122`
image on one GB10 (`pytest tests/models/test_deepseek_v4_1_attention.py --noconftest`; the serving image has no
`tblib`, so `conftest.py` does not import): without the fix 1 failed, 79 passed (`assert [] == [('rope', 'q', 'aux')]`);
with it 80 passed. The patch applies cleanly (`patch --dry-run -p1`) to the prepared 1122 build context
(`.work/build/vllm-19f2c20ed4d6-*/context/vllm-source`).

## Measurements

All probes are in [`probes/`](probes/). They use `/v1/completions` with raw prompts, `temperature 0`, `seed 1`,
128 new tokens and top-5 logprobs unless noted. "Identical" means every token id and every returned logprob equals the
same prompt's idle run on the same boot.

### 1. The symptom, and the fix, from outside

`bi_probe.py --conc K`: 40 greedy prompts, K in flight at once (levels that do not divide 40 send fewer). Prompts whose
tokens match their idle run, of prompts sent:

| K in flight | 1122 as shipped, boot A | 1122 as shipped, boot B | 1122 + fix, boot 1 | 1122 + fix, boot 2 | 1120 as shipped | 1120 + fix |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 40/40 | 38/40 | 40/40 | 40/40 | 40/40 | 40/40 |
| 3 | 31/39 | 32/39 | 39/39 | 39/39 | 31/39 | 39/39 |
| 4 | 9/40 | 17/40 | 40/40 | 40/40 | 13/40 | 40/40 |
| 5 | 4/40 | 4/40 | 40/40 | 40/40 | 5/40 | 40/40 |
| 6 | 30/36 | 15/36 | 36/36 | 36/36 | 13/36 | 36/36 |
| 8 | 22/40 | 16/40 | 40/40 | 40/40 | 30/40 | 40/40 |
| 12 | 29/36 | 32/36 | 36/36 | 36/36 | 26/36 | 36/36 |
| 16 | 20/32 | 19/32 | 32/32 | 32/32 | 21/32 | 32/32 |
| 24 | 21/24 | 20/24 | 24/24 | 24/24 | 18/24 | 24/24 |
| 32 | 28/32 | 32/32 | 32/32 | 32/32 | 28/32 | 32/32 |

- Idle is stable on every boot: two passes agree at all 4,590 positions, and idle output is bit-identical across
  the as-shipped boots, the fixed boots, 1120 and 1122.
- Across the two fixed 1122 boots, every pass (K sweep, 3 / 7 / 15 sampled neighbours, idle repeats) compared
  137,202 logprob positions with idle and none differed.
- `tp4-switched` control (1122, Sparks 1-4): 12 logprob passes (idle, 1 to 31 sampled neighbours, 15 greedy
  neighbours, 8 to 32 in flight), 0 of 4,591 positions differ in any of them.

### 2. The release determinism gate

`scripts/sonar_determinism.py http://<head>:8002 --tokens 192 --long-lines 700 --wide`, then again with
`--no-logprobs`. Cells are `token_diffs` with logprobs / without:

| Scenario | tp4 1122 as shipped | tp8 1122 as shipped, boot A | tp8 1122 as shipped, boot B | tp8 1122 + fix, boot 1 | tp8 1122 + fix, boot 2 | tp8 1120 as shipped | tp8 1120 + fix |
|---|---:|---:|---:|---:|---:|---:|---:|
| `c8` | 0 / 0 | 8 / 5 | 8 / 0 | 0 / 0 | 0 / 0 | 7 / 0 | 0 / 0 |
| `staggered` | 0 / 0 | 6 / 8 | 7 / 7 | 0 / 0 | 0 / 0 | 7 / 7 | 0 / 0 |
| `mixed` | 0 / 0 | 8 / 8 | 8 / 8 | 0 / 0 | 0 / 0 | 8 / 8 | 0 / 0 |
| `cached` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| `again` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| `mixed15` | 0 / 0 | 12 / 12 | 12 / 11 | 0 / 0 | 0 / 0 | 12 / 11 | 0 / 0 |
| `identical` | true / true | false / false | false / false | true / true | true / true | false / false | true / true |

The gate catches this. It had not been run on the wide shapes.

### 3. Which step sizes, measured inside the engine

[`probes/step_trace.patch`](probes/step_trace.patch) (diagnostic, off unless `BI_TRACE_DIR` is set) logs every step on
every rank: the graph descriptor, each request's rows, and a checksum of each row's final hidden state.
`step_sizes.py` sends groups of short token-id prompts sized to total 18..208 rows, three times each, so the engine
runs steps of many sizes; `graph_verdict.py` compares each later row of each request with the same token prefix
computed alone, grouped by the captured graph the step ran on.

| Build | Shape | Empty-slice rows where that rank owns heads | As shipped: wrong graph sizes (rows wrong / checked there) | As shipped: other sizes (rows wrong / checked) | Fixed (rows wrong / checked) |
|---|---|---|---|---|---|
| 1122 | `tp8-switched` | 25..28, 33..35, 41..42, 49 | 25..28, 42 (1,611 / 1,611) | 39 sizes (0 / 57,445) | 0 / 58,288 |
| 1122 | `tp7-switched` | none | none | 44 sizes (0 / 58,530) | 0 / 58,047 |
| 1122 | `tp6-switched` | none | none | 45 sizes (0 / 57,908) | 0 / 58,490 |
| 1120 | `tp8-switched` | 25..28, 33..35, 41..42, 49 | 25..28, 42 (1,390 / 1,390) | 38 sizes (0 / 56,570) | 0 / 57,667 |
| 1120 | `tp7-switched` | none | none | 44 sizes (0 / 58,165) | sweep cut short by a site network drop; idle, K=4/5/8 and long-context passes all identical |
| 1120 | `tp6-switched` | none | not run | not run | sweep cut short by a site network drop; idle, K=4/5/8 and long-context passes all identical |

No graph size was mixed: a size is either wrong on every checked row or right on every one. The API-level view of the
same sweep is noisier (as shipped, prompts differ at totals well above 28) because one request's prompts do not always
land in one step; the trace is the ground truth.

### 4. Where the difference starts

[`probes/stage_trace.patch`](probes/stage_trace.patch) adds per-layer checksums (layer input, `q`, the index and window
tables, sparse-MLA output, attention output, FFN output) on every rank. Two 13-token prompts sharing a 26- or 28-row
step, against the same prompt alone (`results/tp8-1122-t4/stage-first.txt`):

- layers 0 and 1: identical on all eight ranks;
- ranks 0..6: the first differing stage is layer 2's attention output, after the all-reduce;
- rank 7: the first differing stage is layer 2's `q`, at every position except 0;
- a 29-row step: no stage differs on any rank.

Hidden-state dumps from an earlier boot put the wrong rows 38.1% to 85.5% away from the right ones (relative L2 of
the final hidden state), which is what ruled out a rounding-order cause.

### 5. Long context

`long_probe.py`: three 20,000-token prompts, each alone (cold, then prefix-cached), then all three in flight twice,
then alone again. Two `tp4-switched` pods ran at once on 1122, one as shipped and one fixed.

- As shipped and fixed outputs are bit-identical in every phase, and concurrent equals solo on both pods.
- The trace on the as-shipped pod shows all 313 decode steps at 16,384 tokens or deeper replayed a captured graph
  (rows seen: 2, 4, 5, 6, 8, 9), so the eager long-context split was never taken;
  290 of those steps were at row counts that would have an empty last slice if it were.
- Not tested: contexts or shapes where decode runs eager (for example rows above the capture ceiling), and TP3.

### 6. Speed bracket

`quick_bench.py`: C greedy streams of 256 new tokens, three passes, aggregate tokens per second. Median (all three):

| Streams | 1122 as shipped | 1122 + fix, boot 1 | 1122 + fix, boot 2 | 1120 as shipped | 1120 + fix |
|---:|---:|---:|---:|---:|---:|
| 1 | 141.2 (392.7, 141.2, 136.3) | 140.9 (393.5, 140.9, 137.4) | 141.1 (391.4, 141.1, 136.3) | 138.5 (391.2, 138.5, 134.4) | 138.4 (389.4, 138.4, 134.0) |
| 5 | 270.5 (291.5, 270.5, 248.6) | 277.2 (301.1, 277.2, 250.1) | 273.4 (296.9, 273.4, 249.4) | 247.5 (289.2, 247.5, 246.6) | 273.2 (297.5, 273.2, 245.7) |
| 16 | 358.3 (401.7, 356.4, 358.3) | 366.4 (395.8, 366.4, 356.2) | 361.9 (398.6, 361.9, 347.1) | 370.2 (395.9, 370.2, 348.7) | 363.7 (387.1, 363.7, 353.6) |
| 32 | 452.7 (452.7, 451.7, 458.0) | 455.3 (455.3, 444.5, 456.4) | 454.7 (454.7, 446.9, 457.6) | 452.4 (453.8, 447.3, 452.4) | 452.7 (452.7, 446.3, 456.0) |

The first C1 pass repeats the warm-up prompt and is not comparable with the other two. This is three passes per level
on one night, enough to see that nothing large moved and no more.

## Every boot

| # | Arm | Build | Shape | Change from the shape as shipped | Started (CST) | Ready s | Min MemAvailable serving | Result |
|---:|---|---|---|---|---|---:|---:|---|
| 1 | `tp4B-1122` | 1122 | `tp4-switched` | as shipped (Sparks 1-4) | 2026-11-23 20:27:41 | 103 | 17.2 GiB | control: 12 logprob passes against idle, 0 positions differ; his gate passes |
| 2 | `tp8-1122` | 1122 | `tp8-switched` | as shipped | 2026-11-23 21:28:59 | 119 | 12.2 GiB | reproduces; his gate fails |
| 3 | `tp8-1120` | 1120 | `tp8-switched` | as shipped | 2026-11-23 22:05:19 | 165 | 14.8 GiB | fails the same way: not new in 1122 |
| 4 | `tp8-1122-engdisk` | 1122 | `tp8-switched` | Engram tables on disk | 2026-11-23 22:19:59 | 150 | 35.9 GiB | fails the same way: not the device Engram tables |
| 5 | `tp8-1122-s1120` | 1122 | `tp8-switched` | 8,192 batched tokens, long-prefill threshold 3,856, late-admission layer 20 | 2026-11-23 22:29:02 | 119 | 18.0 GiB | fails the same way: not the batch settings |
| 6 | `tp8-1122-b2` | 1122 | `tp8-switched` | as shipped, second boot | 2026-11-23 22:37:37 | 118 | 12.3 GiB | idle bit-identical to the first boot; in-flight sweep K=2..32 |
| 7 | `tp8-1122-nospec` | 1122 | `tp8-switched` | speculative decoding off | 2026-11-23 22:44:35 | 135 | 18.4 GiB | clean at K=2..24; K=32 breaks as streams finish and the step passes 28..25 rows |
| 8 | `tp8-1122-trace` | 1122 | `tp8-switched` | diagnostic: first step trace | 2026-11-23 23:02:39 | 119 | 12.3 GiB | first step trace: every bad row sits on the 25..28 or 42 graph |
| 9 | `tp8-1122-t2dump` | 1122 | `tp8-switched` | diagnostic: step trace + hidden-state dumps + route log | 2026-11-23 23:13:27 | 150 | 12.3 GiB | hidden-state dumps: wrong rows are percent-level away, not last-bit (section 4) |
| 10 | `tp8-1122-t3priv` | 1122 | `tp8-switched` | diagnostic: sparse-MLA decode route with private scratch | 2026-11-23 23:26:26 | 149 | 12.1 GiB | sparse-MLA decode route with private scratch: still fails |
| 11 | `tp8-1122-t2ext` | 1122 | `tp8-switched` | diagnostic: sparse MLA forced to the extend kernel | 2026-11-23 23:31:16 | 133 | 11.8 GiB | sparse MLA forced to the extend kernel: still fails |
| 12 | `tp8-1122-t4` | 1122 | `tp8-switched` | as shipped + per-stage trace | 2026-11-23 23:37:17 | 151 | 7.9 GiB | per-stage trace: first difference is layer 2, rank 7, q; then the graph-size sweep |
| 13 | `tp8-1122-fix1` | 1122 + fix | `tp8-switched` | one file mounted: attention.py with the fix | 2026-11-23 23:45:46 | 134 | 12.2 GiB | the fix, boot 1 (overlay comment differs from the final file; same AST) |
| 14 | `tp8-1122-base` | 1122 | `tp8-switched` | as shipped (proof-set baseline) | 2026-11-24 00:07:34 | 134 | 12.4 GiB | as shipped, same proof set as the fixed boots |
| 15 | `tp8-1122-fix1-b2` | 1122 + fix | `tp8-switched` | the fix, second boot | 2026-11-24 00:29:26 | 136 | 12.4 GiB | the fix, boot 2 (same overlay as boot 1) |
| 16 | `tp4A-1122-trace` | 1122 | `tp4-switched` | as shipped + step trace (Sparks 5-8) | 2026-11-24 00:51:38 | 89 | 17.2 GiB | long-context check, as shipped |
| 17 | `tp4B-1122-fix1-trace` | 1122 + fix | `tp4-switched` | the fix + step trace (Sparks 1-4) | 2026-11-24 00:51:39 | 120 | 17.0 GiB | long-context check, fixed (step trace not collected: wrong file pattern) |
| 18 | `tp8-1122-trace1` | 1122 | `tp8-switched` | as shipped + step trace | 2026-11-24 00:57:09 | 120 | 12.3 GiB | wrong graph sizes: 25..28, 42; 1,717 of 58,212 rows wrong; K sweep 4:7/40, 5:7/40, 8:22/40; long-context concurrent vs solo 3/3/3/3 of 3 |
| 19 | `tp8-1122-fix1-trace` | 1122 + fix | `tp8-switched` | the fix + step trace | 2026-11-24 01:03:18 | 153 | 12.3 GiB | wrong graph sizes: none; 0 of 58,288 rows wrong; K sweep 4:40/40, 5:40/40, 8:40/40; long-context concurrent vs solo 3/3/3/3 of 3 |
| 20 | `tp7-1122-trace` | 1122 | `tp7-switched` | as shipped + step trace | 2026-11-24 01:11:28 | 398 | 8.4 GiB | wrong graph sizes: none; 0 of 58,530 rows wrong; K sweep 4:40/40, 5:40/40, 8:40/40; long-context concurrent vs solo 3/3/3/3 of 3 |
| 21 | `tp7-1122-fix1-trace` | 1122 + fix | `tp7-switched` | the fix + step trace | 2026-11-24 01:22:49 | 132 | 8.5 GiB | wrong graph sizes: none; 0 of 58,047 rows wrong; K sweep 4:40/40, 5:40/40, 8:40/40; long-context concurrent vs solo 3/3/3/3 of 3 |
| 22 | `tp6-1122-trace` | 1122 | `tp6-switched` | as shipped + step trace | 2026-11-24 01:29:48 | 283 | 9.1 GiB | wrong graph sizes: none; 0 of 57,908 rows wrong; K sweep 4:40/40, 5:40/40, 8:40/40; long-context concurrent vs solo 3/3/3/3 of 3 |
| 23 | `tp6-1122-fix1-trace` | 1122 + fix | `tp6-switched` | the fix + step trace | 2026-11-24 01:39:09 | 112 | 9.3 GiB | wrong graph sizes: none; 0 of 58,490 rows wrong; K sweep 4:40/40, 5:40/40, 8:40/40; long-context concurrent vs solo 3/3/3/3 of 3 |
| 24 | `tp8-1120-base` | 1120 | `tp8-switched` | as shipped (proof-set baseline) | 2026-11-24 01:45:32 | 183 | 14.9 GiB | K sweep 2:40/40, 3:31/39, 4:13/40, 5:5/40, 6:13/36, 8:30/40, 12:26/36, 16:21/32, 24:18/24, 32:28/32; his gate identical=false; long-context concurrent vs solo 3/3/3/3 of 3 |
| 25 | `tp8-1120-fix1` | 1120 + fix | `tp8-switched` | one file mounted: attention.py with the fix | 2026-11-24 02:08:48 | 182 | 14.8 GiB | K sweep 2:40/40, 3:39/39, 4:40/40, 5:40/40, 6:36/36, 8:40/40, 12:36/36, 16:32/32, 24:24/24, 32:32/32; his gate identical=true; long-context concurrent vs solo 3/3/3/3 of 3 |
| 26 | `tp8-1120-trace` | 1120 | `tp8-switched` | as shipped + step trace | 2026-11-24 02:31:55 | 183 | 14.8 GiB | wrong graph sizes: 25..28, 42; 1,390 of 57,960 rows wrong; K sweep 4:11/40, 5:5/40, 8:12/40; long-context concurrent vs solo 3/3/3/3 of 3 |
| 27 | `tp8-1120-fix1-trace` | 1120 + fix | `tp8-switched` | the fix + step trace | 2026-11-24 02:39:06 | 167 | 14.8 GiB | wrong graph sizes: none; 0 of 57,667 rows wrong; K sweep 4:40/40, 5:40/40, 8:40/40; long-context concurrent vs solo 3/3/3/3 of 3 |
| 28 | `tp8-1120-t4` | 1120 | `tp8-switched` | as shipped + per-stage trace | 2026-11-24 02:47:00 | 183 | 10.5 GiB | see results |
| 29 | `tp7-1120-trace` | 1120 | `tp7-switched` | as shipped + step trace | 2026-11-24 02:50:29 | 271 | 11.0 GiB | wrong graph sizes: none; 0 of 58,165 rows wrong; K sweep 4:40/40, 5:40/40, 8:40/40; long-context concurrent vs solo 3/3/3/3 of 3 |
| 30 | `tp7-1120-fix1-trace` | 1120 + fix | `tp7-switched` | the fix + step trace | 2026-11-24 02:59:40 | 177 | 11.1 GiB | K sweep 4:40/40, 5:40/40, 8:40/40; long-context concurrent vs solo 3/3/3/3 of 3 |
| 31 | `tp6-1120-trace` | 1120 | `tp6-switched` | as shipped + step trace |  |  |  | see results |
| 32 | `tp6-1120-fix1-trace` | 1120 + fix | `tp6-switched` | the fix + step trace |  |  |  | see results |
| 33 | `tp4A-1120-trace` | 1120 | `tp4-switched` | as shipped + step trace (Sparks 5-8) | 2026-11-24 03:12:49 | 136 | 18.6 GiB | K sweep 5:40/40; long-context concurrent vs solo 3/3/3/3 of 3 |
| 34 | `tp4B-1120-fix1-trace` | 1120 + fix | `tp4-switched` | the fix + step trace (Sparks 1-4) | 2026-11-24 03:12:49 | 150 | 18.6 GiB | K sweep 5:40/40; long-context concurrent vs solo 3/3/3/3 of 3 |
| 35 | `tp4A-1120-ctl` | 1120 | `tp4-switched` | as shipped (control, Sparks 5-8) | 2026-11-24 03:18:47 | 136 | 18.6 GiB | K sweep 2:40/40, 3:39/39, 4:40/40, 5:40/40, 6:36/36, 8:40/40, 12:36/36, 16:32/32, 24:24/24, 32:32/32; his gate identical=true |

`MemAvailable` is the minimum across the arm's nodes once serving. Every arm that started booted first time. A site
network drop at 03:11 (the driving host lost its route to the Sparks for about a minute) cut the `tp7-1120-fix1-trace`
step sweep short and stopped both planned `tp6-switched` boots on 1120 from starting; they were not rerun. No result was
discarded.

## What was ruled out on the way

- A change in 1122 (0830..0857): 1120 fails identically, and the 1122 Python tree minus those 28 patches equals 1120's.
- The 32K scheduling settings, device-resident Engram tables, the sparse-MLA kernel route by row count, and workspace
  sharing between kernels: one boot each, all still failing.
- Speculative decoding as a cause: with it off, K=2..24 is clean and K=32 breaks once streams finish, which is
  consistent with one row per stream passing through 28..25 rows. Graph sizes were not traced on that boot.
- Our own first reading was wrong twice and is recorded in the run log: "needs sampled neighbours" (fifteen steady
  greedy neighbours sit above the affected sizes) and "0835's TP8 tile branch" (1120 has no 0835).

## Reproduce

```
# as shipped, with the step trace (diagnostic patch mounted over the image's file)
bin/spark --cluster-config <tp8-switched profile with BI_TRACE_DIR=/cache/bi-trace> cluster start --apply
python3 probes/bi_probe.py   http://<head>:8002 idle.json --logprobs 5
python3 probes/bi_probe.py   http://<head>:8002 k5.json   --logprobs 5 --conc 5
python3 probes/bi_compare.py idle.json k5.json
python3 probes/step_sizes.py http://<head>:8002 step-sizes.json --totals 18-208 --reps 3
python3 probes/graph_verdict.py <cache>/bi-trace/<head>-rank0-*.jsonl 8
python3 scripts/sonar_determinism.py http://<head>:8002 --tokens 192 --long-lines 700 --wide

# fixed: the same profile with overlay/attention.py mounted read-only over
# /opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/attention.py
```

[`tp8-switched-fix.json`](tp8-switched-fix.json) is that profile for `bin/spark` (it renders here; our boots used the
rendered `docker run` line with the extra `--volume`, started rank by rank from a site script, not `cluster start`).

## Limits

- Output quality on an affected step was not measured, only that the output changes.
- `tp5-switched`, the TP3 shapes and `tp2-*` were not booted, and `tp6-switched` was run on 1122 only. The row-count
  and head-layout argument says none of them has an affected size.
- The overlay file on boots `tp8-1122-fix1` and `tp8-1122-fix1-b2` carried an earlier wording of one comment. Its AST
  equals the final file's (`ast.dump` compared). Every later fixed boot ran the final file.
- Raw probe outputs (about 150 MB of JSON with logprobs) and the step traces are not in this folder;
  [`results/facts.json`](results/facts.json) is computed from them and they are available on request.

## Conclusion

A rank that is dealt no index rows in a rank-split decode step never rotates its queries. At TP8, the one
world size where that rank also owns attention heads, this makes 1120 and 1122 batch-variant at a handful of step sizes
that a few concurrent streams pass through constantly. Issuing the deferred work once per forward fixes it: the gate passes on two boots, every
probe matches idle, and speed does not move.
