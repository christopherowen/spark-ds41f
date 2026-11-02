# DSpark Markov greedy fusion: call the fused kernels (#771, #831 reworked)

## Background

#771 and #831 added a fused DSpark Markov-bias projection that adds the base
logits and reduces each vocabulary tile in its epilogue
(`packed_vocab_markov_greedy_partials`, `bf16_vocab_markov_greedy_partials`,
`finish_vocab_greedy_pairs`, `project_head_markov_greedy_pairs`). Candidate-37
carries them in `0671-perf-dsv41-12-bit-packed-BF16-Markov-W2-with-fragmen.patch`
with their kernel test, but no call site: 0671 changes nothing in the speculator.
Kernel preparation compiles them (for the packed Markov head and every BF16 head),
and the vocabulary-parallel
greedy draft loop still runs 0012's materialized path, which 0673 extended with
top-2 (`DSparkSpeculator._sample_sequential`: `markov_bias_local` writes the
`[requests, shard]` bias, then `greedy_markov_shard_pair` reduces base + bias, at
every draft position).

## Variable

One: the vocabulary-parallel greedy draft steps (TP > 1) reduce base + bias in the
Markov projection's epilogue instead of materializing the bias. Patch
[`0677-perf-dspark-reduce-vocabulary-parallel-greedy-drafts.patch`](0677-perf-dspark-reduce-vocabulary-parallel-greedy-drafts.patch),
on candidate-37's vLLM patch head.

- `_sample_sequential` reduces each draft position through
  `DSparkDeepseekV4ForCausalLM.markov_greedy_pairs_local`, which calls
  `project_head_markov_greedy_pairs` into the speculator's shard pairs.
  `markov_bias_local` and `draft_logits_local_layout`, used only by the
  materialized path, are removed.
- No fallback (TODO, "No fallbacks"). `register_warmups` declares only the
  selections and raises at kernel preparation when the Markov head does not fuse.
  `fuses_markov_greedy` requires a TileLang packed or BF16 head with no logit scale
  or soft cap; an NVFP4 Markov head, `VLLM_DS41_MARKOV_NVFP4` (not in the recipe),
  has no fused projection. `greedy_markov_shard_pair` stays in `greedy.py` as the
  reference the tests compare against.
- Top-2 (0673): `greedy_markov_shard_pair(top2=True)` keeps one partial per
  2048-column part, so the runner-up it reports is the best maximum of the row's
  *other* parts. `finish_vocab_greedy_pairs(top2=True)` reproduces that from the
  tile partials (the tiles outside the pair's part) to the bit. Each Markov head
  declares both finishes.
- Only Markov heads declare the fused kernels: `prepare_bf16_head` no longer
  compiles them for every BF16 head.

Promoted shapes run the packed (12-bit) Markov W2 with FP32 logits
(`VLLM_DS41_HEAD_DTYPE=float32`), top-2 off (no `dspark_top2_branch` in the model
or recipe), 5 drafts, at most 16 requests (`tp4-ring`) or 8 (`tp3-triangle`).

`#771`'s `TILE_CONFIGS[64]` change (`step_groups=2`) is evaluated separately below
and is not in the patch.

## Identity

| | |
|---|---|
| Deployment base commit | `15cba190` |
| vLLM base | candidate-37 series on `19f2c20ed`: patch head `5b7cfa37`, tree `af0fb152` (reproduced) |
| vLLM branch | `perf/dspark-fused-markov-greedy` (dgx4 `~/src/vllm-markov-greedy-fusion`), commit `3b7884972` |
| Series + 0677, fresh tree | patch head `5d70ccb1`, tree `a359ee28` (equals the branch's tree), 562 patches |
| GPU | sp8 (`spark-0016`, GB10), image `spark-ds41f:candidate-37` (`e1ee7b1953b1`, labels vLLM tree `af0fb152`, TileLang tree `222b895d`), this patch's files mounted over the image's source |
| Weights | DS4.1-Flash snapshot `dba1be0a`: `mtp.2.markov_head.head.weight` (W2, sha256 `0c0e1c69...`), `mtp.2.markov_head.embed.weight` (W1, `104a00ec...`), `head.weight` rows 0..43135 (`b375a694...`), packed as serving packs them (2,107 / 2,973 exceptions in the TP4 / TP3 Markov shard) |

## Commands

```sh
# Tree (dgx4): candidate-37's series as `bin/spark build prepare` applies it, then the branch.
scripts/prepare_tree.sh
# Real weights (dgx4, CPU), copied to sp8:/home/chris/src/markov-fusion/bench/weights.
python3 -I scripts/extract_rows.py $SNAPSHOT mtp.2.markov_head.head.weight 0 markov_w2
python3 -I scripts/extract_rows.py $SNAPSHOT mtp.2.markov_head.embed.weight 0 markov_w1
python3 -I scripts/extract_rows.py $SNAPSHOT head.weight 43136 lm_head_rows
# Everything below: scripts/run_overlay.sh "<command>" on sp8 (tree/ = the patch's files,
# /bench = scripts/ and weights/).
python3 -m pytest -q -o addopts="" --noconftest -p tests.kernels.deepseek_v41_pdl_sources \
  tests/kernels/quantization/test_deepseek_v41_tilelang_vocab.py \
  tests/v1/worker/test_dspark_greedy_markov.py tests/v1/spec_decode/test_dspark_vocab_parallel.py \
  tests/v1/worker/gpu/test_greedy_verification.py tests/v1/worker/test_prepare_kernels.py \
  tests/models/test_deepseek_v4_1_sp_prefill.py
python3 /bench/bench_markov_greedy.py --dtype float32 --out /bench/out/markov-fp32-boot{1,2}.json
python3 /bench/profile_markov_kernels.py --out /bench/out/markov-kernels-fp32.json
python3 /bench/bench_tile64_step_groups.py --out /bench/out/tile64-boot{1,2}.json
python3 /bench/soak_markov_greedy.py --dtype float32 --out /bench/out/soak-fp32.json
python3 /bench/soak_markov_greedy.py --dtype float32 --top2 --out /bench/out/soak-fp32-top2.json
python3 /bench/soak_markov_greedy.py --dtype bfloat16 --out /bench/out/soak-bf16.json
```

Each `boot` is a fresh container (new process, cold TileLang load, new allocations).

## Correctness

- Kernel tests (`test_deepseek_v41_tilelang_vocab.py`): packed and BF16 Markov
  heads, TP4 (32,320) and TP3 (43,136; last rank 43,008 real rows) shards and a
  partial-tile shard, BF16 and FP32 logits, top-2 on and off, 1..64 rows; ties in a
  part, across parts and through duplicate weight rows, NaNs, -inf rows. All four
  pair columns are compared as bits against `greedy_markov_shard_pair` on the
  materialized bias. Only Markov heads declare the kernels.
- Speculator tests (`test_dspark_greedy_markov.py`): `_sample_sequential`'s
  five-position loop (anchor embedding, shard pairs, `select_first_argmax` with the
  next embedding and top-2 promotion) through the fused projection draws the same
  tokens, runner-ups, margins and Markov embeddings, as bits, as the materialized
  bias plus `greedy_markov_shard_pair`. Cases: TP4 and TP3 last-rank shards, BF16
  and FP32, top-2 on and off, 1..48 requests. Kernel preparation raises for a
  Markov head that does not fuse.
- Results: baseline (unchanged tree, dgx4) 59 passed. With 0677 (sp8, with the PDL
  sources plugin), the vocab kernel, DSpark greedy, vocabulary-parallel, greedy
  verification, kernel preparation and SP prefill tests give 234 passed; 230
  kernels launched, 62 with PDL, 0 violations. Every benchmark case also compared
  both arms' chains as bits: all equal.
- Fresh tree: the series plus 0677 applies to `19f2c20ed`; `git diff --check` is clean.
- Revision: the first revision (`32686af7e`: 161 passed; PDL check 135 passed, 0
  violations) kept the materialized path for heads that do not fuse. The final
  revision removes that path and raises instead. The kernels, and so every
  benchmark below (which calls them directly), are the same in both.

### Functional soak under serving's workspace discipline

The one thing the fused path changes outside the kernels is where its scratch
lives. The materialized path reduced into the speculator's own buffers. The fused
path borrows its tile partials (`rows × tiles × 8` bytes) from the workspace
manager, and in serving that is the DSpark draft lane's slot: the V2 runner
captures and runs `propose` under `use_workspace_lane(_draft_workspace_lane)`
(lane 1). The kernel tests above take the persistent-scratch fallback instead,
so they don't exercise this.

[`scripts/soak_markov_greedy.py`](scripts/soak_markov_greedy.py) reproduces that
discipline on one GB10 (sp8) and checks every row, not timing:

- a workspace manager with a draft lane, which the drafter's other borrowers size
  first (4 MB);
- every captured size (1, 2, 4, 8, 16, 24, 32, 48 requests) warmed eagerly and
  captured under that lane, then the workspace locked, as after the runner's last
  capture;
- per step, a random captured size and 1..size active rows, the rest padding.
  Active rows get fresh base logits (some with exact ties, a NaN or a −inf row)
  and anchor ids; padding rows get NaN, +inf, −inf, 3e38 or 0 and garbage ids.
  The draft lane's whole slot is filled with 0xFF before each arm's replay.

Results, 3,000 steps per run over 16 shape × size cells (TP4, TP3; 159–201 steps
each):

| run | mismatching steps | tokens outside the shard | non-finite pair rows (padding) | negative control |
|---|---:|---:|---:|---|
| FP32, top-2 off (serving) | 0 | 0 | 16,006 | flagged |
| FP32, top-2 on | 0 | 0 | 16,006 | flagged |
| BF16, top-2 off | 0 | 0 | 16,006 | flagged |

A mismatching step is any bit of any output differing on any row, padding
included. The slot never grew past its 4 MB after locking
([`runs/soak-fp32.json`](runs/soak-fp32.json),
[`runs/soak-fp32-top2.json`](runs/soak-fp32-top2.json),
[`runs/soak-bf16.json`](runs/soak-bf16.json)). The drafts are the same bits as
before for every row, padding rows included, so the target's verification rows,
their count and their padding are those of candidate-37.

Not covered: a serving soak with real concurrent traffic and mixed prompt lengths
where the token budget binds (the lesson of #842, whose arm crashed in the MoE
router only under production traffic). That needs a cluster window, and it has not
been run.

### Serving soak: planned, not run

The arm is [`tp3-soak-dev.json`](tp3-soak-dev.json): candidate-37's dev arm at
`tp3-triangle` with 0677's three runtime files mounted over the image's
candidate-37 copies ([`overlay/vllm`](overlay/vllm), from vLLM `3b7884972`). That is
0677's runtime tree on the image digest every rank already holds (`c4c59de1` on
dgx1..dgx3). `deployment.branch` is unset so the unmerged experiment commit
(`6c221d47`) deploys.

The workload is servebench's `release-tp3`, about 70 minutes:

- 8 and 4 streams in closed loops, and Poisson arrivals;
- a cold 131K prompt prefilling under 3, 6 and 7 decode streams, so the token budget
  binds and rows are padded;
- prefill from 32K to 1M;
- a failed request fails its step.

On 2026-11-02 the cluster was idle and served candidate-36
(`candidate-36/tp3-dev.json`), with no hold or request. The window open was refused
by this session's permission system before anything changed. On dgx1, in the
deployment checkout:

```sh
scripts/lab.py window open --production-config experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json --note "PR 854 serving soak"
git fetch -q origin && git checkout -q --detach 6c221d47
setsid -f python3 experiments/2026-11-02-dspark-markov-greedy-fusion/scripts/hold_heartbeat.py "PR 854 soak" > ~/spark-heartbeat.log 2>&1 < /dev/null
bin/spark --cluster-config experiments/2026-11-02-dspark-markov-greedy-fusion/tp3-soak-dev.json cluster sync --apply
bin/spark --cluster-config experiments/2026-11-02-dspark-markov-greedy-fusion/tp3-soak-dev.json cluster start --replace --apply
bin/spark --cluster-config experiments/2026-11-02-dspark-markov-greedy-fusion/tp3-soak-dev.json bench --suite release-tp3 --label 0677-soak --compare none --output results/private/suites/pr854-soak/0677
# then restore what served
git checkout -q --detach origin/main
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json cluster sync --revision origin/main --apply
touch ~/spark-heartbeat.stop
scripts/lab.py window close
```

Pass: every step completes with no failed request, and no engine error
(`gate logits contain nan or inf`, device asserts) in any rank's log.

## Draft-step microbenchmark (sp8, FP32 logits, top-2 off)

`bench_markov_greedy.py` times one rank's five draft positions, CUDA-graph
captured:

- `chain`: the anchor embedding, then per position the reduction, a copy into a
  rank slot of the gathered pairs (the all-gather's stand-in) and
  `select_first_argmax` with the next embedding;
- `reduce`: the five reductions alone.

`cold` writes 256 MB before each timed replay, so the first position reads the
Markov shard from DRAM, as after the drafter's forward. `hot` replays back to back.
Values are the median of 5 rounds × 200 replays in µs per five-position loop, and
Δ = fused − materialized. Brackets give the range of the per-round paired Δ. Arms
alternate within a round and swap order every round.

### `chain` / cold (the closest to serving)

| shape | M | boot 1 materialized → fused | Δ [range] | boot 2 materialized → fused | Δ [range] |
|---|---:|---|---|---|---|
| tp4 | 1 | 220.1 → 219.4 | −0.7 [−0.9, −0.3] | 221.2 → 221.4 | +0.3 [−1.8, +2.0] |
| tp4 | 2 | 223.9 → 223.2 | −0.8 [−2.0, −0.6] | 224.3 → 224.1 | −0.2 [−2.1, −0.0] |
| tp4 | 3 | 228.3 → 225.2 | −3.1 [−4.3, −3.1] | 231.0 → 228.1 | −2.9 [−5.1, −2.1] |
| tp4 | 4 | 232.4 → 227.1 | −5.3 [−5.8, −5.1] | 235.5 → 229.3 | −6.2 [−6.4, −5.8] |
| tp4 | 6 | 240.6 → 233.9 | −6.6 [−7.1, −6.2] | 243.9 → 237.5 | −6.4 [−7.3, −6.3] |
| tp4 | 8 | 248.8 → 239.6 | −9.2 [−9.3, −8.3] | 251.8 → 242.6 | −9.2 [−9.4, −8.4] |
| tp4 | 12 | 268.2 → 255.6 | −12.6 [−12.7, −12.2] | 272.3 → 258.0 | −14.3 [−15.2, −13.8] |
| tp4 | 16 | 284.8 → 270.3 | −14.6 [−15.3, −14.4] | 289.4 → 273.1 | −16.4 [−17.2, −15.3] |
| tp4 | 24 | 348.0 → 325.1 | −22.9 [−23.2, −21.4] | 347.1 → 327.9 | −19.2 [−19.5, −18.8] |
| tp4 | 32 | 381.2 → 353.3 | −28.0 [−28.5, −26.6] | 386.0 → 356.6 | −29.4 [−30.5, −27.7] |
| tp4 | 48 | 549.1 → 499.6 | −49.5 [−50.9, −48.7] | 545.6 → 505.8 | −39.8 [−40.9, −37.7] |
| tp3 | 1 | 253.9 → 257.0 | +3.0 [+2.7, +3.6] | 243.6 → 245.7 | +2.1 [+2.0, +3.0] |
| tp3 | 2 | 259.0 → 258.0 | −1.0 [−1.9, +1.0] | 260.9 → 261.2 | +0.3 [−0.8, +1.8] |
| tp3 | 3 | 265.7 → 263.1 | −2.6 [−3.6, −2.2] | 269.2 → 267.2 | −2.0 [−2.8, −0.2] |
| tp3 | 4 | 271.3 → 266.2 | −5.1 [−7.1, −4.2] | 274.1 → 270.3 | −3.8 [−4.1, −3.1] |
| tp3 | 6 | 282.9 → 276.4 | −6.5 [−7.2, −6.1] | 286.1 → 280.0 | −6.1 [−7.9, −5.1] |
| tp3 | 8 | 295.9 → 283.5 | −12.4 [−13.2, −10.8] | 298.4 → 285.6 | −12.7 [−13.6, −11.4] |
| tp3 | 12 | 321.8 → 305.8 | −16.0 [−16.7, −15.6] | 326.2 → 308.2 | −18.0 [−18.4, −16.4] |
| tp3 | 16 | 346.0 → 324.5 | −21.5 [−22.6, −20.5] | 350.1 → 327.3 | −22.8 [−23.4, −22.5] |
| tp3 | 24 | 427.9 → 384.2 | −43.7 [−44.0, −42.6] | 437.2 → 389.0 | −48.2 [−51.0, −47.1] |
| tp3 | 32 | 535.5 → 431.8 | −103.7 [−104.5, −103.3] | 537.1 → 447.8 | −89.3 [−92.7, −88.7] |
| tp3 | 48 | 775.1 → 622.5 | −152.6 [−154.6, −151.6] | 757.7 → 641.0 | −116.7 [−116.7, −114.7] |

### `chain` / hot

| shape | M | boot 1 Δ [range] | boot 2 Δ [range] | boot 1 materialized |
|---|---:|---|---|---:|
| tp4 | 1 | +3.3 [+2.5, +3.4] | +3.9 [+2.4, +4.3] | 110.7 |
| tp4 | 4 | +3.0 [+1.5, +4.0] | +2.6 [+2.5, +2.8] | 116.6 |
| tp4 | 8 | +4.5 [+3.6, +5.2] | +4.6 [+4.4, +4.6] | 123.7 |
| tp4 | 12 | +4.7 [+4.5, +5.2] | +5.6 [+2.5, +6.3] | 135.7 |
| tp4 | 16 | −0.7 [−0.7, +0.3] | +2.1 [+2.1, +2.1] | 155.7 |
| tp4 | 24 | −26.0 [−26.2, −24.7] | −16.6 [−17.6, −15.9] | 246.9 |
| tp4 | 48 | −55.1 [−56.2, −54.9] | −51.1 [−51.2, −50.1] | 459.9 |
| tp3 | 1 | +6.6 [+6.3, +7.3] | +6.6 [+6.4, +6.8] | 121.7 |
| tp3 | 4 | +6.4 [+4.9, +7.4] | +1.9 [+1.6, +2.8] | 129.3 |
| tp3 | 6 | +8.3 [+8.0, +8.5] | +8.0 [+6.6, +8.0] | 138.5 |
| tp3 | 8 | +2.0 [+0.5, +2.2] | +0.9 [+0.0, +1.8] | 152.8 |
| tp3 | 12 | −16.7 [−17.6, −16.4] | −19.5 [−20.5, −19.5] | 192.0 |
| tp3 | 48 | −206.9 [−209.5, −206.4] | −160.8 [−161.5, −159.8] | 727.3 |

`reduce` behaves the same way: cold Δ runs from −3.1/−3.9 µs (tp4, M=1) and
−0.7/+0.0 (tp3, M=1) to −38.7/−40.1 (tp4, M=48) and −155.6/−130.0 (tp3, M=48).
Hot Δ is +2 to +7 µs below 16 (tp4) or 8 (tp3) rows and falls to −44/−47 (tp4, 48)
and −209/−173 (tp3, 48). Every row count is in
[`runs/markov-fp32-boot1.json`](runs/markov-fp32-boot1.json),
[`runs/markov-fp32-boot2.json`](runs/markov-fp32-boot2.json) and
[`runs/markov-fp32-boots.txt`](runs/markov-fp32-boots.txt).

### Where the time goes (CUPTI, `reduce` hot, µs)

| shape, M | materialized: projection + partial argmax + finish (×5) | loop span | fused: partials + finish (×5) | loop span |
|---|---|---:|---|---:|
| tp4, 1 | 16.9 + 1.5 + 0.9 | 98.9 | 19.5 + 7.1 | 99.8 |
| tp4, 16 | 21.0 + 3.9 + 0.8 | 131.5 | 26.3 + 9.4 | 134.2 |
| tp4, 48 | 57.2 + 26.9 + 0.8 | 431.1 | 76.0 + 23.1 | 384.2 |
| tp3, 1 | 19.1 + 1.5 + 0.9 | 109.9 | 22.5 + 5.6 | 114.5 |
| tp3, 16 | 25.5 + 12.2 + 0.8 | 200.8 | 33.8 + 7.8 | 173.5 |
| tp3, 48 | 97.2 + 40.1 + 0.8 | 696.1 | 100.6 + 15.2 | 510.0 |

The fused kernels chain with PDL, so their durations include waiting for their
predecessor; the loop span is the comparable figure
([`runs/markov-kernels-fp32.json`](runs/markov-kernels-fp32.json)). At a few rows
the projection's read of the packed shard (12.4 MB TP4, 16.6 MB TP3) is nearly all
of a position's time. The materialized bias it saves (`rows × shard × 4` bytes) is
small there, and the fused epilogue and its 1,010 / 1,348-tile finish cost about as
much. From about 16 rows (TP4) or 8 (TP3), writing and rereading the bias dominates
the materialized path.

### Memory

The fused path allocates no bias: the materialized one takes `rows × shard × 4`
bytes per position from the drafter's graph pool (up to 2.1 MB at 16 requests on
TP4, 1.4 MB at 8 on TP3, 8.3 MB at 48 on TP3). Its tile partials are `rows × tiles × 8`
bytes (at most 0.5 MB at 48 rows), borrowed from the workspace when it is
initialized.

## `step_groups=2` for the 64-row packed head tile (separate variable)

`bench_tile64_step_groups.py`: DS4.1's own output head rows (K = 5,120; TP4 32,320
and TP3 43,136 rows; 25,992 / 34,595 exceptions), packed, FP32 logits, at the
64-row tile (33..64 rows: the draft logits of 7..12 requests and the verification
rows of 6..10). It compares candidate-37's `TILE_CONFIGS[64]`
(`step_groups=1, num_stages=4`, 98,304 B shared memory) with #771's
(`step_groups=2, num_stages=3, act_stages=1`, 86,016 B). Medians of 5 rounds ×
100 replays, µs per call. Δ = step_groups=2 − candidate-37.

| shape | M | projection Δ boot 1 | boot 2 | greedy pairs Δ boot 1 | boot 2 | candidate-37 projection |
|---|---:|---:|---:|---:|---:|---:|
| tp4 | 33 | −1.8 | +18.6 | −23.2 | +2.9 | 1206.3 |
| tp4 | 40 | −2.1 | +25.1 | −19.2 | +11.7 | 1215.6 |
| tp4 | 45 | +15.2 | +35.9 | −2.0 | +24.7 | 1203.4 |
| tp4 | 48 | +18.8 | +50.1 | +9.8 | +43.6 | 1205.5 |
| tp4 | 54 | +40.0 | +78.5 | +57.3 | +67.0 | 1194.0 |
| tp4 | 64 | +71.4 | +123.0 | +123.2 | +128.2 | 1186.1 |
| tp3 | 33 | +30.1 | +22.0 | +12.7 | +3.6 | 1589.0 |
| tp3 | 40 | +44.1 | +17.5 | +16.6 | +14.5 | 1592.6 |
| tp3 | 48 | +64.7 | +40.1 | +63.1 | +42.1 | 1592.5 |
| tp3 | 64 | +152.8 | +149.2 | +179.3 | +188.0 | 1556.5 |

Both configurations give the same bits (projection and pairs, every case). On
candidate-37, `step_groups=2` is slower at TP3 at every row count and at TP4
from 45 rows. At TP4 33..42 rows the boots disagree in sign (boot 1 −2 to −23 µs,
boot 2 +2 to +30 µs). All rows are in
[`runs/tile64-boot1.json`](runs/tile64-boot1.json) and
[`runs/tile64-boot2.json`](runs/tile64-boot2.json). Not adopted.

## Runs and errors

- `runs/smoke.json`: a smoke run (tp4, M = 1 and 16, 2 rounds × 20 replays) before
  the full runs; same direction.
- A first speculator-test run (`-k fused`, 8 passed, ~45 s) ran on dgx4 while
  another session's context-ladder benchmark was starting there. Every later GPU run
  moved to sp8.
- Setup errors only: a wrong source path for the first tree preparation, and a copy
  loop that sent one file to sp8 before it was fixed. No GPU errors.
- `bin/spark doctor`: configuration OK. `bin/spark build check` needs prepared
  build inputs, which this checkout does not have, so it was not run.

## Conclusion

The fused path is bit-exact with the materialized one, top-2 included, and it uses
kernels candidate-37 already compiles. In the closest-to-serving loop (`chain`,
cold) it is faster from 3 requests at TP4 (−2.9 to −16.4 µs per five-position draft
step up to 16 requests) and at TP3 (−2.0 to −12.7 µs up to 8 requests). It is even
at 1–2 requests on TP4 and at 2 requests on TP3. It is 2–3 µs slower at one request
on TP3 (+0.4–0.6 µs per position). With the shard hot in L2 for the whole loop, it is
1–8 µs slower per step up to 12 requests (TP4) or 8 (TP3), even at 16 on TP4, and
faster beyond. These are microbenchmark deltas on one rank. No end-to-end run was
made; at one request the difference is about 0.03% of a 10 ms decode step.

Before it is called ready, it needs a serving soak on a candidate image: real
concurrent traffic, mixed prompt lengths, the token budget binding. The
functional soak above is clean but does not replace it. Promotion is the owner's
decision with a candidate screen. Follow-ups, each its own experiment:

- The 1–2-request overhead is the fused epilogue and the 1,010–1,348-tile finish.
  A cheaper finish, or wider tiles for the Markov head, could remove it.
- 0673's shard runner-up is the best maximum of the *other* 2048-column parts, so a
  runner-up in the top-1's own part is never reported, and the margin can be
  overstated. Top-2 is off in serving. Reporting the true runner-up changes the
  top-2 branch's behaviour and needs its own acceptance check.
