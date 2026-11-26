# spark-ds41f-1126: release notes

`spark-ds41f-1126` ([manifests/baselines/2026-11-26-spark-ds41f-1126.json](../../manifests/baselines/2026-11-26-spark-ds41f-1126.json),
[releases/spark-ds41f-1126/](../../releases/spark-ds41f-1126/)) is `candidate-74`, promoted on 2026-11-26.
The model is DeepSeek-V4.1-Flash-1017, DeepSeek's checkpoint, with native weights.

It advances `spark-ds41f-1120` with vLLM `0830..0869` (`742` active patches in
[candidate-74/vllm/series](candidate-74/vllm/series); `patch_head` `d8784b84`, `expected_tree`
`f3783b0e`), upstream `dgx-spark-networking v0.6.0` (`0` local patches, `expected_tree` `17394077`,
one-shot PP P2P + proxy ABI 10), and `scripts/memguard.sh` transient `/proc/meminfo` read retry.
Image `spark-ds41f:1126` (built on `dgx4` as `spark-ds41f:candidate-74`,
`sha256:94fff0b0b6ffa8ecd7bca576abf1d6c6be17f10c75bfc243b04d0479d356a4e7`), ring-copied identically
across `dgx1..dgx4`.

## Highlights (`spark-ds41f-1126` vs `spark-ds41f-1120`, 2 fresh-rebooted boots each)

- **`tp4-ring` ([manifests/suites/2026-11-26-spark-ds41f-1126-tp4](../../manifests/suites/2026-11-26-spark-ds41f-1126-tp4/)) — `0` `worse` verdicts, `9` `better`:**
  - Engine decode grid (20 shapes): **`-0.71% ±0.10%`** (repeat `+0.19%`, **better**).
  - Paired decode: `c16` **`+0.86% ±0.36%`** (**better**), `c8` **`+0.67% ±0.41%`** (**better**), `single-stream c1` **`+0.62% ±0.15%`** (**better**), `sampled c1` **`+0.61% ±0.36%`** (**better**; `c8`/`c16` `same`).
  - Source-text prefill `131K / 262K / 524K / 1M`: **`+1.62% / +1.95% / +2.38% / +3.07%`** (all **better**; `32K +1.19%` `same`).
  - Warm boot: **`+78.5 s`** (vs `1120`'s `+113–116 s`), `min MemAvailable 13.68–16.09 GiB`, `0.0 GiB` swap.
  - Wide determinism: **`0` token diffs, `0` logprob diffs**, bit-identical golden hashes to `spark-ds41f-1120` (`18d619615e51` / `d65686498eb8` / `0a2dacdde6ff` / `12fd825767d2`).
  - `tool_gate` (`T=0.0, 0.3`, stock + `deepseek-v4.1-flash-weightless` under `GLP-27 alpha=2.0`): **`80/80` strict PASS (`0` derailments)**.
- **`pp2-tp2-ring` (new qualified shape, [manifests/benchmarks/2026-11-26-spark-ds41f-1126-pp2tp2ring.json](../../manifests/benchmarks/2026-11-26-spark-ds41f-1126-pp2tp2ring.json)):**
  - `PP=2, TP=2` over `oneshot-pp2-ring4` + `sparknet v0.6.0` one-shot P2P, `24.0 GiB` KV (`30.70M` tokens = `29.28x` 1M contexts), async zero-sync PP adaptive verification (`row_cost=0.25`), `VLLM_SINK_REJECTION=1`, `SPARK_DS41F_PP_USE_ENGINE_REQS=1`, intra-stage SP (`0859`).
  - Patch `0869` anchors the `77.5 GiB` contiguous CUDA pool inside `[pool_ptr, pool_end)` on `PP1` ranks (`dgx3`/`dgx4`), shrinks `PP0` to `74.0 GiB` (saving `3.5 GiB`), and eliminates buddy-sponge fallback across all 4 ranks (`min MemAvailable 10.22–13.27 GiB`, `0.0 GiB` swap).
  - Wide determinism: `0` token diffs across all 6 scenarios, bit-identical hashes to `tp2-direct` (`7278cd22ed16` / `1dd5b1dbdb63`).
  - Decode `prose-c1..c16` `48.1–48.3 / 85.3–86.9 / 142.9–146.5 / 238.2–256.0 / 421.1–433.2 tok/s`, `code-c16` `438.8–449.4 tok/s`, source-text prefill `32K..1M` `6,173–6,513` / `6,059–6,084 tok/s`.
- **`tp3-triangle` ([manifests/suites/2026-11-26-spark-ds41f-1126-tp3](../../manifests/suites/2026-11-26-spark-ds41f-1126-tp3/)):**
  - Engine decode grid (15 shapes): **`-25.19% ±0.10%`** (`repeat -0.01%`, **better** — also recovers `1120`'s `1x*` dummy-padding grid regression).
  - Paired decode: `c8` **`+0.77% ±0.53%`** (**better**), `c4` **`+1.34% ±0.88%`** (**better**), `single-stream c1` **`+1.30% ±0.11%`** (`71.32 tok/s`, **better**), `sampled c1` **`+1.39% ±0.38%`** (**better**), `sampled c8` **`+0.00%`** (`same`).
  - Sampled `c16`: **`+31.53% ±1.36%`** aggregate `output tok/s` (`167.2 → 220.0 tok/s`, **better**; `TTFT p50` `12.67 s → 0.45 s`, `-96.5%`) from raising `max_num_seqs` `8 → 16` (per-stream paired decode naturally splits across 16 concurrent streams instead of 8 queued behind 8).
  - Source-text prefill `131K / 262K / 524K / 1M`: **`+1.06% / +1.32% / +1.93% / +2.86%`** (all **better**; `32K +0.78%` `same`).
  - `0869` kept the `96.5 GiB` contiguous CUDA pool (`front_clean=True`) across all 3 ranks skipping buddy sponge on both main and draft weights (`min MemAvailable 6.16–9.62 GiB`, `0.0 GiB` swap), `0/0` wide determinism diffs, and `80/80` strict `tool_gate` PASS (`T=0.0, 0.3`).
- **`tp3-triangle-4k` ([manifests/benchmarks/2026-11-26-spark-ds41f-1126-tp34k.json](../../manifests/benchmarks/2026-11-26-spark-ds41f-1126-tp34k.json)):**
  - Prefill `32K / 131K / 262K`: **`8,337–8,377 / 8,417–8,471 / 8,363–8,431 tok/s`** (**`+22.0%..+24.2%`** faster than `1120`).
  - Decode `same`/`faster` across all 8 points (`LRU 5/5`), `0/0` wide determinism diffs (bit-identical hashes to `tp3-triangle`), `min MemAvailable 11.30–13.33 GiB`, `0.0 GiB` swap.
- **`tp2-direct` ([manifests/benchmarks/2026-11-26-spark-ds41f-1126-direct2.json](../../manifests/benchmarks/2026-11-26-spark-ds41f-1126-direct2.json)):**
  - Decode **`+41.0%..+75.5%`** faster (`prose-c1/c2/c4` `28.1–28.3 / 49.5–49.6 / 92.7–92.9 tok/s`, `code-c4` `93.5–94.3 tok/s`; step times `-37.6%..-55.6%`).
  - Prefill **`+1,695%..+1,975%`** faster (`32K 4,646–4,668`, `131K 4,873–4,888 tok/s` via expert-centric prefill streaming `0858/0865`, bundled multidomain hotlist, and BF16 fast lookahead `0863`).
  - `min MemAvailable 9.54–11.65 GiB`, `0.0 GiB` swap, `LRU 5/5`, `prefix hit 0.99`, `admission 4×64K pass`, `0` determinism diffs.
- **`tp1` ([manifests/benchmarks/2026-11-26-spark-ds41f-1126-tp1.json](../../manifests/benchmarks/2026-11-26-spark-ds41f-1126-tp1.json)):**
  - Decode **`+74.6%..+177.2%`** faster (`prose-c1/c2/c4` `9.2–9.3 / 15.8 / 30.2–30.3 tok/s`, `code-c4` `28.2–28.4 tok/s`; step times `-50.6%..-71.3%`), prefill `512/1024` **`219.4 / 324.5 tok/s`** (**`+816% / +1,232%`**), bit-identical across `dgx3` and `dgx4` (`min MemAvailable 9.07–9.17 GiB`, `0.0 GiB` swap).
- **Correctness, steering, & test gates:**
  - Reworked empty-index-slice decode RoPE fix (`0868`, PR `#1323` by James Meadlock).
  - `GLP-27 alpha=2.0` (`0855`, PR `#1254`/`#1319`) with per-model `0.3/0.95` defaults for `deepseek-v4.1-flash-weightless` (`0856`).
  - Full `174`-file in-image `release-tests` bundle (`dgx2`): **`12,082 passed, 356 skipped, 0 failed`** (`exit 0`); Linux repo unit suite (`dgx1`): **`950/950 passed`**.
  - Switched shapes retain their `spark-ds41f-1120` references pending community switched qualification ([docs/switched-qualification.md](../../docs/switched-qualification.md)).

---

# spark-ds41f-1120: release notes

`spark-ds41f-1120` ([manifests/baselines/2026-11-20-spark-ds41f-1120.json](../../manifests/baselines/2026-11-20-spark-ds41f-1120.json),
[releases/spark-ds41f-1120/](../../releases/spark-ds41f-1120/)) is `candidate-62`, promoted on 2026-11-20.
The model is DeepSeek-V4.1-Flash-1017, DeepSeek's checkpoint, with native weights.

It advances `spark-ds41f-1107` onto CUDA `13.5.37` (hash-locked over the base image's CUDA `13.0.2`), retires
B12X completely (`0802`), updates vLLM to `702` active patches through `0829`
([candidate-62/vllm/series](candidate-62/vllm/series); `patch_head` `9b251a18`, `expected_tree` `8fc99ca7`),
TileLang to `14` patches (`expected_tree` `741806c7`), and `sparknet` to `3` patches (`expected_tree`
`e06f6a06`). Image `spark-ds41f:1120` (built on `dgx4` as `spark-ds41f:candidate-62`,
`sha256:5c12472565a708518ec0566b6653b5bd324e06881a34b90f69d6b8210a386d08`), distributed identically across all
12 nodes (`dgx1..dgx4` and `sp1..sp8`).

The release expands qualification from 8 to all **12 topologies**:
- **Switchless (`dgx1..dgx4`):** `tp4-ring` (`1M`, `32` seqs, `16.50 GiB` KV), `tp3-triangle` (`1M`, `8` seqs, `4.90625 GiB`), `tp3-triangle-4k` (`256K`, `8` seqs, `2.20 GiB`, stock 4 KiB pages without memory-saver), and `tp2-direct` (`256K`, `4` seqs, `8.00 GiB`).
- **Switched (`sp1..sp8`):** `tp2-switched` (`256K`, `4` seqs, `8.00 GiB`), `tp3-switched` (`1M`, `8` seqs, `4.90625 GiB`), `tp4-switched` (`16.50 GiB`), `tp5-switched` (`24.00 GiB`), `tp6-switched` (`13.00 GiB`), `tp7-switched` (`16.00 GiB`), and `tp8-switched` (`25.50 GiB`, all `TP4..TP8` at `1M` and `32` seqs).
- **Local (`dgx4`):** `tp1` (`256K`, `4` seqs, `8.00 GiB`, `uni` executor without RDMA).

Per the `rc12 -> rc14` carry-over precedent, `candidate-62` leaves `VLLM_DS41_LSA` unset by default (`0` new kernels, `jit_new_files = 0`) and carries over the 2-boot `candidate-61` / `candidate-61-r2` Sonar suites and `bin/spark bench` matrices alongside its own `12,023`-test in-image gate and 2-boot `tp3-triangle` & `tp1` verification.

## Highlights (`spark-ds41f-1120` vs `spark-ds41f-1107`)

- **Faster decode & prefill across switchless and switched topologies:**
  - **`tp4-ring` ([manifests/suites/2026-11-20-spark-ds41f-1120-tp4](../../manifests/suites/2026-11-20-spark-ds41f-1120-tp4/)):** paired decode `+3.07%` at `c16` (`280.43 tok/s`), `+2.70%` at `c8` (`213.57 tok/s`), `+7.30%` at `single-stream c1` (`89.06 tok/s`); engine decode grid `-7.73%` geometric mean (`19.36 ms` vs `21.05 ms`, `20/20` shapes faster); prefill `+12.91% / +16.51% / +21.60% / +31.78% / +56.31%` across `32K..1M` (`9,615 / 10,109 / 10,056 / 9,833 / 9,397 tok/s`); sampled decode `+3.23% / +3.01% / +8.61%` (`276.58 / 210.02 / 87.07 tok/s`); stall `131K` prefill `+12.44%..+13.73%` (`TTFT p50` `0.73 s / 1.14 s` vs `1.02 s / 1.44 s`).
  - **`tp3-triangle` ([manifests/suites/2026-11-20-spark-ds41f-1120-tp3](../../manifests/suites/2026-11-20-spark-ds41f-1120-tp3/)):** paired decode `+5.99%` at `c8` (`170.68 tok/s`), `+6.26%` at `c4` (`131.59 tok/s`), `+10.39%` at `single-stream c1` (`70.36 tok/s`); sampled `+6.47% / +6.52% / +7.75%` (`167.22 / 167.59 / 68.27 tok/s`); prefill `+12.17%..+55.69%` (`7,270 / 7,502 / 7,445 / 7,259 / 6,922 tok/s`); stall-3/6/7 prefill `+12.64% / +13.19% / +14.31%`; multi-request engine grid shapes improved `-3.10%..-12.69%`, while `1x*` dummy-padding grid rows moved the 15-shape geomean to `25.52 ms` (`+20.20%`) without affecting live single-stream `c1` (`+10.39%`).
  - **`tp4-switched` ([manifests/suites/2026-11-20-spark-ds41f-1120-tp4-switched](../../manifests/suites/2026-11-20-spark-ds41f-1120-tp4-switched/)):** grid `-9.83%` (`19.05 ms` vs `21.14 ms`), paired `c16` `+6.51%` (`280.51 tok/s`), `c8` `+6.82%` (`213.06 tok/s`), `single-stream c1` `+9.39%` (`88.97 tok/s`), sampled `+6.64% / +6.68% / +10.90%` (`276.44 / 209.60 / 87.31 tok/s`), prefill `+12.96%..+56.02%` (`9,569..9,308 tok/s`), stall prefill `+11.88%..+14.93%`.
  - **`tp8-switched` ([manifests/suites/2026-11-20-spark-ds41f-1120-tp8-switched](../../manifests/suites/2026-11-20-spark-ds41f-1120-tp8-switched/)):** grid `-9.92%` (`17.60 ms` vs `19.54 ms`), paired `c16` `+6.81%` (`395.40 tok/s`), `c8` `+7.35%` (`312.94 tok/s`), `single-stream c1` `+11.74%` (`129.43 tok/s`), sampled `+6.53% / +7.21% / +11.19%` (`387.49 / 306.80 / 125.29 tok/s`), prefill `+5.99% / +9.32% / +12.11% / +18.89% / +34.92%` (`11,846 / 12,429 / 12,383 / 12,206 / 11,806 tok/s`), stall prefill `+10.25% / +17.17% / +16.39%`.
  - **`tp3-switched` ([manifests/suites/2026-11-20-spark-ds41f-1120-tp3-switched](../../manifests/suites/2026-11-20-spark-ds41f-1120-tp3-switched/)):** paired `c8` `+6.87%` (`168.92 tok/s`), `c4` `+7.05%` (`130.01 tok/s`), `single-stream c1` `+12.54%` (`69.26 tok/s`), sampled `+6.65% / +6.62% / +10.89%` (`165.74 / 165.56 / 67.36 tok/s`), prefill `7,197 / 7,428 / 7,373 / 7,195 / 6,853 tok/s`.
- **Expanded `TP4..TP8` concurrency (`32` streams, `192`-row CG) & KV capacity:** raises `--max-num-seqs` from `16` to `32` (`--max-cudagraph-capture-size 192`) and expands per-rank KV pools across `TP4..TP8` (`16.50 GiB` / `17.86x` 1M contexts on `tp4-*`, `24.00 GiB` on `tp5`, `13.00 GiB` on `tp6`, `16.00 GiB` on `tp7`, `25.50 GiB` / `27.17x` on `tp8`).
- **Four new topologies (`tp1`, `tp2-direct`, `tp2-switched`, `tp3-triangle-4k`):** adds single-node and two-node `256K` serving backed by the zero-copy `io_uring` expert slot cache (`0822`) and `RM` `nvidiactl` display scanout carve-out (`0821`), plus a `4 KiB`-page `tp3-triangle-4k` profile (`256K` ctx, `2.20 GiB` KV) without Memory Saver.
- **API, structured output, parser, and release-gate hardening (`0823..0829`):** fixes `StructuredOutputManager` bitmask row sizing (`0823`), `TOOL_ARGS` DSML closers and incomplete parameter-tag streaming holdback (`0824`, `0829`), SSE keep-alive per-wait callback races (`0825`), `CustomTool` / OpenResponses normalization (`0826`), quantization `ignored_layers` checkpoint mapping (`0827`), and adds opt-in `VLLM_DS41_LSA=1` FP8 group-32 dominance summaries (`0828`, off by default).
- **Release Gate & Startup Memory Across All 12 Shapes:**
  - `candidate-62` release-tests gate: `12,023 passed, 356 skipped, 0 failed` (`169` files, including `106` parser tests across `test_deepseek_v41.py` and `test_deepseek_v41_structural_closers.py`).
  - `candidate-62` 2-boot `tp3-triangle`: `8.27 / 8.28 GiB` min head `MemAvailable`, `0.0` swap growth, `7/7` wide determinism & greedy identical, live smoke pass.
  - `candidate-62` 2-boot `tp1`: `14.09 / 14.60 GiB` min `MemAvailable`, `0.0` swap growth, `6/6` greedy identical.
  - All 12 `manifests/benchmarks/2026-11-20-spark-ds41f-1120-*.json` runs passed `5/5` quality checks, collectives, and `> 5.0 GiB` head/worker `MemAvailable` with `0.0 GiB` swap growth (`tp5..tp7-switched` prefill up to `11,206 tok/s`, `tp2-direct/tp2-switched` `52.96 / 53.59` `prose-c4`, `tp1` `10.94 / 11.36` `c4`).

---

# spark-ds41f-1107: release notes

`spark-ds41f-1107` ([manifests/baselines/2026-11-07-spark-ds41f-1107.json](../../manifests/baselines/2026-11-07-spark-ds41f-1107.json),
[releases/spark-ds41f-1107/](../../releases/spark-ds41f-1107/)) is a release of this deployment,
named after the day it is promoted. The model is DeepSeek-V4.1-Flash-1017, DeepSeek's checkpoint,
with native weights.

It advances `spark-ds41f-1030` across vLLM (`641` active patches through `0761` in
[1107/vllm/series](1107/vllm/series); `patch_head` `2756478b`, `expected_tree` `ed1bdb7a`) and
`sparknet` (`3` active patches `0002..0004` in [1107/sparknet/series](1107/sparknet/series);
`patch_head` `1f84980b`, `expected_tree` `b86dc07f`). Image `spark-ds41f:1107`
(`sha256:3f32a962ce04b7c8d85337cd427a8946b8b6bcaadd97cc8d8a8cd14e8d68a6ce`).

For the first time, the release is qualified across all eight supported topologies:
- **Switchless (`dgx1..dgx4`):** `tp4-ring` and `tp3-triangle` (`1,048,576` context).
- **Switched (`sp1..sp8`):** `tp3-switched`, `tp4-switched`, `tp5-switched`, `tp6-switched`,
  `tp7-switched`, and `tp8-switched` (`1,048,576` context).

## Highlights (`spark-ds41f-1107` vs `spark-ds41f-1030`)

- **Faster across every workload on both switchless topologies:**
  - **`tp4-ring` (`dgx1..dgx4`):** paired decode `+5.30%` at `c16` (`272.86 tok/s`), `+4.04%` at
    `c8` (`207.45 tok/s`), `+4.92%` at `c1` (`82.62 tok/s`); engine decode grid `-14.36%` geometric
    mean (`21.48 ms` vs `25.09 ms`, `20/20` shapes faster); prefill `+8.34%` to `+17.05%` from
    `32K` to `1M` (`8,508` / `8,657` / `8,261` / `7,460` / `6,012 tok/s`); sampled decode `+4.56%`
    to `+6.40%` (`268.31` / `201.40` / `79.56 tok/s`); short-request TTFT p50 during `131K` prefill
    under `3` and `11` streams `-69.22%` (`3.30 s -> 1.02 s`) and `-59.92%` (`3.59 s -> 1.44 s`);
    greedy `256/256` identical.
  - **`tp3-triangle` (`dgx1..dgx3`):** paired decode `+1.70%` at `c8` (`160.80 tok/s`), `+2.04%` at
    `c4` (`123.73 tok/s`), `+2.59%` at `c1` (`64.28 tok/s`); engine decode grid `-15.36%` geometric
    mean (`21.29 ms` vs `25.15 ms`, `15/15` shapes faster); prefill `+2.99%` to `+10.55%` from
    `32K` to `1M` (`6,452` / `6,480` / `6,135` / `5,514` / `4,434 tok/s`); sampled decode `+1.92%`
    to `+2.28%` (`156.64` / `156.25` / `62.80 tok/s`); short-request TTFT p50 during `131K` prefill
    under `3` and `6` streams `-47.10%` (`2.03 s -> 1.07 s`) and `-87.67%` (`9.59 s -> 1.18 s`);
    greedy `256/256` identical.
- **Full `tp3-switched` through `tp8-switched` qualification (`sp1..sp8`):**
  - Arbitrary tensor-parallel world sizes (`TP=3, 4, 5, 6, 7, 8`) with `64` routed MoE experts and
    `4` attention heads via cyclic KV head replication and deduplicated linear-projection outputs
    (`0756..0759`).
  - All 6 switched shapes passed 2-boot `bin/spark bench` ([manifests/benchmarks/2026-11-07-spark-ds41f-1107-switched.json](../../manifests/benchmarks/2026-11-07-spark-ds41f-1107-switched.json) and `switched{3,5,6,7,8}.json`), collective probes, and `tp3-switched`, `tp4-switched`, `tp8-switched` `servebench`, quality evals, and 1-hour `AgentX` `c8` runs.
- **TP3 Cold-Boot Capture Headroom (`0761`):** flushes the PyTorch CUDA caching allocator and trims
  glibc's host arena between weight load, kernel warmup, KV cache allocation, CUDA Graph capture,
  and speculator capture, raising `tp3-switched` cold-boot `min_available_gib` on `sp5..sp7` from
  `<3.46 GiB` to `6.03..6.45 GiB` on head `sp5` (`7.94..8.10 GiB` on followers `sp6..sp7`).
- **Release Gate, Quality Evals, and AgentX:**
  - Image unit/contract test suite: `648 passed, 0 failed` (`build-check` `6/6`, `patch-contracts`
    `337/337`, `test-suites` `305/305`).
  - Determinism gate: `20/20` bit-identical across `tp4-ring`, `tp3-triangle`, `tp3-switched`,
    `tp4-switched`, and `tp8-switched`.
  - Quality evals (`tp3-triangle`, `tp4-ring`, `tp3-switched`, `tp4-switched`, `tp8-switched`):
    `LongBench v2` `8/12` (`66.67%`), `RULER` (`4K..256K`) `26/26` (`100.0%`), `EvalPlus HumanEval+`
    `164/164` base (`100.0%`) / `156/164` plus (`95.1%`), `Vision Eval` `83.25%` (`1900.5/2283`), and
    `tool-eval-bench` (`v3.0.1.dev10+g1cd7309e5`) `90..91/100` (`Grade A`, `86/100` deployability).
  - `AgentX` (`1h`, `c8`, `seed 42`): `125.91 tok/s` (`tp3-switched`), `158.89 tok/s`
    (`tp4-switched`), `165.32 tok/s` (`tp4-ring`, `756` requests, `96.74%` prefix cache hit rate),
    and `216.92 tok/s` (`tp8-switched`), all with `0` errors and `0` preemptions.

---

# spark-ds41f-1030: release notes

spark-ds41f-1030 is a release of this deployment, named after the day it is promoted. The model
is DeepSeek-V4.1-Flash-1017, DeepSeek's checkpoint, with native weights.

It is candidate 14: spark-ds41f-1025 plus vLLM 0491-0567 (candidates 7-14 in
[README.md](README.md)). Image `spark-ds41f:1030` (built as `spark-ds41f:19f2c20ed4d6-next-14`,
`sha256:7918fd6145b5`), vLLM patch head `b6de07aa`, tree `ea1281f8`. Four-node ring (tp4-ring),
and the three-node triangle (tp3-triangle) from 2026-10-30.

## Results (the ring, 2026-10-29)

The speed comparison ran on candidate 12. Candidates 13 and 14 change two things after it, each
checked for bits, memory and startup on the ring: 0566 stores the 2-bit expert scale codes in
the 4-bit codes' buffer (model loading back to 71.67 GiB a rank from candidate 12's 72.72), and
0567 is the Engram host-hash fix below. Neither changes a kernel's arithmetic or what decode
reads, and every digest is candidate 12's.

| Check | Result |
| --- | --- |
| Release tests | candidate 12's image: 6,686 passed, 163 skipped, 0 failed. Candidate 13's scale-code bundle: 117 passed; 0567's five tests pass |
| Temperature-0 outputs | every digest equals candidate 10's and candidate 12's (determinism with and without logprobs, 16K, greedy, seeded sampled, prompt logprobs, images, CED cross-shape, control spellings and controls); candidates 13 and 14 keep them (`fe2461d8`, `c9cf9dbe`, `fcd34fd2`) |
| Model behaviour against 1025 | greedy 256/256 identical; teacher-forced KL 4.2e-8 nats over 65,536 positions (0 between the candidate boots) |
| GLP dual serving | both names served; steering moves all 8 probe prompts; mixed batches and the prefix cache keep each name's text; beam search refused with 400. AdvBench (520 prompts): no refusal phrase in 2.5% stock, 51.2% steered |
| #677's load | 4 agents to 311K tokens and 8 chats resending each question, 300 s, on two boots of candidate 14: 0 failed of 150 requests each |
| Startup | 277 TileLang calls prepared; init 28-37 s; nothing compiles after |
| Memory | model loading 71.67 GiB a rank (1025's); the ring's lowest free memory in a boot 19.4 GiB or more |
| KV capacity | vLLM's startup figure: 11,554,122 tokens, 11.02 full 1M contexts (#586's count; 1025's line read 6.21 for the same pool) |

Speed against spark-ds41f-1025, one session (candidate 12, spark-ds41f-1025, candidate 12
again), verdicts against the repeat:

| Metric | spark-ds41f-1025 | spark-ds41f-1030 | Change | Verdict |
| --- | ---: | ---: | ---: | --- |
| decode grid, geometric mean (20 shapes) | | | -4.62% | better |
| one stream, decode tok/s (paired) | 75.4 | 78.3 | +3.61% | better |
| one stream, TTFT (paired) | 0.128 s | 0.122 s | -4.5% | better |
| 8 streams, decode tok/s (paired) | | | +2.02% | better |
| 8 streams, output tok/s | 194.6 | 198.4 | +1.96% | better |
| 16 streams, decode tok/s (paired) | | | +2.05% | better |
| 16 streams, output tok/s | 251.7 | 257.7 | +2.40% | better |
| prefill 32K / 131K / 262K / 524K / 1M tok/s | 6,744 / 6,798 / 6,477 / 5,870 / 4,948 | 7,313 / 7,396 / 7,120 / 6,538 / 5,549 | +8.4% to +12.2% | better |
| 131K prefill beside 15 decoding streams | 6,447 | 6,978 | +8.23% | better |
| 131K prefill beside 3 / 11 streams | | | +0.43% / -0.74% | same |
| short requests' TTFT p50 during a 131K prefill | 12.4-12.8 s | 3.3-3.6 s | 3.6-3.7x sooner | better |
| Poisson output tok/s | | | -0.01% | same |
| sampled: c16 / c8 / c1 decode (paired) | | | +1.87% / +2.77% / +2.68% | better |

The records are in [manifests/suites/2026-10-30-spark-ds41f-1030-tp4](../../manifests/suites/2026-10-30-spark-ds41f-1030-tp4/)
and its `-single-stream` and `-sampled` companions: candidate 12's first arm, now tp4-ring's
`benchmark_reference`.

## What changes for users

- **Faster.** Decode 1.9-3.6% at every concurrency, prefill 8-12% from 32K to 1M.
- **Short requests are not stuck behind a long prompt.** Prefill yield (`--prefill-yield short`,
  10 s limit): a waiting prompt that fits in one chunk, and a held first token, take the next
  step from a long prompt's prefill. Beside a 131K prefill a short request's first token comes
  in 3.3-3.6 s instead of 12.4-12.8 s; the long prompt pays about 9% of its rate while short
  requests keep arriving, and nothing when none are.
- **A second served name, `deepseek-v4.1-flash-weightless`,** steered per request with msuiche's
  published GLP vector for 1017 (#401). `deepseek-v4.1-flash` is unchanged and bit-identical
  beside it. Every node needs the vector file (docs/setup.md, section 5); `cluster start` refuses
  without it.
- **Fixed: an engine stop under mixed agent and chat traffic (#677).** A prompt with one token
  left that the scheduler padded with placeholder drafts reached the Engram host hash, whose
  check raised and stopped EngineCore. The GPU now hashes such a request's rows (0567).
  spark-ds41f-1025 has the same path; the trigger was not reproduced on the ring.
- **DeepSeek API semantics:** the effort aliases, `response_format`, `tool_choice: none` and the
  assistant prefix follow DeepSeek's API (0539-0540).
- **CED replays a prompt's last 128 rows wherever its chunks end (0526-0527).** Temperature-0
  text changes for multi-chunk prompts whose final chunk has 1-128 rows (0.79% of multi-chunk
  prompts at TP4); every other prompt is unchanged.
- **The startup KV line counts what the pool holds** (#586): 11.02 full 1M contexts.

## For operators

- TileLang is the only kernel family; `VLLM_DS41_KERNEL_BACKEND` is gone and
  `VLLM_ENABLE_ROCE_ALLREDUCE=1` is refused. B12X stays in the image for its packages only,
  loaded by nothing (`VLLM_PLUGINS` empty, `--load-format direct`).
- IKET's debug mode is in the image, off (`VLLM_DS41_IKET` unset).
- The recipe and its lock are frozen in [releases/spark-ds41f-1030](../../releases/spark-ds41f-1030/);
  the rollback is [releases/spark-ds41f-1025](../../releases/spark-ds41f-1025/) on
  `spark-ds41f:19f2c20ed4d6-next-6`.

## Results (the triangle, 2026-10-30)

The release at the triangle ([rc14/rc-tp3.json](rc14/rc-tp3.json), the 6,144-token budget),
checked in run `candidate-14-tp3` and compared in run `candidate-14-release-tp3` (the release,
spark-ds41f-1025, the release again; verdicts against the repeat).

| Check | Result |
| --- | --- |
| Temperature-0 outputs | at 8,192, every greedy digest equals spark-ds41f-1025's apart from CED's cross-shape prompt (0526); the release at 6,144 equals 8,192 in every digest |
| Model behaviour against 1025 | greedy 256/256 identical; teacher-forced KL 4.2e-8 nats |
| Functional checks and GLP | pass; AdvBench no refusal phrase in 1.9% stock |
| Memory | model loading 93.95 GiB a rank (1025: 94.12); dgx1's lowest MemAvailable 7.29 and 7.95 GiB (1025: 6.23) |
| Startup | 276 TileLang calls prepared; nothing compiles after |

| Metric | spark-ds41f-1025 | spark-ds41f-1030 | Change | Verdict |
| --- | ---: | ---: | ---: | --- |
| decode grid, geometric mean (15 shapes) | | | -5.99% | better |
| one stream, decode tok/s (paired) | 61.2 | 63.5 | +3.76% | better |
| one stream, TTFT (paired) | 0.159 s | 0.153 s | -4.1% | better |
| 4 streams, decode tok/s (paired) | | | +2.68% | better |
| 8 streams, decode tok/s (paired) | | | +2.29% | better |
| 8 streams, output tok/s | 154.7 | 158.6 | +2.51% | better |
| Poisson, output tok/s | 118.9 | 119.0 | +0.09% | same |
| prefill 32K / 131K / 262K / 500K | | | +4.4 / +4.1 / +5.3 / +7.0% | better |
| 131K prefill beside 7 streams | 5,307 | 5,465 | +2.98% | better |
| 131K prefill beside 3 streams, short requests waiting | 5,314 | 5,069 | -4.61% | worse |
| short requests' first token there, p50 | 16.6 s | 2.0 s | -87.8% | better |
| 131K prefill beside 6 streams, short requests waiting | 5,295 | 5,099 | -3.70% | worse |
| short requests' first token there, p50 | 18.3 s | 9.6 s | -47.6% | better |
| sampled c16 / c8 / c1, decode tok/s (paired) | | | +2.76 / +2.41 / +2.94% | better |

The owner accepted the two worse verdicts on 2026-10-30. The suites are
[manifests/suites/2026-10-30-spark-ds41f-1030-tp3](../../manifests/suites/2026-10-30-spark-ds41f-1030-tp3/).

## Limitations

- **On the triangle, a 131K prefill beside 3 or 6 streams with short requests waiting is
  3.7-4.6% slower** than spark-ds41f-1025's: it yields to those requests (their first token
  2-8x sooner) and runs the 6,144 budget (1025 qualified the triangle at 8,192).
- **A lone 131K prefill is 0.8% slower** than candidate 10's (in 0541-0544 or the IKET patches,
  not yet bisected); against 1025 it is still 8.8% faster.
- **Seeded sampling at 8 streams depends on the batch** under adaptive verification, as in
  spark-ds41f-1025.
- **Copy drafts are not in this release** (#619): enabled, they fail to start.
