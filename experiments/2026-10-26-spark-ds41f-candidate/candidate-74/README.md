# `candidate-74` — Promoted as `spark-ds41f-1126`

Release candidate superseding frozen `candidate-73` (`24950b3eca`) and promoted on **2026-11-26** as
**`spark-ds41f-1126`** (`manifests/baselines/2026-11-26-spark-ds41f-1126.json`,
`releases/spark-ds41f-1126/`).

## Changes Over Frozen `candidate-73`

1. **vLLM patch `0869`** (`fix(worker): anchor and atomically release contiguous CUDA pool on fallback`,
   `742` active patches, `patch_head` `d8784b841c36`, `expected_tree` `f3783b0ec618`):
   - Drains stale cached-free CUDA segments (`torch.accelerator.empty_cache()`) before allocating
     the soaked contiguous pool and widens anchor draining to `1024 x 2 MiB` so `PP>1` last stages
     (which initialize speculator/sampler in `GPUModelRunner.__init__` before
     `_preallocate_contiguous_cuda_pool()`) always anchor inside `[pool_ptr, pool_end)` instead of
     silently anchoring an unrelated segment and letting `garbage_collection_threshold:0.6` evict
     the pool.
   - Raises `RuntimeError` if an anchor cannot land inside `[pool_ptr, pool_end)`.
   - Wires `Worker._release_contiguous_cuda_pool` as `release_cb` into
     `direct_loader.set_preallocated_cuda_pool_bytes()` so any fallback in `bind_weight_arena()`
     atomically evacuates and releases the pre-allocated pool before buddy-sponge allocation.
   - Gates both `Keeping` fast paths in `_resize_contiguous_cuda_pool_for_model()` on actual
     `cached_free_bytes >= model_bytes`.
   - Skips the `draft_and_other` pool reservation on non-last PP ranks
     (`pp_size > 1 and not is_last_rank`), freeing `3.5 GiB` on `PP0`.
2. **`scripts/memguard.sh` transient-read retry**: retries up to 3 attempts (`0.05 s` spacing) on
   empty/non-numeric `/proc/meminfo` reads before failing closed (`+2` unit tests in
   `tests/test_memguard.py`).

All other build inputs are identical to `candidate-73` (`sparknet v0.6.0` tree `173940770627` with
`0` local patches, `TileLang 0.1.15`, `TileKernels 2.0.0`, `NCCL 2.30.7`, `CUDA 13.5.37`,
`CUTLASS v4.7.1`, `GLP-27 alpha=2.0` / `0.3/0.95` defaults).

## Image & Build Identity

- **Built image**: `spark-ds41f:candidate-74` (retagged `spark-ds41f:1126` on `dgx1..dgx4`)
- **OCI digest**: `sha256:94fff0b0b6ffa8ecd7bca576abf1d6c6be17f10c75bfc243b04d0479d356a4e7`
- **Labels**: `local.spark-ds41f.vllm.tree=f3783b0ec61851ba481f6a3a316d0a76d0e171d8`,
  `local.spark-ds41f.sparknet.tree=1739407706277142da7a12c95fca3c503c8cb11c`
- **Build host**: `dgx4` (`672 s`, lowest `MemAvailable 78.4 GiB`, GPU/TileLang/sparknet smoke
  passed), ring-copied `dgx4 -> dgx3 -> dgx2 -> dgx1` with identical digest on all four nodes.

## Qualification Receipts (2 Fresh-Rebooted Boots per Switchless Shape)

- **In-image `174`-file `release-tests` bundle (`dgx2`, [`receipts/release-tests/verdict.json`](receipts/release-tests/verdict.json))**:
  **`12,082 passed, 356 skipped, 0 failed`** (`exit 0`, `+3` new `0869` regression tests in
  `test_contiguous_cuda_pool.py`).
- **Linux repo unit suite (`dgx1`, [`receipts/linux-unittest-dgx1.txt`](receipts/linux-unittest-dgx1.txt))**:
  **`950 passed, 0 failed`** (`37.2 s`).
- **`tp4-ring` (`b1` + `b2`, clean reboot before each boot, vs `spark-ds41f-1120`)**:
  - Pool logs ([`receipts/pool-logs/tp4-ring-b1-pool.log`](receipts/pool-logs/tp4-ring-b1-pool.log), [`b2`](receipts/pool-logs/tp4-ring-b2-pool.log)):
    all 4 ranks resized `85.0 -> 74.5 GiB` (`75,700.3 MiB` main / `5,330 MiB` draft cached free,
    skipping buddy sponge), warm ready `+78.5 s`, `min MemAvailable 13.68–16.09 GiB`, `0.0 GiB` swap.
  - Wide determinism ([`receipts/determinism/tp4-ring-wide.jsonl`](receipts/determinism/tp4-ring-wide.jsonl),
    [`tp4-ring-wide-nologprobs.jsonl`](receipts/determinism/tp4-ring-wide-nologprobs.jsonl)):
    **`0` token diffs, `0` logprob diffs**, bit-identical golden hashes to `spark-ds41f-1120`
    (`18d619615e51` / `d65686498eb8` / `0a2dacdde6ff` / `12fd825767d2`).
  - `tool_gate` ([`receipts/tool-gate/tool-gate-tp4-ring.txt`](receipts/tool-gate/tool-gate-tp4-ring.txt)):
    **`80/80` strict PASS (`0` derailments)** across `T=0.0, 0.3` on stock and `weightless`.
  - Sonar verdicts (`manifests/suites/2026-11-26-spark-ds41f-1126-tp4/`): **`0` `worse` verdicts**:
    `grid -0.71% ±0.10%` (`better`), paired `c16 +0.86%` (`better`), `c8 +0.67%` (`better`),
    `single-stream c1 +0.62%` (`better`), `sampled c1 +0.61%` (`better`), prefill `131K..1M`
    `+1.62%..+3.07%` (all `better`), `stall-3/11/15` `same` (`+0.47%..+1.33%`).
- **`pp2-tp2-ring` (`b1` + `b2`, clean reboot before each boot, `24.0 GiB` KV)**:
  - Pool logs ([`b1`](receipts/pool-logs/pp2-tp2-ring-b1-pool.log), [`b2`](receipts/pool-logs/pp2-tp2-ring-b2-pool.log)):
    `PP0` (`dgx1`/`dgx2`) resized `77.5 -> 74.0 GiB` (`75,140.8 MiB` cached free, skipping buddy
    sponge); `PP1` (`dgx3`/`dgx4`) kept `77.5 GiB` (`79,726.8 MiB` main / `10,612.6 MiB` draft
    cached free, skipping buddy sponge on both). Zero buddy-sponge fallbacks, zero memguard kills,
    `min MemAvailable 10.22–13.27 GiB`, `0.0 GiB` swap.
  - Wide determinism ([`receipts/determinism/pp2-tp2-ring-wide-nologprobs.jsonl`](receipts/determinism/pp2-tp2-ring-wide-nologprobs.jsonl)):
    `identical: true`, `0` token diffs across all 6 scenarios — bit-identical hashes to `tp2-direct`
    (`7278cd22ed16` / `1dd5b1dbdb63`).
  - Matrix (`manifests/benchmarks/2026-11-26-spark-ds41f-1126-pp2tp2ring.json`,
    [`receipts/benchmarks/pp2-tp2-ring-b2.json`](receipts/benchmarks/pp2-tp2-ring-b2.json)):
    `LRU 5/5`, `prose-c1..c16` `48.1–48.3 / 85.3–86.9 / 142.9–146.5 / 238.2–256.0 / 421.1–433.2 tok/s`,
    `code-c16` `438.8–449.4 tok/s`, prefill `32K..1M` `6,173–6,513` / `6,059–6,084 tok/s`.
- **`tp3-triangle` (`b1` + `b2` on recabled `dgx1..dgx3`, clean reboot before each boot, vs `1120`)**:
  - Pool logs ([`b1`](receipts/tp3-triangle/startup-pool-mem.txt), [`b2`](receipts/tp3-triangle/startup-pool-mem-b2.txt)):
    all 3 ranks kept `96.5 GiB` contiguous CUDA pool (`front_clean=True`), skipping buddy sponge on
    both main (`98,297.9 MiB` free for `95,092.9 MiB`) and draft (`5,179–5,382 MiB` free for
    `2,542.7 MiB`) weights, plus `905.6 MiB` display carve-out; `min MemAvailable 6.16–9.62 GiB`,
    `0.0 GiB` swap.
  - Wide determinism ([`logprobs`](receipts/tp3-triangle/determinism-wide.jsonl),
    [`no-logprobs`](receipts/tp3-triangle/determinism-wide-nologprobs.jsonl)): `identical: true`,
    `0` token diffs, `0` logprob diffs (`5c22ce41c1b5` / `bce6092e0128` / `26a739a99bca` /
    `12cfee9b740e`).
  - `tool_gate` ([`receipts/tp3-triangle/tool-gate.txt`](receipts/tp3-triangle/tool-gate.txt)):
    **`80/80` strict PASS (`0` derailments)** across `T=0.0, 0.3` on stock and `weightless`.
  - Sonar verdicts (`manifests/suites/2026-11-26-spark-ds41f-1126-tp3/`): `grid -25.19% ±0.10%`
    (`better`), paired `c8 +0.77%` (`better`), `c4 +1.34%` (`better`), `single-stream c1 +1.30%`
    (`71.32 tok/s`, `better`), `sampled c1 +1.39%` (`better`), `sampled c8 +0.00%` (`same`),
    `sampled c16 +31.53%` aggregate output (`220.0 tok/s`, `-96.5%` TTFT from `max_num_seqs
    8 -> 16`), prefill `131K..1M +1.06%..+2.86%` (all `better`; `32K +0.78%` `same`).
- **`tp3-triangle-4k` (`b1` + `b2` on recabled `dgx1..dgx3`, clean reboot before each boot, vs `1120`)**:
  - Wide determinism ([`logprobs`](receipts/tp3-triangle-4k/determinism-wide.jsonl),
    [`no-logprobs`](receipts/tp3-triangle-4k/determinism-wide-nologprobs.jsonl)): `identical: true`,
    `0` token / `0` logprob diffs, bit-identical hashes to `tp3-triangle`.
  - Matrix (`manifests/benchmarks/2026-11-26-spark-ds41f-1126-tp34k.json`): `LRU 5/5`, prefill
    `32K / 131K / 262K` **`8,337–8,377 / 8,417–8,471 / 8,363–8,431 tok/s`** (`+22.0%..+24.2%`),
    all decode points `same`/`faster`, `min MemAvailable 11.30–13.33 GiB`, `0.0 GiB` swap.
- **`tp2-direct` (`b1` + `b2` on `dgx1`+`dgx2`, clean reboot before each boot, vs `1120`)**:
  - Pool logs ([`b1`](receipts/pool-logs/tp2-direct-b1-pool.log), [`b2`](receipts/pool-logs/tp2-direct-b2-pool.log)):
    resized `95.5 -> 10.5 GiB`, skipping buddy sponge on both main and draft weights, `min
    MemAvailable 9.54–11.65 GiB`, `0.0 GiB` swap.
  - Determinism ([`receipts/determinism/tp2-direct-nologprobs.jsonl`](receipts/determinism/tp2-direct-nologprobs.jsonl)):
    `identical: true`, `0` diffs (`7278cd22ed16` / `1dd5b1dbdb63`).
  - Matrix (`manifests/benchmarks/2026-11-26-spark-ds41f-1126-direct2.json`,
    [`receipts/benchmarks/tp2-direct-b2.json`](receipts/benchmarks/tp2-direct-b2.json)):
    decode **`+41.0%..+75.5%`** (`prose-c4 92.7–92.9`, `code-c4 93.5–94.3 tok/s`), prefill
    `32K 4,646–4,668` (`+1,695%..+1,704%`), `131K 4,873–4,888` (`+1,968%..+1,975%`).
- **`tp1` (`b1` on `dgx3`, `b2` on `dgx4`, clean reboot before each boot, vs `1120`)**:
  - Pool logs ([`b1`](receipts/pool-logs/tp1-b1-pool.log), [`b2`](receipts/pool-logs/tp1-b2-pool.log)):
    resized `95.5 -> 20.5 GiB`, skipping buddy sponge on both main and draft weights, `min
    MemAvailable 9.07–9.17 GiB`, `0.0 GiB` swap.
  - Determinism ([`tp1-b1`](receipts/determinism/tp1-b1-nologprobs.jsonl), [`tp1-b2`](receipts/determinism/tp1-b2-nologprobs.jsonl)):
    `identical: true`, `0` diffs, bit-identical across `dgx3` and `dgx4` (`91bbfef57f00` / `d0dae1daa027`).
  - Matrix (`manifests/benchmarks/2026-11-26-spark-ds41f-1126-tp1.json`,
    [`receipts/benchmarks/tp1-b2.json`](receipts/benchmarks/tp1-b2.json)):
    decode **`+74.6%..+177.2%`** (`prose-c1/c2/c4 9.2–9.3 / 15.8 / 30.2–30.3`, `code-c4 28.2–28.4 tok/s`).
  - **Recorded unfavourable attempt (preserved):** on `dgx3`'s very first-ever cold `tp1` boot
    (`20261126T032306Z`), compiling all `488` cold TileLang kernels in-process (`370.8 s`) left
    `AnonPages = 9.77 GiB` right when the `6.0 GiB` KV cache allocated, briefly dipping
    `MemAvailable` to `4.67 GiB` (`340 MiB` below the `5.0 GiB` startup memguard) after all `488`
    kernels were cached; subsequent rebooted warm-JIT boot on `dgx3` succeeded cleanly with
    `min MemAvailable 9.17 GiB` and `0.0 GiB` swap.
