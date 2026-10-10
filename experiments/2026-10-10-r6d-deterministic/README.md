# r6d candidate: temperature-0 determinism and the first TileLang ports

Base: r6c ([2026-10-08-lil-rebase](../2026-10-08-lil-rebase/)), accepted by the
owner on 2026-10-08, on Local Inference Lab vLLM `19f2c20e` and B12X `236ddff0`.
r6d keeps every pin and adds the TileLang migration's ports that passed
([2026-10-09-tilelang-migration](../2026-10-09-tilelang-migration/)): image
`vllm-ds41f-kkref:19f2c20ed4d6-r6d`, vLLM patch head `706760d0`, tree `a00a6b07`.

Owner, 2026-10-10: correctness before speed, then meet or beat the current
kernels, then remove the engines TileLang replaces. After tuning C3 to its
floor, the owner chose to benchmark this candidate first (option 1), then to
overlap the reduce-scatter with compute (2), then to fuse its sum (3).

## What r6d changes

The series is r6c's 39 patches (0001-0047) and 33 from the migration's vLLM branch
(`tilelang-migration`):

- **0048-0072, modules.** Kernels, refactors that compile to the same code and
  imports; they change nothing the model runs. Among them are kernels whose
  ports have not passed yet (B1 WO projection, B6 DSpark context KV, B3 cache
  writers): present, not called.
- **0073-0080, switches.** Each routes one part of the model to TileLang under
  the TileLang kernel backend:

| Patch | Port | What changes | Window evidence |
| --- | --- | --- | --- |
| 0073 | B5 compressor projection | B12X `bf16_gemv` (SIMT, cuBLAS or TMA by row count, different bits) -> one-launch split-K TileLang GEMM, the same bits at every row count | outputs fixed for staggered arrivals; level |
| 0074 | B7 Engram gate | B12X `run_engram_mix` -> TileLang with DeepSeek's arithmetic | level |
| 0075 | B2 RoPE | B12X `rotary.rotate` -> TileLang in place (TileKernels arithmetic) | step -0.8 to -1.4%, 16K prefill +2.7% |
| 0076 | B9 Engram hash | vLLM metadata + B12X Triton hash -> one TileLang launch | level |
| 0077 | B8 index weights | projection + B12X scale -> the power-of-two scale folded into the weight (same bits) | level |
| 0078 | D1 decode rows | decode tiles and split-K for 16-stream steps (65-128 rows) on the production TileLang projections (same bits) | c8 +1.3%, c16 +0.7% |
| 0079 | C3 collectives | sequence parallelism's reduce-scatter with the one-shot all-reduce's arithmetic (unreduced exchange, FP32 rank-order sum) | outputs fixed for c8 and mixed steps; prefill -2 to -4% |
| 0080 | aligned prefill chunks | prefill chunks end at multiples of the long-prefill threshold | outputs fixed across chunk boundaries; level |

With 0073, 0079 and 0080 together, each prompt's tokens and logprobs at
temperature 0 are bit-identical alone, among eight streams, arriving
staggered, beside a 10,000-token prefill that crosses a chunk boundary, from
the prefix cache and run to run (migration windows 5 and 6). r6c changes every
prompt in the first three.

## Recipe changes

[r6d-tp4.json](r6d-tp4.json) and [r6d-tp3.json](r6d-tp3.json) are r6c's with the
image, its vLLM tree label, this lock, and two settings C3 and the aligned
chunks need:

- `SPARKNET_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES=2097152`: the one-shot all-reduce
  up to its registered capacity, so every step below sequence parallelism's
  205 rows reduces one-shot (r6c sent 103-204-row steps to NCCL).
- `--long-prefill-token-threshold`: the step budget less the decode and draft
  rows of every stream (TP4 8192 - 16 x 6 = 8096; TP3 4096 - 8 x 6 = 4048), so a
  whole aligned chunk always fits beside the decodes. The scheduler checks this
  at start (0071) and refuses a recipe where it does not hold.

## Checks so far (CPU)

- The series applies to `19f2c20e` with `git am` as `build prepare` runs it;
  r6c's head `125c404e` is reproduced on the way, and `build prepare --only
  vllm` reproduces the recorded head `706760d0` and tree `a00a6b07` with `git
  diff --check` clean. The tree equals the migration branch's candidate.
- Every switch was screened in the migration windows on the r6c image with the
  modules mounted; together (`deterministic`: C3, B5, aligned chunks) in
  windows 5 and 6.
- The repository's unit tests pass; `doctor` accepts both recipes.

## Gate ([gate.json](gate.json), one lab window on TP4)

1. Build `-r6d` on dgx4 (its Docker cache holds r6c's layers) and copy it over
   the CX7 links (dgx4 -> dgx1, dgx4 -> dgx3, dgx1 -> dgx2) while the lab LAN
   switch is heat-soaked; one image ID on all four nodes.
2. In the image, with `--noconftest`: r6c's gate tests (TileLang router,
   adaptive verification, acceptance estimator, dead rows) and the tests of
   every port r6d switches on (`bundles/gate-tests`).
3. r6c, then r6d: the quality suite and r6's acceptance benchmark (decode
   prose and code at 1-16 streams, prefill 32K-1M), and the temperature-0
   output check.

TP3 needs the triangle cabling; [r6d-tp3.json](r6d-tp3.json) is ready for it.

## Acceptance criteria

- Temperature-0 outputs identical in every scenario of the output check.
- Quality 5/5, every test passes, no failed requests.
- Decode level with r6c or better (step-time intervals including zero or
  favouring r6d).
- Prefill within C3's measured cost: on ring4 an exact rank-order reduce-
  scatter carries a third more wire bytes than NCCL's ring, about 2-3% of
  prefill in the migration windows, partly offset by B2 and B5.

Rollback trigger: a test, quality or request failure, a temperature-0
difference, a decode point slower than r6c with an interval excluding zero, or
prefill slower than r6c by more than 4% at any size.
