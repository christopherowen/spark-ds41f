# MoE SwiGLU: one kernel over the live tiles

Owner, 2026-10-14: set up as an experiment, one piece per branch.

## Why

Between the routed experts' gate/up and down GEMMs, TileKernels'
`swiglu_forward_and_per_token_cast` applies DeepSeek's clamp, the SwiGLU and the
route weight, and casts the down projection's input to MXFP8. At decode it takes
23-30 µs a layer whatever the row count. The shared expert's SwiGLU
(`activation_swiglu_clamp_kernel`) takes 2.4-3.1 µs on the same steps.

TileKernels' kernel launches one 1,024-thread CTA per SM (48; 45 do work). Each warp walks the expert
boundaries in `psum` one expert at a time to skip the alignment padding, inside a
loop bounded by the layout's capacity. That serial walk over 384 experts is the
likely cost at small row counts; the kernel bench confirms or refutes it.

## Ceiling, from existing traces

Source: the 2026-10-13 fixed-row profile
([tilelang-wo](../2026-10-13-tilelang-wo/README.md), run `fused-fixed`, arm `fused`:
the r6j image with r6k's fused WO cast; r6k does not change the MoE), rank 0.
Adaptive verification was off, so every step verifies the same rows whatever the
text. [swiglu_gap.py](swiglu_gap.py) pairs each SwiGLU launch with its GEMMs.
"Serial" is the time from the gate/up GEMM's end to the down GEMM's start.

| Workload | Routes (gate/up tiles) | SwiGLU | Serial | Per target step (40 layers) | Step | Share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| six rows (one stream; 36-route calls in the c8 window) | 36 | 22.9 µs | 23.6 µs | 0.94 ms | ~31.5 ms | 3.0% |
| c8, 48 rows | 288 | 23.7 µs | 24.4 µs | 0.98 ms | 31.67 ms | 3.1% |
| code16, 96 rows | 402 tiles | 29.2 µs | 30.0 µs | 1.20 ms | 45.30 ms | 2.6% |
| prefill, 8,096-row chunk | 1,902 tiles | 607 µs | 613 µs | 12.9 ms a chunk (20 sequence-parallel layers + 20 at 128 rows) | ~1.40 s | 0.9% |

The kernel never overlaps its neighbours: it starts 0.2-3 µs after the gate/up
GEMM ends. Its time is serial in the step.

Bound: a kernel near the shared expert's 3 µs saves 20-27 µs a layer, **about
2.4-2.7% of a decode step at one to sixteen streams**. The drafter's MoE layers
call the same kernel (about 10 µs a call at 40-80 draft rows), which adds a little. The one-stream
row is taken from six-row calls inside the c8 window; window 1 profiles one
stream directly.

At an 8,096-row chunk the kernel moves about 141 MB (the BF16 gate/up output in,
MXFP8 and scales out) at about 230 GB/s, close to the memory system's rate. A
standalone kernel cannot gain there and must not lose. Fusing the SwiGLU into the
gate/up GEMM's epilogue would remove that traffic as well (about 0.9% of
prefill). That is a second step, taken only if this one lands.

## The change (planned: vLLM branch `r6k-moe-swiglu`, on r6k's tree)

- **Module.** A TileLang kernel in `vllm/models/deepseek_v4_1/tilelang/moe.py`
  that reads the dispatch layout's `tile_expert` and `num_tiles` and runs one
  CTA per live 32-row tile and 64-column block. Each CTA:
  - loads its rows' gate and up columns of the BF16 gate/up output;
  - applies TileKernels' arithmetic element for element: the clamp at
    `swiglu_limit` (10.0), the SwiGLU in FP32, the route weight from
    `layout_weights`;
  - computes each 32-column block's UE8M0 scale with `round_sf`;
  - writes MXFP8 and packed scale words into the existing down-input buffers.

  Pad rows are skipped, as TileKernels skips them; the down GEMM's pad outputs
  are never combined.
- **Switch.** The routed experts call the new kernel under the TileLang family.
  When the port passes, the TileKernels call goes (the migration's rule: no
  fallback arm stays).

## The NaN clamp (owner, 2026-10-14)

TileKernels' kernel clamps with `T.min`/`T.max`, which lower to `fminf`/`fmaxf` and turn a
NaN into the limit. DeepSeek's reference clamps with `torch.clamp`, which keeps it
([kernel-contracts](../2026-10-14-kernel-contracts/README.md), the strict `xfail` on the
routed experts). The port clamps by compare-and-select, as the shared expert's fix (vLLM
kernel-contracts 0004) does:
- on finite inputs the result equals `min`/`max`, so the bit-identity gate below still
  holds;
- a NaN gate or up value now passes through.

This settles the TileKernels decision without patching TileKernels. Once the port passes,
the routed-expert contract changes from `xfail` to a required pass. Keep the clamp
expressions pure (no reassigned `alloc_var` scalars), and check TileLang's output for
the serial-loop warning: the indexer's NaN fix lost its vectorization that way.

## Why the bits do not change

Each output depends only on its row's gate and up values, its route weight and
its 32-column block's maximum. The port reproduces TileKernels' operations in
the same order, including its exponential and its UE8M0 rounding path.
Bit-identity against TileKernels' kernel is the gate, not an assumption.

## Tests (kernel bundle, planned)

- Bytes and scale words equal TileKernels' kernel at every serving shape:
  - decode, 1-96 rows (6-576 routes), and the drafter's 128-expert top-3 routes;
  - prefill, 205-8,192 rows;
  - real dispatch layouts from uniform and skewed routings (few experts, one
    expert), with clamped values and route weights present.
- Repeatable under CUDA graph replay. Row-count invariance: a route's bytes do not
  depend on the other routes.
- Timed under CUDA graphs at those shapes, warm and cold in L2. It must beat
  TileKernels' at every decode shape and be level at 8,096 rows.

## Arms and windows (planned)

- Window 1: the kernel bundle on dgx1 (tests and timing), which decides whether
  a port can reach the ceiling; and a `profile` job of r6k with the `decode`
  workload, to measure the one-stream share directly.
- Window 2: `swiglu.json` (the changed files over the r6k image) in a lean
  screen, candidate first, bracketed by r6k, with `token_determinism.py --wide`,
  distinct prompts and mixed traffic; then fixed-row profiles (code16, c8,
  prefill) of both arms, read with `swiglu_gap.py`.

## Acceptance

- The bundle's bytes are identical, and the temperature-0 references are r6k's.
- The decode step is faster at one to sixteen streams, and prefill is level.
- The serial gap between the routed GEMMs falls toward the shared expert's
  SwiGLU time in the profile.

## Status

Planned 2026-10-14. The ceiling is measured from existing traces. Nothing is
built or run.
