# Norm/projection boundary screen

Base: candidate-38 at deployment commit `e9e0f09b15891ba4ab77c74e7e749e9f9be898b5`,
vLLM patch head `1f33102547e4da5c64712390604695d1abd577c4`, tree
`c061e5f6931306fa4655ba531428f639c365bf10`. Full identities are in `base.json`.

Question: does separating the attention front's normalization from its cast,
rotation and cache stores improve the following Q-B projection chain? The initial variable is the boundary: current fused front, separate fused Q/KV
kernels in either order, Q plus projection overlapped with fused KV, fused Q
norm/cast with standalone KV operations, or the five standalone operations.
A separate set of arms varies only CTA packing (64, 128 or 256 threads) in the
fused front, retaining 64 threads per Q row and 16 per KV row and their reduction
order. The derived 128-thread arm controls the experimental factory against the
unchanged serving kernel. All arms use the current plain Q-B GEMM.

`bundle/front_variants.py` is a copy of this exact locked source's front factory,
with the required license notices retained. Its only changes are the thread-count
parameter and Q/KV partition selection. It remains experiment-local. The parallel
split forks after the input is ready and joins after Q-B, including both branches
in the measured graph. The separate projection-only arm has precomputed inputs:
it estimates the removable boundary cost under each cache condition, but is not
an implementable candidate or a rigorous full-step bound.

The kernel screen uses rows 1, 6, 9, 16, 24, 48 and 96; Q-B at TP4 (8192 x 1280)
and TP3 (12288 x 1280), including the last TP3 rank's zero-weight padded heads.
Weights and activations are deterministic synthetic inputs of checkpoint shapes,
not model-quality evidence. Compare every output byte, including quantized values,
scales, rotated KV and untouched cache locations; check NaN/Inf contracts for both latents and row
invariance separately. Every timed row writes a live cache slot. Skipped-slot checks
run separately; inputs cover the first/last RoPE positions and cache page edges. The arithmetic remains norm in FP32, BF16 round, then
E4M3/UE8M0 quantization. No rescaling across the projection is allowed.

Measure CUDA-graph chains with weights resident in L2, after resetting persisting L2 priority in the benchmark context and a 128 MiB eviction,
and after eviction racing the serving TileLang bulk L2 prefetch on another stream.
Keep all raw samples and two fresh process/container runs. The prefetch arm races only the Q-B segments, not a reconstructed whole-model NEXT window. These are kernel-lab
process restarts, not serving boots. A winning screen would require a separate
authorized coordinated window, a candidate boot and repeat boot, fixed-row full
steps, c1/c8 (c16 on TP4), exact outputs and memory checks against recorded suite
runs before a runtime PR. Do not rerun a released build.

No serving operation is part of this screen. Use an idle dgx4, the shared hold on
dgx1, the lab GPU lock and the existing candidate-38 image by its recorded ID.
Start below 55 C, record memory and temperatures, and preserve unsuccessful runs.
No production recipe or published baseline is changed.

## Commands

Local: `bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-38/tp3.json doctor`;
fresh application of that lock's vLLM series from its pinned base;
`git diff --check`.

The remote work area is `dgx4:~/spark-lab/norm-projection`. Source preparation
uses one CPU and at most 4 GiB. Each GPU container is limited to two CPUs,
12 GiB RAM with no additional swap, 256 processes and one compilation worker.

On the idle node, from a clean published checkout at this experiment's commit,
`run_remote.py` claims the shared hold only after the existing holder releases it,
checks that dgx4 has no running container, cools it below 55 C, and runs:

```
python3 scripts/lab.py kernel-local experiments/2026-11-05-norm-projection/bundle --out /home/swank/spark-lab/norm-projection/checkout/experiments/2026-11-05-norm-projection/runs/run-1
python3 scripts/lab.py kernel-local experiments/2026-11-05-norm-projection/bundle --out /home/swank/spark-lab/norm-projection/checkout/experiments/2026-11-05-norm-projection/runs/run-2
```

The runner refreshes its hold every 30 seconds, records memory and temperature
every five seconds and the standard GPU clock/power/slowdown fields every fifteen seconds, enforces memory and thermal limits, releases the hold in `finally`, and never changes serving
state. Its labeled kernel container is stopped on interruption before its hold is released; loss of the heartbeat reservation terminates only that container. It gives up after one hour if the window stays occupied. Start it as a
bounded user service with `CPUQuota=200%`, `MemoryMax=4G`, `TasksMax=128`,
`RuntimeMaxSec=7200`, `TimeoutStopSec=45`; the Docker containers have their own limits above.

Validation so far: remote fresh series reproduces the exact locked patch head
and tree; doctor reports configuration OK; complete existing candidate-38 build
inputs pass `build check`; Python syntax and diff checks pass. The Mac attempts
that failed before recovery are retained in `runs/recovery.txt`.

The first GPU attempt failed during compilation, before timing or a complete exactness gate. See `runs/failed-01-compile` and its conditions/runner receipts. The split factory left inactive legacy-reducer declarations after constant folding; separate Q/KV factories now declare those unused buffers as ordinary fragments. Arithmetic is unchanged. The cold/hot/prefetch screen remains incomplete; no performance conclusion is claimed.

Additional audit: index-key and compression normalization reduce each live row once; Engram gating reduces each distinct token/stream residual and key once. These do not duplicate a hidden-row norm across projection output tiles.

Cache control follows the recorded GB10 L2-policy finding in `experiments/2026-10-25-moe-overlap`: evict-last lines survive an ordinary scrub even at zero set-aside. Cold and racing-prefetch trials therefore reset persisting priority before eviction; warm trials prime the exact graph immediately before timing. No device-wide setting or serving context is changed.

The split fix passes both the rejecting reducer pass and the full CUDA IR lowering pipeline without GPU access (`check_split_ir.py`, `runs/cpu-ir-check.txt`; one CPU, 4 GiB, no GPU device exposed). The two earlier CPU invocation failures (target argument and target context) are preserved separately. This is compilation evidence only. The waiting runner preserves foreign requests, queues ours only when the request file is free, and resumes a completed first run only at the identical published commit.

An independent user-service watchdog is scoped to the acquired hold's exact `since` value and `serving_untouched` marker. If the heartbeat becomes fifteen minutes stale, it stops only this runner and its labeled kernel container, verifies that container stopped, then releases only our hold. It exits on handoff and cannot restore or replace a serving cluster. Full serving tests, if needed after a positive screen, require the coordinated restore-profile watchdog workflow separately.

Capture, warm-up, KV overlap and racing prefetch each use an explicitly retained distinct CUDA stream, checked before the screen. This avoids relying on repeated stream-pool allocation for the overlap comparison.

`model-config.json` records the pinned checkpoint configuration. The attention RMSNorm epsilon is `1e-20` (`rms_norm_eps`), distinct from mHC's `1e-6`. The first failed compilation attempt used `1e-6` and produced no timings; the retry uses the checkpoint epsilon and prints it in its identity record. Synthetic values and the generated RoPE table remain explicitly separate from model-quality evidence.

## First completed process and pending repeat

At published experiment commit c51e824b876c45df79bca0c3c44415a4ed69f0b9,
the first process passed all 21 row/shape cases in 50.6 seconds. The full raw
samples, exact image ID and input hashes are in runs/run-1; process conditions
are in runs/run-records.json and runs/telemetry.jsonl. Peak allocated GPU
memory was 303,746,560 bytes; peak reserved was 400,556,032 bytes. Available system
memory stayed above 121 GB, the hottest recorded board sensor reached 54.5 C, and
the GPU thermal-slowdown counter did not increase. All earlier failures remain
recorded. No released build was remeasured and serving was unchanged.

This is incomplete screening evidence. A foreign request arrived after the first
process, so the runner yielded. Its subsequent generic acquisition waiter was
stopped to honor the explicitly reserved hardware order. The unchanged repeat
remains ready at the original clean checkout and exact c51e824b commit.
refinement/ready.txt records the command, limits and expected one-minute job
duration; its handoff order is a snapshot, not authorization to acquire a lease.
Always re-read the live hold, request and coordination records first.

The experimental 128-thread control and native front generate byte-identical
CUDA source (24,353 bytes, SHA-256
f19c497de4ea911ef0c76639231fd1fbb00d5ddef1c5cef55db8230a98b23deb).
Despite that equality, its first-process L2 timing differences range from
-5.20% to +7.82%. Some within-process intervals exclude zero for this identical
code, exposing systematic measurement effects that repeated samples alone do
not eliminate. No speedup, absence of regression, correctness repair or optimal
implementation is established by this first process.

refinement/bench.py is a prospective measurement revision, outside the active
bundle. It keeps independent-buffer correctness checks, then uses the same output
addresses for every timed arm and checks each call/replay against a frozen
independent oracle. It captures external timing events inside the CUDA graph so
host launch gaps are excluded. Its syntax is checked, but this revision has not
run on the GPU. The recorded image's torch.cuda.Event supports external=True.

Complete and preserve the unchanged second process first. Then publish and
validate the refinement as a separate two-process measurement campaign; archive
the original records and do not pool the two methods. Source alternatives and
GEMM arithmetic are unchanged. The racing-prefetch case is still an isolated
segment experiment, and neither it nor a kernel-only gain proves serving gains.
The full c1/c8/c16 and distinct-agent traffic, cold 32K–1M prefill, candidate boot
and repeat boot remain required wherever applicable before claiming a serving
improvement. No runtime/source PR or promotion is warranted by the current data.

## Completed two-process campaigns and conclusion

Both the unchanged second process at `c51e824b876c45df79bca0c3c44415a4ed69f0b9`
(`runs/original-c51e824b/`) and a separate two-process refined campaign at
`b20452fd18579902cf88e5c98e2ad0e73d0c39ff` (`runs/run-1`, `runs/run-2`,
`runs/refined-two-boot-summary.json`) completed on idle `dgx4` under a scoped
`dgx4:~/spark-hold.json` reservation (`dgx1:~/spark-hold.json` verified absent).
The two measurement methods are archived and analyzed separately without pooling.

1. **Unchanged two-process campaign (`c51e824b`, `runs/original-c51e824b/two-boot-summary.json`)**:
   - `run-2` passed all 21 cases in 26.0 seconds with bit-exact outputs.
   - Across both processes, the identical-code control (`fused128` vs native `fused`)
     ranged from `-3.31%` to `+8.12%` in L2 (`mean +0.75%, median +0.51%`), confirming
     that distinct output buffer addresses and host-side event recording caused up to
     `~8-12%` systematic timing shifts in L2.
2. **Refined two-process campaign (`b20452fd`, `runs/refined-two-boot-summary.json`)**:
   - `run-1` (26.0s) and `run-2` (27.0s) passed all 21 cases in both independent-buffer
     and shared-buffer oracle modes.
   - Capturing `external=True` CUDA events inside the graph and sharing output buffer
     addresses across timed arms collapsed the identical-code `fused128` L2 control
     band to `[-0.50%, +1.18%]` (`mean +0.33%, median +0.36%`, `0/21` cases 95% CI faster,
     `1/21` slower).
   - **CTA thread packing (`fused64`, `fused256`)**: `fused64` is flat against `fused`
     within the control band: L2 `mean -0.68%, median -0.69%` (`[-2.46%, +0.87%]`,
     `-0.02` to `-0.34 us` on a `10.8-44.5 us` chain; 15/21 95% CIs straddle zero),
     cold `mean -0.43%, median -0.27%` (`0/21` cases 95% CI faster), and prefetch
     `mean -0.03%, median +0.26%` (`1/21` faster, `1/21` slower). `fused256` regresses
     L2 by `+1.87%` mean (`10/21` cases 95% CI slower, `0/21` faster).
   - **Boundary splits and stream overlap (`split_qkv`, `split_kvq`, `split_parallel`, `split_q`, `standalone`)**:
     Every boundary split regresses L2 and cold execution because the entire fused
     attention front (`attention_front`: Q RMSNorm + BF16 round + MXFP8 cast + KV RMSNorm
     + RoPE + paged SWA cache write) adds only `1.09-2.76 us` (median `1.39 us`) over
     `projection_only` (`_block32_gemm`). Splitting Q and KV adds an extra kernel launch
     (`+6.39%` / `+5.73%` L2, `+1.55%` / `+2.12%` cold), dual-stream overlap (`split_parallel`)
     pays stream fork/join event overhead (`+2.09%` L2, `+1.33%` cold, `+2.65%` prefetch),
     and standalone PDL ops (`split_q`, `standalone`) add 3-4 launches (`+20.02%` / `+24.17%`
     L2, `+10.55%` / `+12.69%` cold). In the isolated prefetch race, multi-kernel fronts
     appear negative only because their slower front delays `_block32_gemm`, giving the
     concurrent side-stream weight prefetch extra head start (`projection_only` is
     correspondingly `+8.58%` slower in that synthetic race).

### Conclusion

Negative / flat hardware ceiling. The current single-kernel fused `attention_front`
already executes the entire Q+KV normalization, RoPE, MXFP8 quantization, and cache
write in `~1.1-1.8 us` on GB10, leaving no room to amortize extra kernel launches or
stream fork/join synchronization, and 64/128/256-thread CTA packing is flat within
`~0.1-0.3 us`. Keep the current promoted `attention_front` unchanged; no runtime or
vLLM patch is warranted.
