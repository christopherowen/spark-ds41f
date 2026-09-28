# CuTe DSL 4.7.1

Base: the promoted configuration (`2026-09-28-karmic-kraken-r5h`).

## Question

The image build installed B12X's exact CuTe DSL pin (4.6.2) over the 4.7.1
the vLLM nightly base ships and vLLM requires. That broke quack-kernels 0.6.5
(it imports `cutlass.base_dsl.enums`, new in 4.7), so vLLM's DS4.1 L2 weight
prefetch failed to compile at every startup and disabled itself:

    [l2_prefetch] disabled: kernel compile failed: No module named 'cutlass.base_dsl.enums'

Shared runtime packages move forward: does B12X run correctly and as fast on
CuTe DSL 4.7.1 (B12X patch 0002 moves its pin), and what does the L2
prefetch, now able to run, do to decode?

## Arms

| Arm | Change from the promoted configuration |
|---|---|
| `control` | none (r5h, CuTe DSL 4.6.2, prefetch unable to compile) |
| `candidate` | image r5i: CuTe DSL 4.7.1, B12X patch 0002; the prefetch runs (default on SM121) |
| `candidate-nol2` | r5i with `VLLM_DS41_L2_PREFETCH=0` |

## Workload and gates

1. `b12x_tests.sh`: the B12X tests covering the DS4.1 kernels, on the r5h and
   r5i images (one node, cluster stopped). A test that passes on r5h must pass
   on r5i.
2. `sequence.sh`: lean screen, ABC twice (candidate first, since its first
   boot compiles every B12X kernel for the new CuTe version): LRU gate, decode
   at one and eight streams on prose and code with and without reasoning.

## Results

Pending.
