# TileLang router: TileKernels' top-k gate

Base deployment commit: `f5c41f3` (the [TileLang TP4 candidate](../2026-10-03-packed-bf16-head/tilelang-tp4/README.md)
with the packed BF16 head).

In the TileLang kernel family, the target's MoE layers routed through vLLM's
Triton `dsv4_topk` and the drafter's through a CUDA op. DeepSeek publishes a
TileLang router: TileKernels' `moe_topk_gate_forward`, already in the image
(TileKernels 2.0.0). It implements the V4.1 reference routing:

- scores `sqrt(softplus(logits))`, with softplus the identity above 20;
- the top-k of score plus the correction bias, or the vision bias for image
  tokens, with the lowest expert first on ties;
- the chosen scores without bias, divided by their sum plus 1e-20 and scaled
  by the routed scaling factor.

It refuses non-finite logits. Triton's kernel differs in its exponent and
division arithmetic.

## Change

vLLM patch [0032](vllm/0032-deepseek-v41-tilekernels-router.patch), on 0029:

- `TileKernelsRouter` routes both the target (384 experts, top 6) and the
  drafter (128, top 3).
- The kernel family picks it through `kernels.moe_router_class()`, and
  `DeepseekV4MoE` passes it to the fused-MoE factory. B12X keeps vLLM's router,
  and MegaMoE refuses one.
- The image-token mask is computed once per step and shared by every layer.
- The ids are int64, which the TileLang dispatch reads.

[router.json](router.json) is the candidate, launchable. Against the
[packed arm](../2026-10-03-packed-bf16-head/tilelang-tp4/packed.json), whose
reports it is compared with, only the vLLM tree (this patch) and the B12X tree
differ. The B12X difference is the fused packer of `candidate.json`, which
changes load-time memory, not speed. The arm keeps the packed arm's DSpark
cost directory, since routing changes no shape.

## Kernel results

The [router tests](bundles/router-tests/candidate.json) pass 29 of 29. TileKernels'
expert ids are identical to a torch rendering of DeepSeek's reference, and
its scores are bit-identical to torch's `sqrt(softplus)`. They were run on the
TileLang image with the patch's files overlaid, at decode and prefill row
counts, for both routers. The weights differ only in the six-term sum's
rounding. Exact ties go to the lower expert id, where `torch.topk` leaves the
order unspecified.

[Per call](bundles/router-bench/candidate.json), under CUDA graphs:

| Rows | TileKernels, 384 experts top 6 | Triton `dsv4_topk` | TileKernels, 128 top 3 | CUDA op |
| ---: | ---: | ---: | ---: | ---: |
| 1–96 | 4.31–4.50 µs | 2.05–2.26 µs | 1.96–2.05 µs | 1.95 µs |
| 512 | 4.82 µs | 3.25 µs | 2.21 µs | 2.15 µs |
| 8,192 | 33.41 µs | 28.88 µs | 11.17 µs | 12.21 µs |

For the target, TileKernels costs about 2 µs more per call. That is about
90 µs per decode step over 40 layers, or 0.3%. Its non-finite checks take
0.3–0.7 µs of that, and PDL recovers 0.2 µs. The rest is its argmax: ten
shuffles per round. The next step folds the gate projection's split-K reduce
into the router, which removes one launch and the FP32 logits round trip
from every MoE layer.

## Serving results

Window 2026-10-04 12:58–13:06 UTC, against the packed arm's reports:

| | Router | Packed arm | Change |
| --- | ---: | ---: | --- |
| prose, 1 / 8 streams (tok/s) | 57.5 / 213.1 | 62.3 / 213.6 | −7.6% / −0.2% |
| code, 1 / 8 streams | 74.7 / 248.4 | 80.8 / 247.5 | −7.6% / +0.3% |
| Single-stream step time, prose / code (ms) | 37.79 / 39.29 | 33.58 / 36.73 | +12.6% / +6.9% |
| Prefill 1K / 32K / 64K / 256K (tok/s) | 2,586 / 4,895 / 4,906 / 4,652 | 2,575 / 4,885 / 4,908 / 4,651 | level |

Quality passed 5/5, and every node kept at least 29.9 GiB available.
Eight-stream decode and prefill are level. Single-stream step time is
4.2 ms longer, far more than the router's 2 µs per call (about 0.1 ms per
step). The packed arm's reports come from another day's boot, and the
TileLang family's routed experts have varied 16.15–18.86 ms across boots.
The [gate router](../2026-10-04-tilelang-gate-router/README.md) window
reruns the packed arm beside this one, with a decode profile of each.

## Procedure

Build `-tilelang-router-v1`, run both bundles, then one window with the
router arm. It runs decode (prose and code at 1 and 8 streams, three samples)
and prefill (1K, 32K, 64K and 256K, two repeats), against the packed arm's
reports, with per-second memory logs. Quality, DSpark acceptance and step
time show whether the routing change moved anything.
