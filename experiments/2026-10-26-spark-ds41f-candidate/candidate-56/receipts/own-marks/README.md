# candidate-56 own marks (sp8, 2026-11-10)

#957 part 1 (`0795`) was measured in one session, ABBA, against the base: candidate-54's tree on candidate-54's image. The candidate arm is the same image with `0795`'s three files mounted. Both arms run on one image, so the build host is not a variable. Decode is judged by CUDA-graph replay at TP4 rank shapes.

| PR | Verdict | Evidence |
| --- | --- | --- |
| #957 (part 1) | miss | Projection cells: with L2 hot, 3 us a step lower at 1-6 rows and 2-8 us higher from 16 rows; with L2 cold, 226-238 us higher at every row count. The PR's 2x is an eager loop's launch cost. The one-GPU decode step at TP4 rank shapes is 0.23-0.35 ms slower at 1-4 requests and the same at 16. Outputs are bit-identical to TP1, at +23.4 MiB a rank |

**Cause.** The loss sits in the three ratio-2 owner layers (2, 8, 14), at +77-85 us each by nsys:
- the COMP L2 window prefetches the replicated 10.5 MB weight with the WO budget, where the base's shard is 2.0 MB;
- its fill cannot land before the compressor's GEMM, so the GEMM reads from DRAM against the fill and the 18.9 MB WO fill queued behind it;
- COMP plus WO (29.4 MB) no longer fits the 24 MB L2.

Dropping only the COMP window turns the one-GPU loss into a 0.15-0.44 ms gain.

One GPU shows about a fifth of the sp1 grid's +1.6-1.8 ms. The rest needs the testbed, where the prefetch stream overlaps the RoCE all-reduces and sits near the critical path at small batch (the TP3 attribution).

**Fix direction.** Give COMP its own budget, defaulting to the sharded size or 0 for a replicated weight. No existing setting does this alone: `VLLM_DS41_L2_PREFETCH_WO_MB` caps WO and COMP together. The TP3 caps (`VLLM_DS41_L2_PREFETCH_WO_MB=15`, `_FFN_MB=12`, `_NEXT_MB=12`) leave COMP whole, so whether they recover TP4 needs a tp4-switched grid run.

Predicted serving share: about -4.5% at c1 if the testbed's grid loss carries to serving (-0.6 to -0.9% from the one-GPU part alone), about 0 at c16. Details are in `pr-957/result.json` (`cause`, `fix_direction`).

`common/` holds the setup shared with candidates 55 and 57: `setup.txt`, the run and arm scripts, and tree verification. Every arm's container sees its expected tree exactly.

The ABBA halves of the projection bench ran 49 minutes apart: base boot 1 at 06:12Z, then the GPU lock went to other agents' jobs, then the candidate boots and base boot 2 at 07:01-07:06Z. The two base boots agree within about 4% in every 4-layer step cell except one 12-row graph-hot cell (35.7 vs 38.9 us). Each run's times are in `pr-957/raw/commands-sp8.txt`.
