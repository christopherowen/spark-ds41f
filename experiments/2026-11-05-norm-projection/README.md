# Norm/projection boundary screen

Base: candidate-38 at deployment commit `e9e0f09b15891ba4ab77c74e7e749e9f9be898b5`,
vLLM patch head `1f33102547e4da5c64712390604695d1abd577c4`, tree
`c061e5f6931306fa4655ba531428f639c365bf10`. Full identities are in `base.json`.

Question: does separating the attention front's normalization from its cast,
rotation and cache stores improve the following Q-B projection chain? The one
variable is the boundary: current fused front, fused Q norm/cast with standalone
KV operations, or the five standalone operations. All use the current plain Q-B GEMM.

The kernel screen uses rows 1, 6, 9, 16, 24, 48 and 96; Q-B at TP4 (8192 x 1280)
and TP3 (12288 x 1280), including the last TP3 rank's zero-weight padded heads.
Weights and activations are deterministic synthetic inputs of checkpoint shapes,
not model-quality evidence. Compare every output byte, including quantized values,
scales, rotated KV and untouched cache locations; check NaN/Inf contracts and row
invariance separately. The arithmetic remains norm in FP32, BF16 round, then
E4M3/UE8M0 quantization. No rescaling across the projection is allowed.

Measure CUDA-graph chains with weights resident in L2, after a 128 MiB eviction,
and after eviction racing the serving TileLang bulk L2 prefetch on another stream.
Keep all raw samples and two fresh process/container runs. These are kernel-lab
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

On the idle node, from a clean published checkout at this experiment's commit:

```
python3 scripts/lab.py kernel-local experiments/2026-11-05-norm-projection/bundle --out /home/swank/spark-lab/norm-projection/run-1
python3 scripts/lab.py kernel-local experiments/2026-11-05-norm-projection/bundle --out /home/swank/spark-lab/norm-projection/run-2
```

Status: prepared; no performance conclusion until the raw receipts exist.
