#!/usr/bin/env bash
# PR #991 rework r1: two boots of the corrected gemm.py (+ its test) bind-mounted over
# spark-ds41f:candidate-49. Per boot, each in its own container: the README section 4
# pytest, the PR bench unchanged, its CUDA-graph companion, CUPTI kernel durations.
. $HOME/spark-lab/agents/candidate-49-rework-991/env.sh
G=$W/variants/r1/gemm.py
TL=$W/variants/r1/test_deepseek_v41_tilelang_linear.py
MG=$V/vllm/models/deepseek_v4_1/tilelang/gemm.py
MT=$V/tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py
J=$W/out/json; mkdir -p $J
K="test_lsa or test_block_scale_words or test_mxfp8_gemm_rope_mxfp4_decode_matches_reference or test_hw_e2m1x2_codegen_in_gemm_and_indexer_quantizers or test_mxfp4_quantize or test_write_index_keys or test_norm_rope_write_index_keys or test_indexer_dense or test_indexer_candidates"
COMMON="-v /usr/local/cuda-13.0/bin/nvdisasm:/usr/local/bin/nvdisasm:ro -v $G:$MG:ro -v $TL:$MT:ro"
for boot in 1 2; do
  $W/run.sh pytest-r1-boot$boot $C -- flock $L docker run --rm --gpus all --ipc=host --name rw991-pyt-b$boot --entrypoint pytest $COMMON -w $V $C --noconftest -o addopts= -o cache_dir=/tmp/pytest_cache -v -k "$K" tests/models/test_deepseek_v4_1_lsa.py tests/kernels/quantization/test_deepseek_v41_block_scale_words.py tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py tests/kernels/attention/test_deepseek_v41_tilelang.py
  $W/run.sh bench-r1-boot$boot $C -- flock $L docker run --rm --gpus all --ipc=host --name rw991-bench-b$boot --entrypoint "" $COMMON -v $EXP/bench_hw_e2m1x2_quant.py:/tmp/bench_hw_e2m1x2_quant.py:ro -v $J:/out -w $V $C bash -lc "python3 /tmp/bench_hw_e2m1x2_quant.py --out /out/bench-r1-boot$boot.json"
  $W/run.sh graph-r1-boot$boot $C -- flock $L docker run --rm --gpus all --ipc=host --name rw991-graph-b$boot --entrypoint "" $COMMON -v $EXP/bench_hw_e2m1x2_quant.py:/tmp/bench_hw_e2m1x2_quant.py:ro -v $W/scratch/bench_graph_companion.py:/tmp/bench_graph_companion.py:ro -v $J:/out -w $V $C bash -lc "python3 /tmp/bench_graph_companion.py --out /out/graph-r1-boot$boot.json"
  $W/run.sh cupti-r1-boot$boot $C -- flock $L docker run --rm --gpus all --ipc=host --name rw991-cupti-b$boot --entrypoint "" $COMMON -v $EXP/bench_hw_e2m1x2_quant.py:/tmp/bench_hw_e2m1x2_quant.py:ro -v $W/scratch/diag_variants.py:/tmp/diag_variants.py:ro -v $W/scratch/diag_cupti.py:/tmp/diag_cupti.py:ro -v $J:/out -w $V $C bash -lc "python3 /tmp/diag_cupti.py --out /out/cupti-r1-boot$boot.json"
done
