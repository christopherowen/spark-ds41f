#!/bin/bash
# Round 1 of #963 and #1001 on sp8: a smoke run of the new tests, the corrected arm's
# change markers, then each PR's bench ABBA in one session (base boot1, c51r1 boot1,
# c51r1 boot2, base boot2), then each PR's tests on c51r1, two container runs.
W=$HOME/spark-lab/agents/candidate-51-rework-963-1001; RW=$W/bin/rw.sh
mkdir -p $W/out/pr-963/json $W/out/pr-1001/json $W/out/pr-common
TB=(-e PYTHONPATH=/opt/spark-ds41f/candidate/vllm:/h)
$RW common smoke-new-tests-c51r1 c51r1 "${TB[@]}" -- pytest -p tblib_stub -v -x \
  tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py::test_packed_bf16_gemm_wide_and_narrow_grids_exact \
  tests/kernels/quantization/test_deepseek_v41_tilelang_vocab.py::test_screen_tile_config_128_uses_matched_two_stage_pipeline \
  tests/kernels/quantization/test_deepseek_v41_tilelang_vocab.py::test_tile_config_128_uses_matched_two_stage_pipeline || exit 1
$RW common verify-overlay-c51r1 c51r1 -- python3 /h/verify_overlay.py
for spec in "base 1" "c51r1 1" "c51r1 2" "base 2"; do set -- $spec
  $RW 963 pack12-$1-boot$2 $1 -- python3 /h/h963r1.py /out/json/pack12-$1-boot$2.json
done
for spec in "base 1" "c51r1 1" "c51r1 2" "base 2"; do set -- $spec
  $RW 1001 vocab-$1-boot$2 $1 -- python3 /h/warm.py /h/bench_vocab_r1.py /out/json/vocab-$1-boot$2.json
done
for k in 1 2; do
  $RW 963 pytest-c51r1-boot$k c51r1 "${TB[@]}" -- pytest -p tblib_stub -p tests.kernels.deepseek_v41_pdl_sources -v \
    tests/kernels/quantization/test_deepseek_v41_tilelang_vocab.py \
    tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py \
    tests/kernels/test_deepseek_v41_tilelang_embedding.py \
    tests/kernels/test_deepseek_v41_tilelang_pdl.py
  $RW 1001 pytest-c51r1-boot$k c51r1 "${TB[@]}" -- pytest -p tblib_stub -v \
    tests/kernels/quantization/test_deepseek_v41_tilelang_vocab.py \
    tests/kernels/test_deepseek_v41_tilelang_pdl.py \
    tests/v1/worker/gpu/test_greedy_verification.py
done
