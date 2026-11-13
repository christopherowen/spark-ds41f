#!/bin/bash
# #963: h963.py (the timing half of the PR's screen_pack12_kernels.py on the image's own
# kernels, plus graph replay, kernel time, the certified screen and the split-K compressor
# GEMM), ABBA in one session, then the PR's tests on c51 (with the release bundle's tblib stand-in), two boots.
B=$HOME/spark-lab/agents/candidate-51-own-marks; OM=$B/bin/om.sh; mkdir -p $B/out/pr-963/json
for spec in "base 1" "c51 1" "c51 2" "base 2"; do set -- $spec
  $OM 963 pack12-$1-boot$2 $1 -- python3 /h/h963.py /out/json/pack12-$1-boot$2.json
done
for k in 1 2; do
  $OM 963 pytest-c51-boot$k c51 -e PYTHONPATH=/opt/spark-ds41f/candidate/vllm:/h -- pytest -p tblib_stub -v \
    tests/kernels/quantization/test_deepseek_v41_tilelang_vocab.py \
    tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py \
    tests/kernels/test_deepseek_v41_tilelang_embedding.py \
    tests/kernels/test_deepseek_v41_tilelang_pdl.py
done
