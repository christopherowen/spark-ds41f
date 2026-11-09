#!/bin/bash
# #1001's tests again with the release bundle's tblib stand-in (the image lacks tblib,
# which tests/conftest.py imports; the first pytest runs stopped there, exit 4).
B=$HOME/spark-lab/agents/candidate-51-own-marks; OM=$B/bin/om.sh
for k in 1 2; do
  $OM 1001 pytest-stub-c51-boot$k c51 -e PYTHONPATH=/opt/spark-ds41f/candidate/vllm:/h -- pytest -p tblib_stub -v \
    tests/kernels/quantization/test_deepseek_v41_tilelang_vocab.py \
    tests/kernels/test_deepseek_v41_tilelang_pdl.py \
    tests/v1/worker/gpu/test_greedy_verification.py
done
