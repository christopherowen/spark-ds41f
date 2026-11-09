#!/bin/bash
# The PR's pytest suite on the r1 overlay, two boots.
cd $HOME/spark-lab/agents/candidate-47-rework-1016
for b in boot1 boot2; do
  ./om-pytest.sh pytest-r1-$b r1 -- pytest -v -p no:cacheprovider -p tblib_stub -p offline_skip \
    tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py \
    tests/kernels/quantization/test_deepseek_v41_block_scale_words.py \
    tests/models/test_deepseek_v4_1_attention.py \
    tests/models/test_deepseek_v4_1_l2_schedule.py \
    tests/kernels/test_deepseek_v41_tilelang_pdl.py
  echo "pytest $b rc=$?"; tail -1 out/pytest-r1-$b.txt
done
