#!/bin/bash
# 0795-r1 (final) on candidate-54's image: the compressor test (TP1 identity at TP2..TP8) and the
# L2 schedule test (COMP's own budget); the same two files on candidate-54 itself; then the
# testbed profile's code (candidate-56 image + candidate-54 + 0795-r1 + 0796) with #956 round 3's
# test files: compressor (incl. test_sp_ratio2_projection_matches_full_rows), compressor_prepare,
# ced_model, and the L2 schedule test.
W=$HOME/spark-lab/agents/candidate-56-rework-957p1; OM=$W/bin/om.sh
for spec in "r1 2" "base 2"; do set -- $spec
  $OM pytest-$1-boot$2 $1 -- pytest --noconftest -o cache_dir=/tmp/pc -q \
    tests/models/test_deepseek_v4_1_compressor.py tests/models/test_deepseek_v4_1_l2_schedule.py
done
$OM pytest-c56r1-boot1 c56r1 -- pytest --noconftest -o cache_dir=/tmp/pc -q \
  tests/models/test_deepseek_v4_1_compressor.py tests/models/test_deepseek_v4_1_compressor_prepare.py \
  tests/models/test_deepseek_v4_1_ced_model.py tests/models/test_deepseek_v4_1_l2_schedule.py
